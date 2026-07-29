"""Building blocks shared by the early-exit Transformer.

Holds the pieces that are independent of how the decoder stack is assembled:
the key/value cache, normalization, rotary position embeddings, the exit
modules that turn a hidden state into logits, and the uncertainty measures that
decide when a token has been processed deeply enough.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Growable per-layer cache of attention keys and values.

    During autoregressive decoding every generated token attends to all of its
    predecessors. Recomputing their keys and values at each step is quadratic
    work, so they are stored here and extended one step at a time.

    The cache serves two execution styles, and the difference between them is
    the whole point of ``max_depth``.

    **Token-level early exit** runs every layer for the sequence as a whole,
    and fills the layers above a token's exit point by propagating that token's
    hidden state instead of running the block. Every layer ends up holding an
    entry for every position, so this style saves attention and feed-forward
    work but *not* cache memory. Leave ``max_depth`` unset for it.

    **Request-level depth capping** assigns one maximum depth to a whole
    request and never executes, or stores, anything above it. Memory then falls
    in proportion to the depth. Passing ``max_depth`` turns writes above the cap
    into errors rather than letting an upper layer be filled by accident, which
    is what keeps a claimed memory saving honest: a cache that silently
    materializes upper layers looks identical from the outside.

    Either way every populated layer holds the same number of positions, which
    is what lets :meth:`__len__` read layer zero alone.

    Args:
        n_layers: Number of decoder layers in the model.
        max_depth: Number of layers this cache is allowed to store, counted as
            executed blocks so layer indices ``0 .. max_depth - 1`` are
            writable. ``None`` means the full stack.

    Raises:
        ValueError: If ``n_layers`` is not positive, or ``max_depth`` falls
            outside ``[1, n_layers]``.
    """

    def __init__(self, n_layers: int, max_depth: int | None = None) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")

        depth = n_layers if max_depth is None else int(max_depth)
        if not 1 <= depth <= n_layers:
            raise ValueError(
                f"max_depth must lie in [1, {n_layers}], got {max_depth}."
            )

        self.n_layers = n_layers
        self.max_depth = depth
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

        #: Residual stream handed to block ``boundary_depth``, retained only
        #: when escalation may need to run upper layers over the same prompt.
        self.boundary: torch.Tensor | None = None
        self.boundary_depth: int | None = None

    def __len__(self) -> int:
        """Returns the number of tokens currently cached.

        Layer zero runs for every request at any depth of at least one, so its
        cache is never shorter than any other layer's and can stand in for all
        of them.
        """
        if self.keys[0] is None:
            return 0
        return self.keys[0].size(-2)

    @property
    def active_depth(self) -> int:
        """Number of layers this cache may store, in executed blocks."""
        return self.max_depth

    @property
    def seq_len(self) -> int:
        """Number of positions currently cached."""
        return len(self)

    @property
    def layer_presence(self) -> tuple[bool, ...]:
        """Whether each layer holds any entries, indexed by layer.

        Read this to check that depth capping did what it claims: a request
        routed to depth ``d`` must show ``False`` from index ``d`` upward.
        """
        return tuple(key is not None for key in self.keys)

    @property
    def bytes_allocated(self) -> int:
        """Total bytes held by the cached keys and values.

        Counts what is actually materialized rather than what a full-depth
        cache would have needed, so the number falls when depth capping works
        and does not when it only appears to.
        """
        total = 0
        for tensor in (*self.keys, *self.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    @property
    def boundary_bytes(self) -> int:
        """Bytes held by the retained boundary activation, if any."""
        if self.boundary is None:
            return 0
        return self.boundary.numel() * self.boundary.element_size()

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends a new key/value chunk and returns the full history.

        Args:
            layer_idx: Index of the layer whose cache is being updated.
            key: Keys for the incoming tokens, shaped
                ``(batch, n_kv_heads, new_len, head_dim)``.
            value: Values for the incoming tokens, same shape as ``key``.

        Returns:
            A tuple ``(keys, values)`` covering every token seen so far by this
            layer, each shaped ``(batch, n_kv_heads, total_len, head_dim)``.

        Raises:
            ValueError: If ``layer_idx`` lies at or above the depth cap.
        """
        self._check_layer(layer_idx)

        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat((self.keys[layer_idx], key), dim=-2)
            self.values[layer_idx] = torch.cat(
                (self.values[layer_idx], value), dim=-2
            )
        return self.keys[layer_idx], self.values[layer_idx]

    def _check_layer(self, layer_idx: int) -> None:
        """Rejects a write above the depth cap.

        Args:
            layer_idx: Layer being written.

        Raises:
            ValueError: If the layer is at or above :attr:`max_depth`.
        """
        if layer_idx >= self.max_depth:
            raise ValueError(
                f"Layer {layer_idx} lies above this cache's depth cap of "
                f"{self.max_depth}. A request routed to depth "
                f"{self.max_depth} must not materialize keys and values for "
                f"layers it never executes, since that is where the memory "
                f"saving comes from. To run deeper, escalate the request onto "
                f"a cache built with a larger max_depth."
            )

    def retain_boundary(self, hidden: torch.Tensor, depth: int) -> None:
        """Stores the activation an escalation would need to resume from.

        Continuing a request through upper blocks requires the residual stream
        those blocks would have received, for **every** prompt position, not
        just the last. Keeping it costs ``batch * seq_len * d_model`` elements
        and buys exact escalation with no recomputation of the lower prefix.

        Args:
            hidden: Residual stream leaving block ``depth - 1``, shaped
                ``(batch, seq_len, d_model)``.
            depth: Number of blocks already executed, so the retained state is
                the input to block ``depth``.
        """
        self.boundary = hidden.detach().clone()
        self.boundary_depth = depth

    def select_rows(
        self,
        index: torch.Tensor,
        max_depth: int | None = None,
    ) -> KVCache:
        """Extracts a subset of the batch into an independent cache.

        Request-level routing sends different rows to different depths, and a
        single batched cache cannot represent that. Rows are therefore split
        into one cache per depth bucket after the shared probe, reusing the
        probe's entries instead of recomputing them.

        Args:
            index: Row indices to keep, shaped ``(selected,)``.
            max_depth: Depth cap for the new cache. Defaults to this cache's
                own cap.

        Returns:
            A new cache holding copies of the selected rows. The copy is what
            makes the buckets independent; a view would alias the parent.
        """
        selected = KVCache(self.n_layers, max_depth=max_depth or self.max_depth)
        for layer in range(min(self.max_depth, selected.max_depth)):
            if self.keys[layer] is None:
                continue
            selected.keys[layer] = self.keys[layer].index_select(0, index).clone()
            selected.values[layer] = (
                self.values[layer].index_select(0, index).clone()
            )
        if self.boundary is not None:
            selected.boundary = self.boundary.index_select(0, index).clone()
            selected.boundary_depth = self.boundary_depth
        return selected

    def overwrite_last(
        self,
        layer_idx: int,
        mask: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Replaces the most recent cached position for selected rows.

        During batched generation a layer runs for the whole batch even though
        some sequences have already exited. Those sequences must not keep the
        keys and values that computation produced, because at batch size one
        they would never have been computed at all. This rewrites their entry
        with the propagated values, making a row's cache independent of whoever
        else shares its batch.

        Args:
            layer_idx: Index of the layer whose cache is being corrected.
            mask: Boolean tensor of shape ``(batch,)`` selecting rows to
                overwrite.
            key: Replacement keys shaped ``(batch, n_kv_heads, 1, head_dim)``.
                Only the rows selected by ``mask`` are read.
            value: Replacement values, same shape as ``key``.

        Raises:
            ValueError: If the layer lies above the depth cap or has no cached
                entries yet.
        """
        self._check_layer(layer_idx)
        if self.keys[layer_idx] is None:
            raise ValueError(
                f"Layer {layer_idx} has no cached entries to overwrite."
            )
        self.keys[layer_idx][mask, :, -1, :] = key[mask, :, 0, :]
        self.values[layer_idx][mask, :, -1, :] = value[mask, :, 0, :]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    RMSNorm rescales activations by their root mean square without subtracting
    the mean, which removes the centering term of LayerNorm at essentially no
    quality cost while being cheaper to compute.

    Args:
        dim: Size of the feature dimension being normalized.
        eps: Epsilon added to the mean square before the reciprocal square
            root, guarding against division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes and rescales the final dimension of ``x``.

        Args:
            x: Input tensor shaped ``(..., dim)``.

        Returns:
            A tensor with the same shape and dtype as ``x``.

        Note:
            The statistics are accumulated in float32 even under autocast, then
            cast back, so that low-precision training stays stable.
        """
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding tables.

    RoPE encodes absolute positions as a rotation applied to pairs of channels
    in the query and key vectors. Because the dot product of two rotated
    vectors depends only on the difference of their angles, attention scores
    become a function of relative distance.

    Args:
        head_dim: Per-head feature size. Must be even.
        max_seq_len: Number of positions to precompute rotations for.
        theta: Base period controlling how fast the rotation frequencies decay
            across channel pairs.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}.")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = torch.outer(positions, inv_freq)

        # Duplicated so the tables line up with the split-half rotation below.
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the two halves of the feature dimension into each other.

        Args:
            x: Tensor shaped ``(..., head_dim)``.

        Returns:
            The tensor ``[-x2, x1]`` where ``x1`` and ``x2`` are the first and
            second halves of the final dimension of ``x``.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def rotate(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotates a single query or key tensor.

        Named ``rotate`` rather than ``apply`` because :class:`torch.nn.Module`
        already defines ``apply`` for recursive initialization, and shadowing
        it breaks every ``model.apply(...)`` call.

        Args:
            x: Tensor shaped ``(batch, heads, seq_len, head_dim)``.
            offset: Absolute position of the first token in ``x``. During
                cached decoding this is the number of tokens already processed.

        Returns:
            The rotated tensor, with the shape and dtype of ``x``.

        Raises:
            ValueError: If ``offset + seq_len`` exceeds the precomputed table.
        """
        seq_len = x.size(-2)
        end = offset + seq_len
        if end > self.cos.size(0):
            raise ValueError(
                f"Position {end} exceeds the precomputed RoPE table of length "
                f"{self.cos.size(0)}."
            )

        cos = self.cos[offset:end].to(x.dtype)
        sin = self.sin[offset:end].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the rotation to a batch of queries and keys.

        Args:
            q: Queries shaped ``(batch, n_heads, seq_len, head_dim)``.
            k: Keys shaped ``(batch, n_kv_heads, seq_len, head_dim)``.
            offset: Absolute position of the first token in ``q`` and ``k``.

        Returns:
            A tuple ``(q_rotated, k_rotated)`` with the input shapes and dtypes.
        """
        return self.rotate(q, offset), self.rotate(k, offset)


class ExitAdapter(nn.Module):
    """A zero-initialized residual bottleneck placed before an exit's readout.

    A frozen parent gives shallow exits whatever happens to be linearly
    decodable from its intermediate states. That is a real lower bound, but it is
    a weak one, and unfreezing the backbone to improve it costs the parent's
    output. An adapter buys nonlinear remapping of the intermediate state without
    touching a single backbone weight, which keeps the no-regret property exact
    rather than approximate.

    The up-projection is zero, so at initialization the module returns its input
    unchanged. That matters for two reasons: adding adapters to a checkpoint
    cannot perturb it, and a training curve that starts from the frozen-exit
    result is directly comparable to it.

    Args:
        d_model: Width of the residual stream.
        rank: Bottleneck width. Cost is two ``d_model x rank`` matrices, which
            is negligible beside a vocabulary-sized head.

    Attributes:
        down: Projection into the bottleneck.
        up: Projection back out, zero-initialized.

    Raises:
        ValueError: If ``rank`` is not positive.
    """

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}.")
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.reset_identity()

    def reset_identity(self) -> None:
        """Restores exact identity behaviour.

        Called again after the model-wide initialization traversal, which visits
        every submodule and would otherwise fill ``up`` with the standard normal
        initialization — turning a documented identity into a random perturbation
        of the parent. That failure is invisible in the constructor and only
        appears once the whole model has been built, which is where the
        regression test for it lives.
        """
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the residual correction.

        Args:
            x: Hidden state shaped ``(..., d_model)``.

        Returns:
            ``x + up(silu(down(x)))``, shaped like ``x``.
        """
        return x + self.up(F.silu(self.down(x)))


class ExitModule(nn.Module):
    """Turns a layer's hidden state into next-token logits.

    One of these sits on each layer that is allowed to terminate a token. The
    projection is deliberately *not* owned here: every exit shares the model's
    single output matrix, which is itself tied to the input embedding. Giving
    each layer its own ``d_model x vocab_size`` head would add hundreds of
    millions of parameters to a model of a hundred million, and the shared head
    is what makes per-layer exits affordable at all.

    What remains per layer is the normalization. Each depth has its own scale
    and its own idea of what a "finished" residual stream looks like, so the
    norm has to be learned separately even though the projection is not.

    Args:
        d_model: Width of the residual stream.
        vocab_size: Number of output logits.
        eps: Epsilon for the normalization.
        adapter_rank: Width of an optional :class:`ExitAdapter` placed before the
            normalization. ``0`` omits it, which is the historical behaviour.

    Attributes:
        adapter: Optional zero-initialized residual bottleneck, or ``None``.
        norm: Per-layer :class:`RMSNorm`.
        proj: Output projection whose weight is shared with every other exit
            and with the input embedding.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        eps: float = 1e-5,
        adapter_rank: int = 0,
    ) -> None:
        super().__init__()
        self.adapter = ExitAdapter(d_model, adapter_rank) if adapter_rank else None
        self.norm = RMSNorm(d_model, eps=eps)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produces logits from a hidden state.

        Args:
            x: Hidden state shaped ``(batch, seq_len, d_model)``.

        Returns:
            Logits shaped ``(batch, seq_len, vocab_size)``.
        """
        if self.adapter is not None:
            x = self.adapter(x)
        return self.proj(self.norm(x))


def rms_direction(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Returns the part of a hidden state that normalization preserves.

    This is :class:`RMSNorm` without its learned gain: the direction that
    survives normalization, stripped of the magnitude that does not. It exists
    so that a reconstruction objective can be stated in terms of what the
    downstream projections actually consume. Matching raw hidden states instead
    wastes capacity on a scale factor the very next operation discards.

    Args:
        x: Input tensor shaped ``(..., dim)``.
        eps: Epsilon guarding the reciprocal square root.

    Returns:
        A tensor shaped like ``x``, with unit root-mean-square along the last
        dimension. Involves no parameters, so gradients flow only to whatever
        produced ``x``.
    """
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class KVPropagator(nn.Module):
    """Repairs an exited token's hidden state before a deeper layer caches it.

    When a token stops at layer ``L``, the layers above it still need keys and
    values for it. Plain state propagation reuses ``h_L`` unchanged, betting
    that residual connections keep neighbouring layers similar. That bet gets
    worse the further the state travels, and nothing in training ever tells the
    model to make it hold.

    This module learns the correction instead. Given the exit state and how far
    it has been carried, it predicts the residual stream the target layer would
    genuinely have received, so that layer's own key/value projections then
    operate on something close to their real input.

    Two properties are deliberate. The output projection is zero-initialized,
    so an untrained adapter reproduces plain propagation *exactly* and can only
    improve on it. And the correction is conditioned on the depth gap rather
    than being one fixed map, because carrying a state one layer is a very
    different problem from carrying it eight.

    Args:
        d_model: Width of the residual stream.
        rank: Bottleneck width. The adapter costs two ``d_model x rank``
            matrices, which is negligible beside a vocabulary-sized head.
        max_gap: Largest depth gap that will be requested, normally
            ``n_layers - 1``.

    Attributes:
        gap_embed: Per-gap bias injected into the bottleneck.
    """

    def __init__(self, d_model: int, rank: int, max_gap: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.gap_embed = nn.Embedding(max_gap + 1, rank)
        self.up = nn.Linear(rank, d_model, bias=False)

        self.reset_identity()

    def reset_identity(self) -> None:
        """Restores exact plain propagation after global model initialization."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gap_embed.weight)

    def forward(self, x: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        """Corrects a propagated hidden state.

        Args:
            x: Exit hidden state shaped ``(batch, seq_len, d_model)``.
            gap: How many layers the state has been carried. Either
                ``(batch,)``, when a whole row shares one exit depth as it does
                during decoding, or ``(batch, seq_len)``, when positions exit
                at different depths as they do when a sequence is simulated in
                parallel. Zero means the state is already the target layer's
                true input and needs no correction.

        Returns:
            The corrected state, shaped like ``x``. Entries whose gap is zero
            are returned untouched.
        """
        # Normalize a per-row gap to a length-one sequence axis so both cases
        # broadcast identically from here on.
        if gap.dim() == x.dim() - 2:
            gap = gap.unsqueeze(-1)

        correction = self.up(F.silu(self.down(x) + self.gap_embed(gap)))

        # A gap of zero means this state already is the target layer's true
        # input, so there is nothing to repair and any correction can only do
        # harm. Enforced rather than assumed: the training objective never
        # supervises gap zero, so an unmasked adapter would apply an arbitrary
        # learned offset to a state that was exact.
        return x + correction * (gap != 0).unsqueeze(-1).to(x.dtype)


def uncertainty(logits: torch.Tensor, criterion: str = "entropy") -> torch.Tensor:
    """Scores how unsure a prediction is, on a common ``[0, 1]`` scale.

    The three measures disagree about what confidence means but agree on
    direction and range, so a single threshold works for all of them and a
    token exits whenever its score falls *below* that threshold. Sharing the
    direction matters: entropy is naturally low when confident while the other
    two are naturally high, and flipping the comparison per criterion is a
    reliable source of silent bugs.

    Args:
        logits: Unnormalized scores shaped ``(..., vocab_size)``.
        criterion: Which measure to use.

            * ``"entropy"`` — Shannon entropy divided by ``log(vocab_size)``.
              Reads the whole distribution, so it stays high when probability
              is spread thinly over many tokens.
            * ``"max_prob"`` — one minus the largest probability. Cheap, but
              blind to how the remaining mass is arranged.
            * ``"top2_margin"`` — one minus the gap between the two largest
              probabilities. Sensitive to the case that actually matters for
              greedy decoding, namely two candidates nearly tied for first.

    Returns:
        Uncertainty in ``[0, 1]`` with the vocabulary dimension reduced, so
        shape ``(...)``. Zero means completely certain.

    Raises:
        ValueError: If ``criterion`` is not one of the three names above.
    """
    # Computed in float32 regardless of autocast: under bf16 the log-softmax of
    # a 50k vocabulary loses enough precision to move an exit decision.
    log_probs = F.log_softmax(logits.float(), dim=-1)

    if criterion == "entropy":
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return entropy / math.log(logits.size(-1))
    if criterion == "max_prob":
        return 1.0 - log_probs.max(dim=-1).values.exp()
    if criterion == "top2_margin":
        top2 = log_probs.topk(2, dim=-1).values.exp()
        return 1.0 - (top2[..., 0] - top2[..., 1])

    raise ValueError(
        "criterion must be one of ('entropy', 'max_prob', 'top2_margin'), got "
        f"{criterion!r}."
    )