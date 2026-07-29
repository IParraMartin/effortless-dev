"""A decoder-only Transformer that can be executed to a chosen depth.

The backbone is a modern LLM stack (RMSNorm pre-norm, rotary positions,
grouped-query attention, SwiGLU, tied embeddings). What it adds is an
:class:`~modules.ExitModule` on each layer, so a prediction can be read off at
any depth — and two different ways of deciding which depth to use.

**Request-level vertical routing** (:meth:`Transformer.generate_routed`) picks
one depth per request from a cheap probe of the prompt, then never executes or
allocates anything above it. Nothing is approximated: every layer that runs sees
exactly the keys and values it would have seen at full depth, because every
layer below it ran too. Cache memory falls in proportion to the depth.

**Token-level early exit** (:meth:`Transformer.generate`) lets each token stop
as soon as it is confident. This came first and is retained. Its difficulty is
the key/value cache: if a token stops at layer three, layers four and up never
saw it, yet later tokens still need its keys and values at those depths. The
fix, following CALM (Schuster et al., 2022), propagates the exit hidden state
upward and computes only the cheap key/value projections there. That *is* an
approximation, and ``TransformerConfig.full_depth_kv`` turns it off so its cost
can be measured.

Throughout, **depth means the number of executed blocks**, from 1 to
``n_layers``; a layer *index* is one less. The conversion lives in
:meth:`Transformer.layer_of_depth` and :meth:`Transformer.depth_of_layer` and
nowhere else.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RoutingConfig, TransformerConfig
from src.modules import (
    ExitModule,
    KVCache,
    KVPropagator,
    RMSNorm,
    RotaryEmbedding,
    rms_direction,
    uncertainty,
)
from src.routing import (
    DepthController,
    RoutingTrace,
    build_controller,
    pool_prompt_features,
)
from utils.costs import AnalyticalCostModel, CostCounters


@dataclass
class ExitOutput:
    """Result of a forward pass.

    Attributes:
        logits: Logits from the final exit, shaped
            ``(batch, seq_len, vocab_size)``. Training always reads the deepest
            exit; early exiting is a generation-time behaviour.
        loss: Weighted multi-exit objective, or ``None`` when no targets were
            supplied.
        exit_losses: Detached cross-entropy per exit, keyed by layer index.
            Only the exits that took part in this step appear, which matters
            when ``exits_per_step`` is sampling a subset.
        kv_loss: Detached reconstruction error of the key/value adapters,
            relative to the scale of what they are predicting, or ``None`` when
            learned propagation is off. Values near one mean the adapters are
            no better than predicting zero.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    exit_losses: dict[int, float] = field(default_factory=dict)
    kv_loss: float | None = None


@dataclass
class DepthState:
    """Where a request has got to in the stack, and what that cost.

    Returned by :meth:`Transformer.forward_to_depth` and
    :meth:`Transformer.continue_from_depth`. It carries the three things a
    caller needs to go further — the activation, how many blocks produced it,
    and where those tokens sit on the rotary table — plus the accounting for
    any work spent replaying a prompt through upper blocks.

    Attributes:
        hidden: Residual stream leaving block ``depth - 1``, shaped
            ``(batch, seq_len, d_model)``.
        depth: Number of blocks executed, in ``[0, n_layers]``.
        offset: Absolute position of the first token, needed so a later
            continuation rotates and caches at the right positions.
        boundary: Retained activation for every position, present only when
            requested. Escalation needs the whole prompt, not just its last
            token, so this is what makes exact escalation possible.
        backfill_tokens: Positions re-run through upper blocks by a
            continuation. Zero for ordinary prefill.
        backfill_blocks: Blocks those positions were re-run through.
        boundary_bytes: Size of the activation the continuation consumed.
    """

    hidden: torch.Tensor
    depth: int
    offset: int
    boundary: torch.Tensor | None = None
    backfill_tokens: int = 0
    backfill_blocks: int = 0
    boundary_bytes: int = 0

    @property
    def last_token(self) -> torch.Tensor:
        """The final position's state, which is what a readout needs."""
        return self.hidden[:, -1]


@dataclass
class ExitStatistics:
    """Per-exit behaviour on a teacher-forced batch.

    Everything here is reduced over the vocabulary, so the whole structure is a
    few megabytes even for long sequences. That is what makes it practical to
    sweep exit thresholds without recomputing logits.

    Attributes:
        uncertainty: Uncertainty of each exit at each position, shaped
            ``(n_exits, batch, seq_len)``.
        correct: Whether each exit's greedy prediction matched the target, same
            shape.
        agrees_with_final: Whether each exit's greedy prediction matched the
            *deepest* exit's, same shape. This is what an oracle exit policy
            would key on: it marks the shallowest depth at which the full
            model's own answer is already available.
        nll: Negative log-likelihood of the target under each exit, same shape.
        valid: Positions that were not padding, shaped ``(batch, seq_len)``.
        exit_layers: Layer index behind each row of the leading dimension.
        n_layers: Depth of the model, for computing compute-saved figures.
    """

    uncertainty: torch.Tensor
    correct: torch.Tensor
    agrees_with_final: torch.Tensor
    nll: torch.Tensor
    valid: torch.Tensor
    exit_layers: tuple[int, ...]
    n_layers: int


@dataclass
class GenerationOutput:
    """Result of :meth:`Transformer.generate`.

    Attributes:
        sequences: Prompt plus generated tokens, shaped
            ``(batch, prompt_len + generated_len)``.
        exit_layers: Layer each generated token stopped at, shaped
            ``(batch, generated_len)``. The prompt is always processed at full
            depth, so its first continuation is recorded at the final layer.
    """

    sequences: torch.Tensor
    exit_layers: torch.Tensor

    @property
    def mean_exit_layer(self) -> float:
        """Average stopping layer *index*, one less than the executed depth."""
        return float(self.exit_layers.float().mean())

    @property
    def mean_exit_depth(self) -> float:
        """Average number of blocks executed per generated token.

        This is the quantity compute is proportional to, and the one to quote:
        a token stopping at layer index ``L`` ran ``L + 1`` blocks.
        """
        return self.mean_exit_layer + 1.0


@dataclass
class RoutedGenerationOutput:
    """Result of :meth:`Transformer.generate_routed`.

    Attributes:
        sequences: Prompts followed by generated tokens, shaped
            ``(batch, padded_prompt_len + max_new_tokens)``. Rows whose prompt
            was shorter than the batch's widest, or which stopped early, are
            filled with the pad token, so ``prompt_lengths`` is needed to read
            them back.
        depths: Depth each request was routed to, in executed blocks.
        prompt_lengths: Real prompt length per request.
        max_new_tokens: Tokens requested, which is how the generated span is
            located within a row whose prompt was shorter than the batch's
            widest.
        trace: Routing decisions and their accounting, or ``None``.
        counters: What execution actually did, counted as it happened.
        estimated_macs: Multiply-accumulate breakdown derived from those
            counters. An estimate, never a latency — a routed system can save
            arithmetic and still be slower, and the two are reported in
            separate columns everywhere in this repository.
    """

    sequences: torch.Tensor
    depths: torch.Tensor
    prompt_lengths: torch.Tensor
    max_new_tokens: int = 0
    trace: RoutingTrace | None = None
    counters: CostCounters | None = None
    estimated_macs: dict[str, float] = field(default_factory=dict)

    @property
    def mean_depth(self) -> float:
        """Average routed depth across requests."""
        return float(self.depths.float().mean())

    def completions(self) -> list[torch.Tensor]:
        """Extracts each request's generated tokens.

        Rows are laid out as ``prompt`` then ``generated`` then padding, and a
        short prompt shifts its generated span left relative to the batch, so
        the span is located from that row's own prompt length rather than from
        the tensor's width.

        Returns:
            One tensor of up to ``max_new_tokens`` ids per request, in the
            original row order. A sequence stopped by an end-of-sequence token
            is padded out to that length.
        """
        return [
            self.sequences[
                row,
                int(self.prompt_lengths[row]) : int(self.prompt_lengths[row])
                + self.max_new_tokens,
            ]
            for row in range(self.sequences.size(0))
        ]


class Attention(nn.Module):
    """Causal grouped-query self-attention.

    Queries are projected to ``n_heads`` heads while keys and values are
    projected to a smaller number of ``n_kv_heads`` heads that each serve a
    group of queries. The scaled dot product itself is delegated to
    :func:`torch.nn.functional.scaled_dot_product_attention`, which dispatches
    to a fused FlashAttention kernel where one is available.

    Args:
        config: Model configuration.
        layer_idx: Position of this layer in the stack, used as the key into
            the KV cache.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.d_model, bias=False
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def project_kv(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes keys and values without attending to anything.

        This is the entire cost a layer pays for a token that already exited:
        two projections and a rotation, with no attention matrix and no
        feed-forward network.

        Args:
            x: Normalized activations shaped ``(batch, seq_len, d_model)``.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``.

        Returns:
            A tuple ``(keys, values)``, each shaped
            ``(batch, n_kv_heads, seq_len, head_dim)``.
        """
        batch, seq_len, _ = x.shape
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        k = rope.rotate(k.transpose(1, 2), offset=offset)
        return k, v.transpose(1, 2)

    def write_kv(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int,
        cache: KVCache,
    ) -> None:
        """Projects keys and values straight into the cache.

        Args:
            x: Normalized activations shaped ``(batch, seq_len, d_model)``.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``.
            cache: Cache to extend.
        """
        k, v = self.project_kv(x, rope, offset)
        cache.update(self.layer_idx, k, v)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int = 0,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Runs masked self-attention over the sequence.

        Args:
            x: Input activations shaped ``(batch, seq_len, d_model)``.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``. Must be
                supplied by the caller rather than read from ``cache``, because
                shallower layers have already extended the cache by the time
                this layer runs.
            cache: Optional KV cache. When supplied, the keys and values for
                ``x`` are appended to it and attention runs against the full
                cached history.

        Returns:
            Attention output shaped ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # (batch, heads, seq_len, head_dim) is the layout SDPA expects.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = rope(q, k, offset=offset)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        # When the queries span the whole key sequence the fused causal kernel
        # applies directly. Otherwise the queries sit at the end of a longer
        # cached history: a lone decoding step may attend to everything, while
        # a longer chunk still needs an explicit shifted causal mask.
        total_len = k.size(-2)
        attn_mask = None
        is_causal = total_len == seq_len
        if not is_causal and seq_len > 1:
            attn_mask = torch.ones(
                seq_len, total_len, dtype=torch.bool, device=x.device
            ).tril(diagonal=total_len - seq_len)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.resid_dropout(self.o_proj(out))


class SwiGLU(nn.Module):
    """Gated feed-forward network with a SiLU-activated gate.

    The block computes ``down(silu(gate(x)) * up(x))``. Splitting the hidden
    projection into a gate and a value path consistently outperforms a single
    activation at matched parameter count.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ff_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ff_dim, bias=False)
        self.down_proj = nn.Linear(config.ff_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the gated projection.

        Args:
            x: Input activations shaped ``(batch, seq_len, d_model)``.

        Returns:
            A tensor of the same shape as ``x``.
        """
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class DecoderBlock(nn.Module):
    """One pre-norm decoder block: attention then feed-forward.

    Both sublayers are wrapped in residual connections with normalization
    applied to the branch input rather than the sum. This leaves an unnormalized
    identity path through the whole network, which is what makes deep stacks
    trainable without learning-rate warmup tricks.

    Args:
        config: Model configuration.
        layer_idx: Position of this block in the stack.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

        self.kv_adapter = (
            KVPropagator(config.d_model, config.kv_adapter_rank, config.n_layers)
            if config.learned_kv_propagation
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int = 0,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Passes activations through both sublayers.

        Args:
            x: Input activations shaped ``(batch, seq_len, d_model)``.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``.
            cache: Optional KV cache forwarded to the attention sublayer.

        Returns:
            Updated activations of the same shape as ``x``.
        """
        x = x + self.attn(self.attn_norm(x), rope, offset, cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def repair(self, x: torch.Tensor, gap: torch.Tensor | None) -> torch.Tensor:
        """Applies this block's learned correction to a propagated state.

        Args:
            x: Residual-stream state shaped ``(batch, seq_len, d_model)``.
            gap: Layers the state has been carried, shaped ``(batch,)``. When
                ``None``, or when no adapter is configured, the state is
                returned unchanged.

        Returns:
            The corrected state, shaped like ``x``.
        """
        if self.kv_adapter is None or gap is None:
            return x
        return self.kv_adapter(x, gap)

    def project_kv(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int = 0,
        gap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes what this layer would cache for a propagated state.

        The state must pass through this block's own ``attn_norm`` first,
        because that is exactly what :meth:`forward` feeds to ``k_proj``.
        Skipping the normalization writes plausible-looking but wrong entries
        into the cache, which degrades generation without ever raising.

        Args:
            x: Residual-stream state shaped ``(batch, seq_len, d_model)``,
               before normalization.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``.
            gap: Layers the state has been carried, for the learned adapter.

        Returns:
            A tuple ``(keys, values)`` for this layer.
        """
        return self.attn.project_kv(self.attn_norm(self.repair(x, gap)), rope, offset)

    def propagate_kv(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        offset: int,
        cache: KVCache,
        gap: torch.Tensor | None = None,
    ) -> None:
        """Caches a propagated state's keys and values, skipping the block.

        Args:
            x: Residual-stream state shaped ``(batch, seq_len, d_model)``.
            rope: Shared rotary embedding tables.
            offset: Absolute position of the first token of ``x``.
            cache: Cache to extend.
            gap: Layers the state has been carried, for the learned adapter.
        """
        self.attn.write_kv(
            self.attn_norm(self.repair(x, gap)), rope, offset, cache
        )


class Transformer(nn.Module):
    """Decoder-only Transformer with per-layer exit modules.

    Args:
        config: Model configuration. Defaults to a ~124M parameter setup with
            an exit on every layer.

    Example:
        >>> model = Transformer(TransformerConfig(vocab_size=32000))
        >>> tokens = torch.randint(0, 32000, (2, 128))
        >>> out = model(tokens, targets=tokens)
        >>> out.logits.shape
        torch.Size([2, 128, 32000])
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()

        self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.embed_dropout = nn.Dropout(self.config.dropout)
        self.rope = RotaryEmbedding(
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config, i) for i in range(self.config.n_layers)
        )
        self.exit_modules = nn.ModuleList(
            ExitModule(
                self.config.d_model, self.config.vocab_size, self.config.norm_eps
            )
            for _ in self.config.exit_layers
        )
        #: Maps a layer index to its position in :attr:`exit_modules`.
        self.exit_index = {
            layer: i for i, layer in enumerate(self.config.exit_layers)
        }
        # Drives the exit rotation when only a subset is scored per step. Kept
        # as a buffer so it advances identically on every DDP rank without any
        # communication, and left out of checkpoints since it only affects
        # which exits a given step happens to visit.
        self.register_buffer(
            "_step_counter", torch.zeros((), dtype=torch.long), persistent=False
        )
        # Set by :meth:`score_all_exits` to suspend that rotation. Not a buffer:
        # it is scoped to a context manager and must never reach a checkpoint.
        self._score_every_exit = False
        # Plain integers rather than buffers: these are instrumentation, they
        # must not enter a checkpoint, and they are read from tests and cost
        # accounting on the host anyway.
        self._head_calls = 0
        self._head_tokens = 0

        # Routing is opt-in. With nothing attached the model behaves exactly as
        # it did before request-level routing existed, which is what keeps the
        # unrouted baseline a genuine baseline rather than a special case of
        # the new path.
        self.routing: RoutingConfig | None = None
        self.depth_controller: DepthController | None = None

        self.apply(self._init_weights)
        self._scale_residual_projections()
        # ``Module.apply`` also visits propagation adapters and would overwrite
        # their deliberate zero initialization. Restore exact identity after the
        # model-wide initialization pass.
        for block in self.blocks:
            if block.kv_adapter is not None:
                block.kv_adapter.reset_identity()

        if self.config.tie_embeddings:
            # Every exit shares one output matrix, itself tied to the input
            # embedding. Independent heads would cost n_exits * d_model *
            # vocab_size parameters, several times the size of the backbone.
            for exit_module in self.exit_modules:
                exit_module.proj.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes a single submodule in place.

        Linear and embedding weights are drawn from a zero-mean normal with
        standard deviation ``config.init_std``; biases, where present, start at
        zero.

        Args:
            module: The submodule visited by :meth:`torch.nn.Module.apply`.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_projections(self) -> None:
        """Downscales the projections that write into the residual stream.

        Every layer adds two branches to the residual stream, so its variance
        would otherwise grow linearly with depth. Shrinking the output
        projections by ``1 / sqrt(2 * n_layers)`` keeps activations at roughly
        unit scale at initialization regardless of how deep the stack is.
        """
        scale = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(
                block.attn.o_proj.weight,
                mean=0.0,
                std=self.config.init_std * scale,
            )
            nn.init.normal_(
                block.ffn.down_proj.weight,
                mean=0.0,
                std=self.config.init_std * scale,
            )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Counts the model's parameters.

        Args:
            trainable_only: When ``True``, frozen parameters are excluded.

        Returns:
            The total number of parameters. Weights shared between the
            embedding and the exit modules are counted once.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in {id(p): p for p in params}.values())

    def _check_length(self, offset: int, seq_len: int) -> None:
        """Rejects sequences the rotary tables cannot cover.

        Args:
            offset: Number of tokens already cached.
            seq_len: Number of incoming tokens.

        Raises:
            ValueError: If the total exceeds ``config.max_seq_len``.
        """
        total_len = offset + seq_len
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_len} exceeds the model's maximum of "
                f"{self.config.max_seq_len}."
            )

    # ------------------------------------------------------------------
    # Depth-capped execution
    #
    # "Depth" throughout this section means the number of blocks *executed*,
    # so it runs from 1 to n_layers and a request routed to depth d runs
    # blocks 0 .. d-1 and reads the exit attached to layer index d-1. The
    # off-by-one against layer indices is real and has already produced one
    # bug in this repository, so the conversion happens in exactly two places
    # -- `layer_of_depth` and `depth_of_layer` -- and nowhere else.
    # ------------------------------------------------------------------

    @staticmethod
    def layer_of_depth(depth: int) -> int:
        """Converts an executed depth to the layer index that produced it."""
        return depth - 1

    @staticmethod
    def depth_of_layer(layer: int) -> int:
        """Converts a layer index to the number of blocks executed to reach it."""
        return layer + 1

    def _check_depth(self, depth: int, name: str = "depth") -> int:
        """Validates an executed-depth argument.

        Args:
            depth: Number of blocks to run.
            name: Argument name, used in the error message.

        Returns:
            The depth, unchanged.

        Raises:
            ValueError: If the depth falls outside ``[0, n_layers]``.
        """
        if not 0 <= depth <= self.config.n_layers:
            raise ValueError(
                f"{name} counts executed blocks and must lie in "
                f"[0, {self.config.n_layers}], got {depth}. Note this is a "
                f"depth, not a layer index: the deepest layer is "
                f"{self.config.n_layers - 1} but full depth is "
                f"{self.config.n_layers}."
            )
        return depth

    def _run_range(
        self,
        x: torch.Tensor,
        start_depth: int,
        stop_depth: int,
        offset: int,
        cache: KVCache | None,
    ) -> torch.Tensor:
        """Runs a contiguous range of blocks over an activation.

        This is the single execution primitive the rest of the depth-capped
        machinery is built from, so prefix execution, suffix continuation, and
        full-depth execution cannot drift apart.

        Args:
            x: Residual stream entering block ``start_depth``, shaped
                ``(batch, seq_len, d_model)``.
            start_depth: First block to run, as an executed-depth index.
            stop_depth: One past the last block to run.
            offset: Absolute position of the first token of ``x``. Passed
                explicitly rather than read from the cache, because shallower
                layers have already extended it by the time deeper ones run.
            cache: Optional cache to extend.

        Returns:
            The residual stream leaving block ``stop_depth - 1``, or ``x``
            itself when the range is empty.
        """
        for layer_idx in range(start_depth, stop_depth):
            x = self.blocks[layer_idx](x, self.rope, offset, cache)
        return x

    def forward_to_depth(
        self,
        input_ids: torch.Tensor,
        stop_depth: int,
        cache: KVCache | None = None,
        offset: int | None = None,
        return_boundary_state: bool = False,
    ) -> DepthState:
        """Executes the first ``stop_depth`` blocks and stops there.

        Nothing above ``stop_depth`` runs, and with a depth-capped ``cache``
        nothing above it is stored either. That is the difference between
        request-level routing and token-level early exit: the latter still
        visits every layer to keep the cache complete.

        Works for both prefill and incremental decode. The distinction is
        entirely in ``offset``, which places the incoming tokens on the rotary
        table and defaults to whatever the cache already holds.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            stop_depth: Number of blocks to execute.
            cache: Optional cache to extend. Its depth cap must be at least
                ``stop_depth``.
            offset: Absolute position of the first incoming token. Defaults to
                the cache's current length, or zero without a cache.
            return_boundary_state: Whether to retain the activation leaving the
                last executed block for every position, so upper blocks can
                later be run over the same prompt without repeating the lower
                ones. Costs ``batch * seq_len * d_model`` elements.

        Returns:
            A :class:`DepthState` at ``stop_depth``.

        Raises:
            ValueError: If ``stop_depth`` is out of range, if the cache cannot
                hold it, or if the sequence exceeds the rotary tables.
        """
        self._check_depth(stop_depth, "stop_depth")
        if cache is not None and cache.max_depth < stop_depth:
            raise ValueError(
                f"Cache is capped at depth {cache.max_depth} but execution to "
                f"depth {stop_depth} was requested. Build the cache with "
                f"max_depth >= {stop_depth}."
            )

        if offset is None:
            offset = len(cache) if cache is not None else 0
        self._check_length(offset, input_ids.size(1))

        x = self.embed_dropout(self.embed(input_ids))
        x = self._run_range(x, 0, stop_depth, offset, cache)

        state = DepthState(hidden=x, depth=stop_depth, offset=offset)
        if return_boundary_state:
            state.boundary = x
            if cache is not None:
                cache.retain_boundary(x, stop_depth)
        return state

    def continue_from_depth(
        self,
        hidden_states: torch.Tensor,
        start_depth: int,
        stop_depth: int,
        cache: KVCache | None = None,
        offset: int = 0,
        return_boundary_state: bool = False,
    ) -> DepthState:
        """Runs the blocks between two depths over an already computed state.

        **Escalation is exact only when the caller supplies the boundary
        activation for every prompt position**, not just the last one. Upper
        blocks need keys and values at every position they will attend to, and
        those come from running the suffix over the whole retained activation.
        This repository takes that strategy — retain and replay — rather than
        recomputing the lower prefix, and :attr:`DepthState.backfill_tokens`
        records what the replay cost. Handing in only the final position
        produces a cache with holes and silently wrong attention, so the length
        is checked against the cache rather than trusted.

        Args:
            hidden_states: Residual stream entering block ``start_depth``,
                shaped ``(batch, seq_len, d_model)``.
            start_depth: Number of blocks already executed.
            stop_depth: Number of blocks to have executed on return.
            cache: Optional cache to extend, capped at ``stop_depth`` or more.
            offset: Absolute position of the first token of ``hidden_states``,
                normally taken from the originating :class:`DepthState`.
            return_boundary_state: Whether to retain the new boundary
                activation for a possible further escalation.

        Returns:
            A :class:`DepthState` at ``stop_depth``, with the backfill
            accounting filled in.

        Raises:
            ValueError: If the depths are out of range or inverted, if the
                cache cannot hold ``stop_depth``, or if the supplied state
                covers fewer positions than the cache already holds.
        """
        self._check_depth(start_depth, "start_depth")
        self._check_depth(stop_depth, "stop_depth")
        if stop_depth < start_depth:
            raise ValueError(
                f"stop_depth ({stop_depth}) cannot be shallower than "
                f"start_depth ({start_depth})."
            )
        if cache is not None:
            if cache.max_depth < stop_depth:
                raise ValueError(
                    f"Cache is capped at depth {cache.max_depth} but "
                    f"continuation to depth {stop_depth} was requested."
                )
            cached = len(cache)
            supplied = hidden_states.size(1)
            if stop_depth > start_depth and offset + supplied < cached:
                raise ValueError(
                    f"Continuing to depth {stop_depth} needs the boundary "
                    f"activation for all {cached} cached positions, but only "
                    f"{supplied} were supplied at offset {offset}. Upper "
                    f"blocks would be left without keys and values for the "
                    f"earlier positions they must attend to. Re-run the prefix "
                    f"with return_boundary_state=True."
                )

        self._check_length(offset, hidden_states.size(1))
        x = self._run_range(
            hidden_states, start_depth, stop_depth, offset, cache
        )

        blocks_run = max(stop_depth - start_depth, 0)
        state = DepthState(
            hidden=x,
            depth=stop_depth,
            offset=offset,
            backfill_tokens=hidden_states.size(1) if blocks_run else 0,
            backfill_blocks=blocks_run,
            boundary_bytes=(
                hidden_states.numel() * hidden_states.element_size()
                if blocks_run
                else 0
            ),
        )
        if return_boundary_state:
            state.boundary = x
            if cache is not None:
                cache.retain_boundary(x, stop_depth)
        return state

    def endpoint_logits(self, hidden: torch.Tensor, depth: int) -> torch.Tensor:
        """Reads out the vocabulary once, at one chosen depth.

        A fixed-depth endpoint applies that depth's exit normalization and the
        shared output projection, and nothing else. In particular it does not
        evaluate the shallower exits on the way past: those cost ``d_model *
        vocab_size`` multiplications each, which for a realistic vocabulary is
        several times a whole block, so testing every checkpoint with a full
        projection can cost more than the depth it saves.

        Args:
            hidden: Residual stream leaving block ``depth - 1``, shaped
                ``(..., d_model)``.
            depth: Executed depth the state came from.

        Returns:
            Logits shaped ``(..., vocab_size)``.

        Raises:
            ValueError: If no exit module sits at that depth.
        """
        self._check_depth(depth, "depth")
        layer = self.layer_of_depth(depth)
        if layer not in self.exit_index:
            raise ValueError(
                f"Depth {depth} (layer index {layer}) carries no exit module. "
                f"Exits sit at depths {self.config.exit_depths}."
            )
        return self._readout(self.exit_index[layer], hidden)

    def _readout(self, exit_position: int, hidden: torch.Tensor) -> torch.Tensor:
        """Applies one exit module, counting the vocabulary projection.

        Every vocabulary projection in this class goes through here, so the
        counters cannot drift from what was actually computed. That matters
        because the central efficiency claim of request-level routing is that
        it pays for the head once per generated token, and a counter that
        misses a call site would make the claim unfalsifiable.

        Args:
            exit_position: Index into :attr:`exit_modules`.
            hidden: Hidden state shaped ``(..., d_model)``.

        Returns:
            Logits shaped ``(..., vocab_size)``.
        """
        self._head_calls += 1
        self._head_tokens += hidden.numel() // hidden.size(-1)
        return self.exit_modules[exit_position](hidden)

    def reset_head_counters(self) -> None:
        """Zeroes the vocabulary-projection counters."""
        self._head_calls = 0
        self._head_tokens = 0

    @property
    def head_calls(self) -> int:
        """Number of vocabulary projections performed since the last reset."""
        return self._head_calls

    @property
    def head_tokens(self) -> int:
        """Number of token positions projected to the vocabulary."""
        return self._head_tokens

    def _run_blocks(
        self,
        input_ids: torch.Tensor,
        offset: int = 0,
        cache: KVCache | None = None,
    ) -> list[torch.Tensor]:
        """Runs the full stack, keeping every layer's output.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            offset: Number of tokens already cached.
            cache: Optional KV cache to extend.

        Returns:
            One hidden state per layer, each shaped
            ``(batch, seq_len, d_model)``. Holding all of them costs
            ``n_layers * batch * seq_len * d_model``, small next to the logits
            they will be turned into, and the layers between exits are what the
            key/value adapters are trained against.
        """
        x = self.embed_dropout(self.embed(input_ids))

        hidden_states = []
        for block in self.blocks:
            x = block(x, self.rope, offset, cache)
            hidden_states.append(x)
        return hidden_states

    @torch.no_grad()
    def simulate_early_exit(
        self,
        input_ids: torch.Tensor,
        exit_layers: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduces early-exit decoding for a whole sequence in one pass.

        Decoding one token at a time is the obvious way to obtain the states
        early exiting really produces, and it is far too slow to put inside a
        training loop. It is also unnecessary. Attention here is strictly
        causal, and a position's cached keys and values depend only on its own
        trajectory, so once each position's exit layer is fixed the entire
        corrupted forward can be evaluated in parallel: at every layer, keys
        and values come from the propagated state for positions that have
        already stopped and from the ordinary path for the rest.

        The result is the state distribution that early exiting actually
        induces, including the compounding effect of positions attending to
        approximations left behind by earlier positions. That is what makes it
        possible to train the propagation adapters on the inputs they will meet
        at inference rather than on the clean states a full-depth pass gives.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            exit_layers: Layer each position stops at, shaped
                ``(batch, seq_len)``.

        Returns:
            Each position's hidden state at its own exit layer, shaped
            ``(batch, seq_len, d_model)``.

        Note:
            Dropout follows the module's current mode, so call this with the
            model in eval mode if the simulation is meant to mirror inference
            exactly. The default configuration uses no dropout.
        """
        batch, seq_len = input_ids.shape
        x = self.embed_dropout(self.embed(input_ids))
        frozen = x
        exited = torch.zeros_like(exit_layers, dtype=torch.bool)

        for layer_idx, block in enumerate(self.blocks):
            attention = block.attn
            normed = block.attn_norm(x)

            q = attention.q_proj(normed).view(
                batch, seq_len, attention.n_heads, attention.head_dim
            ).transpose(1, 2)
            k = attention.k_proj(normed).view(
                batch, seq_len, attention.n_kv_heads, attention.head_dim
            ).transpose(1, 2)
            v = attention.v_proj(normed).view(
                batch, seq_len, attention.n_kv_heads, attention.head_dim
            ).transpose(1, 2)
            q, k = self.rope(q, k, offset=0)

            if bool(exited.any()):
                # Positions that stopped below this layer contribute the keys
                # and values propagation would have written for them.
                gap = (layer_idx - 1 - exit_layers).clamp(min=0)
                propagated_k, propagated_v = block.project_kv(
                    frozen, self.rope, 0, gap
                )
                swap = exited.view(batch, 1, seq_len, 1)
                k = torch.where(swap, propagated_k, k)
                v = torch.where(swap, propagated_v, v)

            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=attention.dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=attention.n_kv_heads != attention.n_heads,
            )
            updated = x + attention.resid_dropout(
                attention.o_proj(
                    attended.transpose(1, 2).reshape(batch, seq_len, -1)
                )
            )
            updated = updated + block.ffn(block.ffn_norm(updated))

            stopping = (exit_layers == layer_idx) & ~exited
            frozen = torch.where(stopping.unsqueeze(-1), updated, frozen)
            exited = exited | stopping
            # Positions that have stopped keep their frozen state; the rest
            # carry on down the stack.
            x = torch.where(exited.unsqueeze(-1), frozen, updated)

        return frozen

    def _sample_exit_layers(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Draws a random exit depth for each position.

        Sampling uniformly over the permitted depths, rather than using the
        exits the model's own confidence would currently choose, keeps the
        adapters covered across every gap they might later be asked to bridge.
        Training only on today's exits would chase a target that moves as the
        model's confidence changes.

        Args:
            shape: Shape ``(batch, seq_len)`` to fill.
            device: Device to allocate on.

        Returns:
            Exit layer indices, one per position.
        """
        eligible = torch.tensor(
            [
                layer
                for layer in self.config.exit_layers
                if layer >= self.config.min_exit_layer
            ],
            dtype=torch.long,
            device=device,
        )
        indices = torch.randint(0, eligible.numel(), shape, device=device)
        return eligible[indices]

    def _propagation_loss(
        self,
        hidden_states: list[torch.Tensor],
        valid: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Teaches the adapters to reconstruct states they never see directly.

        For a source exit layer ``L``, every deeper block is asked to recover
        the residual stream it would truly have received, from nothing but the
        state at ``L`` and the depth gap. The full-depth pass already computed
        those true states, so supervision is free: this is self-distillation
        from the model's own deeper layers.

        The comparison is made on the RMS-normalized state, not the raw one,
        because that is what the layer's key and value projections receive. The
        distinction is not cosmetic: matching raw hidden states spends the
        adapter's capacity on a magnitude that normalization immediately
        discards, which measurably fails to reduce the cache error it was
        meant to fix.

        The target is detached and :func:`rms_direction` has no parameters, so
        this objective trains the adapters alone and cannot distort the
        backbone into making its own states easier to predict.

        A single source layer is used per step, rotating deterministically so
        that all ranks agree and every depth is covered over time.

        Args:
            hidden_states: Output of :meth:`_run_blocks`, one per layer.
            valid: Positions that are not padding, shaped ``(batch, seq_len)``.
            input_ids: Token ids, forwarded to the simulation when the
                exposure-matched scheme is in use.

        Returns:
            A scalar loss, or ``None`` when there is no gap large enough to
            need correcting.
        """
        if self.config.n_layers < 3:
            return None

        exit_layers = self._sample_exit_layers(valid.shape, valid.device)
        source = self._exit_states(hidden_states, exit_layers, input_ids)

        total = source.new_zeros(())
        terms = 0
        # A gap of zero means the state already is that block's true input, so
        # the block directly above an exit is exact and has nothing to learn.
        for target in range(2, self.config.n_layers):
            gap = (target - 1 - exit_layers).clamp(min=0)
            active = valid & (gap >= 1)
            if not bool(active.any()):
                continue

            mask = active.unsqueeze(-1)
            predicted = rms_direction(self.blocks[target].repair(source, gap))
            truth = rms_direction(hidden_states[target - 1].detach())

            error = ((predicted - truth) * mask).pow(2).sum()
            scale = (truth * mask).pow(2).sum().clamp(min=1e-6)
            total = total + error / scale
            terms += 1

        return total / max(terms, 1) if terms else None

    def _exit_states(
        self,
        hidden_states: list[torch.Tensor],
        exit_layers: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Collects the state each position would carry away from its exit.

        This is the one place the two training schemes differ, and the
        difference is the whole point:

        ``"teacher"`` reads the state out of the full-depth pass. It is free,
        but no such state exists at inference, where every layer below the exit
        attended over a cache already full of approximations. An adapter fitted
        this way is solving a cleaner problem than the one it will face.

        ``"simulated"`` runs :meth:`simulate_early_exit` to produce the states
        early exiting really induces, so the adapter is fitted on its own
        deployment distribution. The states are detached, making this the
        supervised step of a DAgger-style loop: collect under the current
        policy, fit, repeat.

        Args:
            hidden_states: Output of :meth:`_run_blocks`, one per layer.
            exit_layers: Sampled exit depth per position, shaped
                ``(batch, seq_len)``.
            input_ids: Token ids, needed to re-run the simulation.

        Returns:
            Detached exit states shaped ``(batch, seq_len, d_model)``.
        """
        if self.config.kv_exposure == "simulated":
            return self.simulate_early_exit(input_ids, exit_layers).detach()

        stacked = torch.stack([state.detach() for state in hidden_states])
        index = exit_layers.unsqueeze(0).unsqueeze(-1).expand(
            1, *exit_layers.shape, stacked.size(-1)
        )
        return torch.gather(stacked, 0, index).squeeze(0)

    def _select_exits(self, n_exits: int) -> list[int]:
        """Chooses which exits contribute to the loss on this step.

        With ``exits_per_step`` set, exits are visited in a deterministic
        rotation rather than sampled randomly. Two properties follow, and both
        matter under DDP: every rank picks the same exits without having to
        share a random seed, and consecutive steps cover the whole stack
        instead of leaving some exit starved by chance.

        Being deterministic in the step, the rotation also aliases against any
        other schedule keyed on the step. With five non-final exits and a budget
        of two, the choice depends only on ``step % 5``, so an evaluation every
        500 steps lands on the same position every time and scores the same two
        exits for an entire run -- which is exactly what happened, leaving
        depths 6, 8 and 10 with no held-out number after 38,140 steps. Callers
        that are not a training step should hold :meth:`score_all_exits`.

        Args:
            n_exits: Total number of exits, including the final one.

        Returns:
            Ascending indices into :attr:`exit_modules`. The final exit is
            always present, since it anchors both the loss and the
            distillation teacher.
        """
        budget = None if self._score_every_exit else self.config.exits_per_step
        last = n_exits - 1
        if budget is None or budget >= last:
            return list(range(n_exits))

        step = int(self._step_counter.item())
        chosen = {(step * budget + offset) % last for offset in range(budget)}
        chosen.add(last)
        return sorted(chosen)

    @contextmanager
    def score_all_exits(self):
        """Scores every exit for the duration of the block.

        ``exits_per_step`` exists to bound memory during training: logits are
        ``batch x seq_len x vocab`` per scored exit, and ``cross_entropy``
        retains its log-softmax for the backward pass, so they cannot be freed
        as the loop advances. Under :func:`torch.no_grad` nothing is retained
        and that reason is gone, leaving only the rotation's aliasing against
        the evaluation schedule -- a cost with no remaining benefit.

        Evaluation is therefore the intended caller. It costs the extra forward
        work of the unscored exits, which is real at this width: one vocabulary
        projection is about 5.5 blocks at ``d_model=768``. It buys a held-out
        number for every tier instead of for whichever ones the rotation
        happened to align with.

        Yields:
            Nothing. The previous setting is restored on exit, including when
            the block raises.
        """
        previous = self._score_every_exit
        self._score_every_exit = True
        try:
            yield
        finally:
            self._score_every_exit = previous

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ExitOutput:
        """Runs the model at full depth and scores every participating exit.

        Training always runs the whole stack: an exit cannot learn to predict
        from a depth it never sees. Early exiting is applied at generation
        time by :meth:`generate`.

        The objective is a depth-weighted sum over exits. Each shallow exit is
        trained both on the true next token and on the final layer's
        distribution, the latter detached so the teacher is not dragged toward
        its students. Without that distillation term shallow exits stay weak,
        rarely clear the confidence threshold, and the whole mechanism idles.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            targets: Optional gold token ids of the same shape. Positions equal
                to ``-100`` are ignored by the loss.
            cache: Optional KV cache. When provided it is updated in place and
                ``input_ids`` is treated as a continuation of the cached
                prefix.

        Returns:
            An :class:`ExitOutput` carrying the final exit's logits, the
            combined loss, and each exit's own cross-entropy for logging.

        Raises:
            ValueError: If the combined cached and incoming length exceeds
                ``config.max_seq_len``.

        Note:
            Logits are the memory bottleneck here, not activations: every exit
            produces ``batch * seq_len * vocab_size`` values and
            ``cross_entropy`` holds its log-softmax for the backward pass, so
            they cannot simply be freed as the loop advances. Twelve exits over
            a batch of 2 by 1024 tokens with a 50k vocabulary is roughly 5 GB.
            ``config.exits_per_step`` trims that by scoring a rotating subset.
        """
        offset = len(cache) if cache is not None else 0
        self._check_length(offset, input_ids.size(1))

        all_hidden = self._run_blocks(input_ids, offset, cache)
        hidden_states = [all_hidden[layer] for layer in self.config.exit_layers]
        final_logits = self._readout(len(self.exit_modules) - 1, hidden_states[-1])

        if targets is None:
            return ExitOutput(logits=final_logits)

        weights = self.config.exit_weights
        selected = self._select_exits(len(hidden_states))
        last = len(hidden_states) - 1

        # Scoring a subset shrinks the loss, so the sampled exits stand in for
        # the ones left out. Redistributing the *total* shallow weight across
        # whichever exits were picked keeps the objective's scale identical on
        # every step. Scaling by the plain count ratio instead would also be
        # correct in expectation, but the weights differ sharply by depth, so
        # the per-step total would wander and drag gradient clipping and the
        # learning-rate schedule around with it.
        sampled = [i for i in selected if i != last]
        rescale = 1.0
        if sampled and len(sampled) < last:
            shallow_total = sum(weights[:last])
            sampled_total = sum(weights[i] for i in sampled)
            if shallow_total > 0.0 and sampled_total > 0.0:
                rescale = shallow_total / sampled_total

        flat_targets = targets.reshape(-1)
        valid = targets != -100
        teacher = final_logits.detach() if self.config.self_distill_weight else None

        total = final_logits.new_zeros(())
        exit_losses: dict[int, float] = {}

        for i in selected:
            logits = (
                final_logits if i == last else self._readout(i, hidden_states[i])
            )
            layer = self.config.exit_layers[i]

            cross_entropy = F.cross_entropy(
                logits.view(-1, logits.size(-1)), flat_targets, ignore_index=-100
            )
            exit_losses[layer] = float(cross_entropy.detach())

            term = cross_entropy
            if teacher is not None and i != last:
                term = term + self.config.self_distill_weight * self._distillation(
                    logits, teacher, valid
                )

            weight = weights[i] * (rescale if i != last else 1.0)
            total = total + weight * term

        kv_loss = None
        if (
            self.config.learned_kv_propagation
            and self.config.kv_propagation_weight != 0.0
        ):
            propagation = self._propagation_loss(all_hidden, valid, input_ids)
            if propagation is not None:
                total = total + self.config.kv_propagation_weight * propagation
                kv_loss = float(propagation.detach())

        if self.training:
            self._step_counter += 1

        return ExitOutput(
            logits=final_logits,
            loss=total,
            exit_losses=exit_losses,
            kv_loss=kv_loss,
        )

    def _distillation(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Measures how far an exit is from the final layer's distribution.

        Uses the forward KL from teacher to student, which penalizes a student
        for putting no mass where the teacher puts some. That direction is what
        transfers the teacher's ranking over plausible alternatives rather than
        just its argmax, which is the whole reason distillation beats training
        the shallow exits on hard targets alone.

        Args:
            student_logits: Shallow exit's logits, shaped
                ``(batch, seq_len, vocab_size)``.
            teacher_logits: Detached final-layer logits, same shape.
            valid: Positions that are not padding, shaped ``(batch, seq_len)``.

        Returns:
            A scalar, already scaled by the squared temperature so its gradient
            stays comparable to the cross-entropy term as the temperature
            changes.
        """
        temperature = self.config.self_distill_temperature
        student = F.log_softmax(student_logits / temperature, dim=-1)
        teacher = F.log_softmax(teacher_logits / temperature, dim=-1)

        divergence = F.kl_div(
            student, teacher, log_target=True, reduction="none"
        ).sum(dim=-1)
        divergence = (divergence * valid).sum() / valid.sum().clamp(min=1)
        return divergence * temperature**2

    @torch.no_grad()
    def exit_statistics(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> ExitStatistics:
        """Records how every exit behaves on a teacher-forced batch.

        Logits are reduced to scalars per position as they are produced, so
        peak memory holds one exit's logits rather than all of them. The result
        is small enough to sweep thresholds over repeatedly without touching
        the model again.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            targets: Gold token ids of the same shape, ``-100`` where padded.

        Returns:
            An :class:`ExitStatistics` describing every exit at every position.
        """
        self._check_length(0, input_ids.size(1))
        all_hidden = self._run_blocks(input_ids)
        hidden_states = [all_hidden[layer] for layer in self.config.exit_layers]

        safe_targets = targets.clamp(min=0)
        valid = targets != -100

        uncertainties, corrects, predictions, nlls = [], [], [], []
        for i, hidden in enumerate(hidden_states):
            logits = self._readout(i, hidden)
            uncertainties.append(uncertainty(logits, self.config.exit_criterion))
            predictions.append(logits.argmax(dim=-1))
            corrects.append(predictions[-1] == safe_targets)
            nlls.append(
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(targets)
            )

        stacked_predictions = torch.stack(predictions)
        return ExitStatistics(
            uncertainty=torch.stack(uncertainties),
            correct=torch.stack(corrects) & valid,
            agrees_with_final=(stacked_predictions == stacked_predictions[-1]) & valid,
            nll=torch.stack(nlls),
            valid=valid,
            exit_layers=self.config.exit_layers,
            n_layers=self.config.n_layers,
        )

    def _decode_step(
        self,
        token: torch.Tensor,
        cache: KVCache,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advances one token, stopping each sequence as soon as it is sure.

        A layer runs for the whole batch whenever any row still needs it, so
        rows that already exited would otherwise pick up keys and values they
        were never supposed to have. Those entries are overwritten with the
        propagated ones, which keeps a row's cache identical to what it would
        be if it were decoded alone. Wall-clock savings therefore appear once
        every row has exited, or at batch size one; the accuracy of the
        measurement does not depend on that.

        Args:
            token: The single new token per sequence, shaped ``(batch, 1)``.
            cache: Cache holding everything before this token.
            threshold: Uncertainty below which a sequence stops.

        Returns:
            A tuple ``(logits, exit_layers)`` with shapes ``(batch, vocab)``
            and ``(batch,)``.
        """
        batch = token.size(0)
        offset = len(cache)
        self._check_length(offset, 1)

        x = self.embed(token)
        frozen = x
        exited = torch.zeros(batch, dtype=torch.bool, device=token.device)
        exit_layers = torch.full(
            (batch,), self.config.n_layers - 1, dtype=torch.long, device=token.device
        )
        logits = x.new_zeros(batch, self.config.vocab_size)

        for layer_idx, block in enumerate(self.blocks):
            skipping = not self.config.full_depth_kv
            # How far each row's frozen state has travelled to reach this
            # layer's input. Rows exit at different depths, so this is per-row.
            gap = (layer_idx - 1 - exit_layers).clamp(min=0)

            if skipping and bool(exited.all()):
                # Nothing in the batch needs this layer: pay only for the
                # key/value projections so later tokens can still attend here.
                block.propagate_kv(frozen, self.rope, offset, cache, gap)
                continue

            updated = block(x, self.rope, offset, cache)

            if skipping and bool(exited.any()):
                keys, values = block.project_kv(frozen, self.rope, offset, gap)
                cache.overwrite_last(layer_idx, exited, keys, values)
                x = torch.where(exited.view(batch, 1, 1), frozen, updated)
            else:
                x = updated

            if layer_idx not in self.exit_index:
                continue

            is_final = layer_idx == self.config.n_layers - 1
            candidate = self._readout(self.exit_index[layer_idx], x[:, -1])

            if is_final:
                fires = ~exited
            elif layer_idx < self.config.min_exit_layer:
                continue
            else:
                score = uncertainty(candidate, self.config.exit_criterion)
                fires = (~exited) & (score < threshold)

            if bool(fires.any()):
                logits = torch.where(fires.unsqueeze(-1), candidate, logits)
                exit_layers = torch.where(fires, layer_idx, exit_layers)
                frozen = torch.where(fires.view(batch, 1, 1), x, frozen)
                exited = exited | fires

        return logits, exit_layers

    @torch.inference_mode()
    def trace_decode(
        self,
        input_ids: torch.Tensor,
        threshold: float | None = None,
        refresh_every: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes a fixed token sequence one step at a time, under early exit.

        This is the instrument for measuring drift. Because the tokens are
        given rather than sampled, the context is identical to what a
        full-depth pass would see, so any difference in the logits comes purely
        from the cache having been filled with propagated keys and values. That
        isolates compounding cache error from the ordinary divergence caused by
        sampling a different token.

        Compare against ``self(input_ids).logits``, which is the exact
        full-depth reference for the same positions.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            threshold: Exit threshold, defaulting to the configured one.
            refresh_every: Force a full-depth token this often, defaulting to
                the configured value. ``0`` disables refreshing.

        Returns:
            A tuple ``(logits, exit_layers)`` shaped
            ``(batch, seq_len, vocab_size)`` and ``(batch, seq_len)``, where
            entry ``i`` is the prediction made after consuming token ``i``.

        Raises:
            ValueError: If ``input_ids`` holds fewer than two tokens.
        """
        if input_ids.size(1) < 2:
            raise ValueError(
                f"trace_decode needs at least 2 tokens, got {input_ids.size(1)}."
            )

        threshold = self.config.exit_threshold if threshold is None else threshold
        refresh_every = (
            self.config.refresh_every if refresh_every is None else refresh_every
        )

        self.eval()
        cache = KVCache(self.config.n_layers)

        # The first token is prefilled at full depth, matching generation.
        step_logits = [self(input_ids[:, :1], cache=cache).logits[:, -1]]
        depths = [
            torch.full(
                (input_ids.size(0),),
                self.config.n_layers - 1,
                dtype=torch.long,
                device=input_ids.device,
            )
        ]

        for position in range(1, input_ids.size(1)):
            logits, exits = self._decode_step(
                input_ids[:, position : position + 1],
                cache,
                self._threshold_at(position, threshold, refresh_every),
            )
            step_logits.append(logits)
            depths.append(exits)

        return torch.stack(step_logits, dim=1), torch.stack(depths, dim=1)

    def _threshold_at(
        self,
        position: int,
        threshold: float,
        refresh_every: int,
    ) -> float:
        """Returns the exit threshold for one decoding position.

        On refresh positions the threshold drops to zero. Since uncertainty is
        non-negative, nothing can then satisfy ``uncertainty < 0``, so the
        token runs the full stack and writes exact keys and values without
        needing a separate code path.

        Args:
            position: Index of the token being decoded.
            threshold: The normal exit threshold.
            refresh_every: Refresh period, or ``0`` to never refresh.

        Returns:
            The threshold to use at this position.
        """
        if refresh_every and position % refresh_every == 0:
            return 0.0
        return threshold

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        threshold: float | None = None,
        refresh_every: int | None = None,
    ) -> GenerationOutput:
        """Samples a continuation, letting each token choose its own depth.

        The prompt is encoded in one full-depth pass. Prompt tokens are
        processed in parallel, so exiting them individually would save no time
        while greatly complicating the cache; only generated tokens exit early.

        Args:
            input_ids: Prompt token ids shaped ``(batch, prompt_len)``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature. Values below one sharpen the
                distribution; ``0.0`` selects the argmax deterministically.
            top_k: When set, sampling is restricted to the ``top_k`` most
                likely tokens at each step.
            eos_token_id: When set, generation stops early once every sequence
                in the batch has emitted this token.
            threshold: Overrides ``config.exit_threshold`` for this call, which
                is how a calibration sweep explores the accuracy/depth
                tradeoff. ``0.0`` disables early exiting entirely.
            refresh_every: Overrides ``config.refresh_every``. Forcing a
                periodic full-depth token bounds how far the residual stream
                can drift from propagated states over a long generation.

        Returns:
            A :class:`GenerationOutput` with the completed sequences and the
            layer each generated token stopped at.
        """
        self.eval()
        if threshold is None:
            threshold = self.config.exit_threshold
        if refresh_every is None:
            refresh_every = self.config.refresh_every

        cache = KVCache(self.config.n_layers)
        # Only the last position's readout is needed to continue, and the
        # vocabulary projection is the most expensive single operation here, so
        # the prompt is not projected in full. This also makes the prefill
        # readout bit-identical to the one generate_routed performs.
        prompt_state = self.forward_to_depth(
            input_ids, self.config.n_layers, cache=cache
        )
        logits = self.endpoint_logits(prompt_state.last_token, self.config.n_layers)

        generated = input_ids
        finished = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=input_ids.device
        )
        # The prompt runs at full depth, so the first continuation is credited
        # to the last layer rather than to an exit that never got to vote.
        depths = [
            torch.full(
                (input_ids.size(0),),
                self.config.n_layers - 1,
                dtype=torch.long,
                device=input_ids.device,
            )
        ]

        for step in range(max_new_tokens):
            next_token = self._sample(logits, temperature, top_k)

            if eos_token_id is not None:
                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1), eos_token_id
                )
                finished |= next_token.squeeze(-1) == eos_token_id

            generated = torch.cat((generated, next_token), dim=1)

            if step == max_new_tokens - 1 or (
                eos_token_id is not None and bool(finished.all())
            ):
                break

            logits, exit_layers = self._decode_step(
                next_token,
                cache,
                self._threshold_at(step + 1, threshold, refresh_every),
            )
            depths.append(exit_layers)

        return GenerationOutput(
            sequences=generated, exit_layers=torch.stack(depths, dim=1)
        )

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draws one token per sequence from a logit distribution.

        Args:
            logits: Scores for the next token, shaped ``(batch, vocab_size)``.
            temperature: Softmax temperature; ``0.0`` means greedy.
            top_k: Optional truncation to the ``top_k`` most likely tokens.
            generator: Random source. Supplying one is what lets two runs that
                group requests differently still be compared, since the global
                stream would otherwise be consumed in a different order.

        Returns:
            Token ids shaped ``(batch, 1)``.
        """
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            kth_value = logits.topk(k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth_value, -float("inf"))
        return torch.multinomial(
            F.softmax(logits, dim=-1), num_samples=1, generator=generator
        )

    # ------------------------------------------------------------------
    # Request-level vertical routing
    # ------------------------------------------------------------------

    def attach_router(
        self,
        routing: RoutingConfig,
        controller: DepthController | None = None,
    ) -> DepthController | None:
        """Installs a routing configuration and its controller.

        Args:
            routing: Routing settings. Resolved against this model, so an
                invalid combination is rejected here rather than mid-generation.
            controller: Trained controller. When ``None`` and the mode needs
                one, a fresh controller is built — untrained, so its routing is
                only as meaningful as its initialization, which is why
                :meth:`generate_routed` records that as a fallback reason.

        Returns:
            The attached controller, or ``None`` for modes that need none.
        """
        resolved = routing.resolve(self.config)
        self.routing = resolved

        if resolved.routing_mode != "request":
            self.depth_controller = controller
            return controller

        if controller is None:
            controller = build_controller(self.config, resolved)
        self.depth_controller = controller.to(self.embed.weight.device)
        return self.depth_controller

    @staticmethod
    def _row_groups(values: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        """Groups row indices by a per-row integer value.

        Args:
            values: One value per row, shaped ``(batch,)``.

        Returns:
            ``(value, row_indices)`` pairs in ascending value order.
        """
        groups = []
        for value in sorted(set(values.tolist())):
            rows = torch.nonzero(values == value, as_tuple=False).flatten()
            groups.append((int(value), rows))
        return groups

    @torch.inference_mode()
    def generate_routed(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        prompt_lengths: torch.Tensor | None = None,
        routing_lambda: float | None = None,
        depth: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
        return_routing_trace: bool = True,
        generator: torch.Generator | None = None,
    ) -> RoutedGenerationOutput:
        """Generates with one depth chosen per request.

        The sequence of events is: run the prompt through the probe blocks,
        pool their output, choose a depth, finish the prefill only to that
        depth, and decode there with a cache that never allocates anything
        above it.

        **Requests are grouped by prompt length, then by chosen depth, and each
        group runs separately.** That is what makes the result exact. A single
        padded batch would let one row's padding enter another's attention, and
        a single cache cannot hold different depths for different rows. Under
        greedy decoding the tokens are therefore identical to routing each
        request on its own — which is the property the tests check.

        This is not a claim about throughput. Grouping in Python serializes
        what a server would overlap, and the loop here is written to be
        correct, countable, and easy to profile rather than fast. What it does
        establish is that the depth cap is real: no upper-layer keys or values
        exist to be counted, so the memory saving can be measured rather than
        asserted.

        Args:
            input_ids: Prompt ids shaped ``(batch, prompt_len)``, right-padded
                when lengths differ.
            max_new_tokens: Tokens to append.
            prompt_lengths: Real length of each prompt, shaped ``(batch,)``.
                ``None`` treats every row as full length.
            routing_lambda: Price of compute for this call, overriding the
                configured value. This is the knob a live service would turn,
                and it needs no retraining because the controller predicts
                quality and the cost is supplied separately.
            depth: Forces every request to this depth, overriding the router.
                Used for fixed-depth baselines, which must run through exactly
                this path so endpoint quality is not confounded with routing
                quality.
            temperature: Sampling temperature; ``0.0`` is greedy.
            top_k: Optional truncation before sampling.
            eos_token_id: Stops a sequence once it emits this token.
            pad_token_id: Fill for rows that finish early or start shorter.
            return_routing_trace: Whether to build the trace.
            generator: Random source for sampling.

        Returns:
            A :class:`RoutedGenerationOutput`.

        Raises:
            ValueError: If no routing configuration has been attached, or if a
                ``"request"`` run has no controller.
        """
        if self.routing is None:
            raise ValueError(
                "No routing configuration attached. Call "
                "model.attach_router(RoutingConfig(...)) first."
            )

        self.eval()
        routing = self.routing
        batch, padded_len = input_ids.shape
        device = input_ids.device

        if prompt_lengths is None:
            prompt_lengths = torch.full(
                (batch,), padded_len, dtype=torch.long, device=device
            )
        prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)

        cost_model = AnalyticalCostModel.from_config(self.config)
        counters = CostCounters()
        trace = RoutingTrace(
            tiers=routing.selectable_tiers, probe_depth=routing.probe_depth
        )
        # Per-request slots, filled out of order because requests are grouped.
        depths = torch.zeros(batch, dtype=torch.long, device=device)
        scores: list[list[float]] = [[] for _ in range(batch)]
        reasons: list[str | None] = [None] * batch
        kv_bytes = [0] * batch
        boundary_bytes = [0] * batch

        output = torch.full(
            (batch, padded_len + max_new_tokens),
            pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )

        head_before = self.head_calls, self.head_tokens

        for length, rows in self._row_groups(prompt_lengths):
            prompt = input_ids[rows, :length]
            chosen, group_scores, reason, probed = self._route_group(
                prompt, length, routing, routing_lambda, depth, counters
            )

            for offset, row in enumerate(rows.tolist()):
                depths[row] = int(chosen[offset])
                scores[row] = [float(v) for v in group_scores[offset]]
                reasons[row] = reason

            # Only a run that actually consulted the controller pays for a
            # probe. Charging one in fixed or unrouted mode would leave the
            # total block count right but split it wrongly, and the probe cost
            # is exactly the number a reader uses to judge whether routing is
            # worth its overhead.
            probe_depth = (
                min(routing.probe_depth, int(chosen.min())) if probed else 0
            )
            if probe_depth:
                counters.record_prefill(probe_depth, length, rows.numel())
                trace.probe_blocks += probe_depth * length * rows.numel()

            for tier, local in self._row_groups(chosen):
                bucket = rows[local]
                generated, bucket_kv, bucket_boundary = self._generate_bucket(
                    input_ids[bucket, :length],
                    tier,
                    probe_depth,
                    routing,
                    max_new_tokens,
                    temperature,
                    top_k,
                    eos_token_id,
                    pad_token_id,
                    generator,
                    counters,
                    trace,
                )
                output[bucket, :length] = input_ids[bucket, :length]
                output[bucket, length : length + generated.size(1)] = generated
                for row in bucket.tolist():
                    kv_bytes[row] = bucket_kv
                    boundary_bytes[row] = bucket_boundary

        counters.head_calls = self.head_calls - head_before[0]
        counters.head_tokens = self.head_tokens - head_before[1]

        trace.depths = depths.tolist()
        trace.scores = scores
        trace.fallback_reasons = reasons
        trace.kv_bytes = kv_bytes
        trace.boundary_bytes = boundary_bytes
        trace.head_calls = counters.head_calls
        trace.head_tokens = counters.head_tokens
        trace.controller_calls = counters.controller_calls

        controller = self.depth_controller
        breakdown = counters.estimated_macs(
            cost_model,
            adapter_rank=self.config.kv_adapter_rank,
            controller_feature_dim=controller.feature_dim if controller else 0,
            controller_hidden=controller.hidden_dim if controller else 0,
            n_tiers=len(routing.selectable_tiers),
        )

        return RoutedGenerationOutput(
            sequences=output,
            depths=depths,
            prompt_lengths=prompt_lengths,
            max_new_tokens=max_new_tokens,
            trace=trace if return_routing_trace else None,
            counters=counters,
            estimated_macs=breakdown,
        )

    def _route_group(
        self,
        prompt: torch.Tensor,
        length: int,
        routing: RoutingConfig,
        routing_lambda: float | None,
        forced_depth: int | None,
        counters: CostCounters,
    ) -> tuple[torch.Tensor, torch.Tensor, str | None, bool]:
        """Chooses a depth for every request in one equal-length group.

        Args:
            prompt: Prompts shaped ``(rows, length)``, no padding.
            length: Prompt length.
            routing: Resolved routing settings.
            routing_lambda: Per-call cost price, or ``None`` for the
                configured one.
            forced_depth: Depth to use regardless of the controller.
            counters: Counters to record the controller evaluation in.

        Returns:
            A tuple ``(depths, scores, fallback_reason, probed)``. ``scores`` is
            shaped ``(rows, n_tiers)`` and is all zeros when no controller ran;
            ``probed`` says whether the probe blocks were executed, which is
            what decides who pays for them.

        Raises:
            ValueError: If a forced depth is not among the tiers, or if request
                routing was asked for without a controller.
        """
        rows = prompt.size(0)
        tiers = routing.selectable_tiers
        empty = prompt.new_zeros((rows, len(tiers)), dtype=torch.float)

        def constant(depth: int, reason: str | None):
            """A depth chosen without consulting the controller, so no probe."""
            return (
                torch.full((rows,), depth, dtype=torch.long, device=prompt.device),
                empty,
                reason,
                False,
            )

        if forced_depth is not None:
            if forced_depth not in routing.depth_tiers:
                raise ValueError(
                    f"depth={forced_depth} is not one of the configured tiers "
                    f"{routing.depth_tiers}."
                )
            return constant(forced_depth, "forced_depth")

        if routing.routing_mode == "none":
            return constant(self.config.n_layers, "routing_disabled")
        if routing.routing_mode == "fixed":
            return constant(routing.fixed_depth, "fixed_mode")

        if self.depth_controller is None:
            raise ValueError(
                "routing_mode is 'request' but no controller is attached. "
                "Pass one to attach_router, or load one with controller_path."
            )

        # The probe is the only thing the controller may see. It runs without a
        # cache: the group is about to be split by depth, and each bucket needs
        # its own depth-capped cache anyway, so a shared probe cache would have
        # to be sliced apart immediately.
        probe = self.forward_to_depth(prompt, routing.probe_depth)
        features = pool_prompt_features(
            probe.hidden,
            lengths=None,
            pooling=routing.controller_pooling,
            include_length=routing.controller_use_length,
            max_seq_len=self.config.max_seq_len,
        )
        counters.controller_calls += rows

        lam = routing.routing_lambda if routing_lambda is None else routing_lambda
        depths, scores = self.depth_controller.select(
            features,
            tiers,
            routing_lambda=lam,
            sufficiency_threshold=routing.sufficiency_threshold,
            deterministic=routing.deterministic_routing,
            generator=None,
        )

        reason = None
        if routing.safety_depth is not None:
            raised = depths < routing.safety_depth
            if bool(raised.any()):
                depths = depths.clamp(min=routing.safety_depth)
                reason = "safety_depth_floor"
        return depths, scores, reason, True

    def _generate_bucket(
        self,
        prompt: torch.Tensor,
        depth: int,
        probe_depth: int,
        routing: RoutingConfig,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        eos_token_id: int | None,
        pad_token_id: int,
        generator: torch.Generator | None,
        counters: CostCounters,
        trace: RoutingTrace,
    ) -> tuple[torch.Tensor, int, int]:
        """Prefills and decodes one bucket of requests at a single depth.

        Args:
            prompt: Prompts shaped ``(rows, length)``.
            depth: Executed depth for this bucket.
            probe_depth: Blocks already accounted for by the probe.
            routing: Resolved routing settings.
            max_new_tokens: Tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional truncation.
            eos_token_id: Stop token.
            pad_token_id: Fill for finished rows.
            generator: Random source.
            counters: Counters to record into.
            trace: Trace to record block counts into.

        Returns:
            A tuple ``(tokens, kv_bytes, boundary_bytes)`` where ``tokens`` is
            shaped ``(rows, generated)``.
        """
        rows, length = prompt.shape
        cache = KVCache(self.config.n_layers, max_depth=depth)

        state = self.forward_to_depth(
            prompt,
            depth,
            cache=cache,
            return_boundary_state=routing.retain_boundary_state,
        )
        # The probe blocks were already counted for the whole group, so only
        # the remainder is charged here. Double counting them would make deep
        # routes look cheaper than they are relative to shallow ones.
        counters.record_prefill(
            depth, length, rows, start_depth=min(probe_depth, depth)
        )
        trace.endpoint_blocks += max(depth - probe_depth, 0) * length * rows

        logits = self.endpoint_logits(state.last_token, depth)

        tokens: list[torch.Tensor] = []
        finished = torch.zeros(rows, dtype=torch.bool, device=prompt.device)

        for step in range(max_new_tokens):
            next_token = self._sample(logits, temperature, top_k, generator)
            if eos_token_id is not None:
                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1), pad_token_id
                )
                finished |= next_token.squeeze(-1) == eos_token_id
            tokens.append(next_token)

            if step == max_new_tokens - 1 or bool(finished.all()):
                break

            step_state = self.forward_to_depth(next_token, depth, cache=cache)
            counters.record_blocks(depth, rows, context_len=len(cache))
            counters.decode_tokens += rows
            trace.endpoint_blocks += depth * rows
            logits = self.endpoint_logits(step_state.last_token, depth)

        # Reported per request, not per bucket. A bucket's cache is one tensor
        # covering every row in it, so quoting its size against a single
        # request would make a popular depth look expensive purely for being
        # popular.
        return (
            torch.cat(tokens, dim=1),
            cache.bytes_allocated // rows,
            cache.boundary_bytes // rows,
        )

    @torch.inference_mode()
    def escalate(
        self,
        cache: KVCache,
        boundary: torch.Tensor,
        from_depth: int,
        to_depth: int,
        offset: int = 0,
    ) -> tuple[DepthState, KVCache]:
        """Raises a request's depth using retained prompt state.

        Escalation is only exact if the upper blocks can attend over *every*
        position they would have seen at full depth. The strategy taken here is
        to retain the boundary activation for the whole prompt and replay just
        the suffix over it, which reproduces full-depth execution exactly
        because the retained activation is, by construction, what block
        ``from_depth`` would have received.

        The cost is not hidden: replaying ``T`` positions through
        ``to_depth - from_depth`` blocks is real work, and the returned state
        records it. The alternative strategy — recomputing the lower prefix —
        would cost more and is not used.

        Args:
            cache: Depth-capped cache built during the original run.
            boundary: Retained activation for every position, shaped
                ``(batch, seq_len, d_model)``.
            from_depth: Depth the request ran at.
            to_depth: Depth to reach.
            offset: Absolute position of the first retained token.

        Returns:
            A tuple ``(state, cache)`` with a new cache capped at
            ``to_depth``. The original cache is left untouched, so a failed
            escalation cannot corrupt the request it started from.

        Raises:
            ValueError: If the target depth is not deeper than the current one.
        """
        if to_depth <= from_depth:
            raise ValueError(
                f"Escalation must go deeper: from_depth={from_depth}, "
                f"to_depth={to_depth}."
            )

        index = torch.arange(boundary.size(0), device=boundary.device)
        widened = cache.select_rows(index, max_depth=to_depth)
        state = self.continue_from_depth(
            boundary, from_depth, to_depth, cache=widened, offset=offset
        )
        return state, widened


if __name__ == "__main__":
    torch.manual_seed(0)

    config = TransformerConfig(
        vocab_size=1_000,
        d_model=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=256,
    )
    model = Transformer(config)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")
    print(f"exits at layers: {config.exit_layers}")

    # Training-style pass: every exit is scored, weighted by its depth.
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    out = model(tokens[:, :-1], targets=tokens[:, 1:])
    print(f"logits: {tuple(out.logits.shape)}  loss: {out.loss.item():.4f}")
    for layer, value in out.exit_losses.items():
        print(f"  layer {layer}: ce {value:.4f}")
    out.loss.backward()

    # Inference-style pass: each token stops as soon as it is confident.
    result = model.generate(
        tokens[:, :4], max_new_tokens=16, temperature=0.8, top_k=50
    )
    print(f"generated: {tuple(result.sequences.shape)}")
    print(f"mean exit layer: {result.mean_exit_layer:.2f} of {config.n_layers}")
