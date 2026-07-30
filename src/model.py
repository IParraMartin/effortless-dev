"""A decoder-only Transformer.

A modern LLM stack: RMSNorm pre-normalization, rotary position embeddings,
grouped-query attention dispatched through PyTorch's fused scaled-dot-product
kernel, SwiGLU feed-forward blocks, and an output projection tied to the input
embedding. Training uses ordinary next-token cross-entropy; generation is
autoregressive with a key/value cache.

Example:
    >>> model = Transformer(TransformerConfig(vocab_size=32000))
    >>> tokens = torch.randint(0, 32000, (2, 128))
    >>> out = model(tokens[:, :-1], targets=tokens[:, 1:])
    >>> out.logits.shape
    torch.Size([2, 127, 32000])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TransformerConfig
from src.modules import KVCache, RMSNorm, RotaryEmbedding


@dataclass
class ModelOutput:
    """Result of a forward pass.

    Attributes:
        logits: Next-token logits shaped ``(batch, seq_len, vocab_size)``.
        loss: Mean cross-entropy over non-ignored positions, or ``None`` when
            no targets were supplied.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class Attention(nn.Module):
    """Causal grouped-query self-attention.

    Queries are projected to ``n_heads`` heads while keys and values are
    projected to a smaller number of ``n_kv_heads`` heads that each serve a
    group of queries. The scaled dot product is delegated to
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
            offset: Absolute position of the first token of ``x``. Supplied by
                the caller rather than read from ``cache``, because shallower
                layers have already extended the cache by the time this one runs.
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


class Transformer(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        config: Model configuration. Defaults to a ~124M parameter setup.
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
        self.norm = RMSNorm(self.config.d_model, eps=self.config.norm_eps)
        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )

        self.apply(self._init_weights)
        self._scale_residual_projections()

        if self.config.tie_embeddings:
            # The output projection reuses the input embedding matrix. Both are
            # (vocab_size, d_model), so the assignment shares one tensor rather
            # than copying, and it happens last so no initialization pass undoes
            # the tie.
            self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes a single submodule in place.

        Linear and embedding weights are drawn from a zero-mean normal with
        standard deviation ``config.init_std``; biases, where present, start at
        zero. RMSNorm gains are left at their constructor value of one.

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
                block.attn.o_proj.weight, mean=0.0, std=self.config.init_std * scale
            )
            nn.init.normal_(
                block.ffn.down_proj.weight, mean=0.0, std=self.config.init_std * scale
            )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Counts the model's parameters.

        Args:
            trainable_only: When ``True``, frozen parameters are excluded.

        Returns:
            The total number of parameters. A weight shared between the
            embedding and the output projection is counted once.
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ModelOutput:
        """Runs the model and, given targets, computes the training loss.

        Args:
            input_ids: Token ids shaped ``(batch, seq_len)``.
            targets: Optional gold token ids of the same shape. Positions equal
                to ``-100`` are ignored by the loss.
            cache: Optional KV cache. When provided it is updated in place and
                ``input_ids`` is treated as a continuation of the cached prefix.

        Returns:
            A :class:`ModelOutput` with logits and, when targets are given, the
            mean cross-entropy.

        Raises:
            ValueError: If the combined cached and incoming length exceeds
                ``config.max_seq_len``.
        """
        offset = len(cache) if cache is not None else 0
        self._check_length(offset, input_ids.size(1))

        x = self.embed_dropout(self.embed(input_ids))
        for block in self.blocks:
            x = block(x, self.rope, offset, cache)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(logits=logits, loss=loss)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Samples a continuation autoregressively.

        The prompt is encoded in one pass, filling a key/value cache; each
        subsequent token is produced from that cache in a single-token forward.

        Args:
            input_ids: Prompt token ids shaped ``(batch, prompt_len)``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature. Values below one sharpen the
                distribution; ``0.0`` selects the argmax deterministically.
            top_k: When set, sampling is restricted to the ``top_k`` most likely
                tokens at each step.
            eos_token_id: When set, once a sequence emits this token it is padded
                with it and generation stops early if every sequence has emitted
                it.

        Returns:
            The prompt followed by the generated tokens, shaped
            ``(batch, prompt_len + generated_len)``.
        """
        self.eval()
        cache = KVCache(self.config.n_layers)
        logits = self(input_ids, cache=cache).logits[:, -1, :]

        generated = input_ids
        finished = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=input_ids.device
        )

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

            logits = self(next_token, cache=cache).logits[:, -1, :]

        return generated

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        """Draws one token per sequence from a logit distribution.

        Args:
            logits: Scores for the next token, shaped ``(batch, vocab_size)``.
            temperature: Softmax temperature; ``0.0`` means greedy.
            top_k: Optional truncation to the ``top_k`` most likely tokens.

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
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


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

    tokens = torch.randint(0, config.vocab_size, (2, 32))
    out = model(tokens[:, :-1], targets=tokens[:, 1:])
    print(f"logits: {tuple(out.logits.shape)}  loss: {out.loss.item():.4f}")
    out.loss.backward()

    sequences = model.generate(tokens[:, :4], max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"generated: {tuple(sequences.shape)}")
