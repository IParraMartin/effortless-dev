"""What a routed forward pass actually costs.

"Layers skipped" is not "compute saved", and neither is "compute saved" the
same as "latency saved". This module supplies the first two honestly and refuses
to supply the third: analytical multiply-accumulate counts from
:class:`AnalyticalCostModel`, and counters recording what an execution really
did in :class:`CostCounters`. Measured wall-clock lives in
``experiments/benchmark_latency.py`` and is never mixed into the same column.

Three costs are easy to leave out, and leaving any of them out flatters
routing:

* **The vocabulary head.** One projection costs ``d_model * vocab_size``
  multiply-accumulates. For the repository's 768-wide default with a 52k
  vocabulary that is 39.9M against 7.1M for an entire block — the head is
  **5.6 blocks**. Testing several candidate depths with a full projection each
  can therefore cost more than the depth it saves, which is the whole reason
  the controller reads hidden states instead.
* **Key/value projections for skipped layers.** Token-level propagation still
  pays these at every layer above the exit, so its saving is smaller than the
  block count suggests.
* **The controller itself.** Small, but it must appear, or the comparison is
  against a system that decides for free.

Every formula here omits normalization, rotary application, activation
functions, and elementwise work. Those are linear in ``d_model`` where the
retained terms are quadratic, so the omission is small, but it is an omission
and the analytical numbers are labelled ``estimated`` everywhere they surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import TransformerConfig

#: Bytes per cached element under each precision. Caches are commonly kept at
#: lower precision than the weights, so this is a parameter rather than a
#: constant.
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class AnalyticalCostModel:
    """Closed-form multiply-accumulate counts for one architecture.

    Attributes:
        d_model: Residual width.
        ff_dim: SwiGLU inner width.
        n_heads: Query heads.
        n_kv_heads: Key/value heads.
        head_dim: Per-head width.
        vocab_size: Output vocabulary.
        n_layers: Full depth.
    """

    d_model: int
    ff_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    n_layers: int

    @classmethod
    def from_config(cls, config: TransformerConfig) -> AnalyticalCostModel:
        """Builds a cost model from a model configuration.

        Args:
            config: The architecture to describe.

        Returns:
            A cost model for it.
        """
        return cls(
            d_model=config.d_model,
            ff_dim=config.ff_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            n_layers=config.n_layers,
        )

    @property
    def gqa_ratio(self) -> float:
        """Fraction of query heads that key/value heads are shared across."""
        return self.n_kv_heads / self.n_heads

    @property
    def kv_width(self) -> int:
        """Width of the key (or value) vector cached per token per layer."""
        return self.n_kv_heads * self.head_dim

    @property
    def projection_macs(self) -> float:
        """Attention projection cost per token per block.

        The query and output projections cost ``d_model^2`` each; the key and
        value projections cost ``rho * d_model^2`` each under grouped-query
        attention, which is where GQA's saving shows up.
        """
        return (2.0 + 2.0 * self.gqa_ratio) * self.d_model**2

    @property
    def ffn_macs(self) -> float:
        """Feed-forward cost per token per block: gate, up, and down."""
        return 3.0 * self.d_model * self.ff_dim

    @property
    def kv_projection_macs(self) -> float:
        """Key and value projections alone, per token per block.

        This is the entire cost a layer pays for a token that has already
        exited under token-level propagation, and the reason that scheme saves
        less than its skipped-block count implies.
        """
        return 2.0 * self.gqa_ratio * self.d_model**2

    def attention_macs(self, context_len: int) -> float:
        """Cost of attending over a context, per token per block.

        Args:
            context_len: Number of positions attended to, including the token
                itself.

        Returns:
            Approximately ``2 * context_len * d_model`` multiply-accumulates,
            covering the query-key product and the value-weighted sum.
        """
        return 2.0 * context_len * self.d_model

    def block_macs(self, context_len: int) -> float:
        """Full cost of one decoder block for one token.

        Args:
            context_len: Positions attended to.

        Returns:
            ``(2 + 2*rho) d^2 + 3 d f + 2 T d``.
        """
        return self.projection_macs + self.ffn_macs + self.attention_macs(context_len)

    @property
    def head_macs(self) -> float:
        """Cost of one full vocabulary projection for one token.

        Weight tying saves parameters, not these multiplications.
        """
        return float(self.d_model * self.vocab_size)

    @property
    def head_to_block_ratio(self) -> float:
        """How many blocks one vocabulary projection costs, ignoring context.

        Above one, evaluating an extra exit is more expensive than running an
        extra block, and a policy that reads the vocabulary at every checkpoint
        cannot pay for itself.
        """
        return self.head_macs / (self.projection_macs + self.ffn_macs)

    def adapter_macs(self, rank: int) -> float:
        """Cost of one key/value propagation adapter for one token.

        Args:
            rank: Bottleneck width.

        Returns:
            The down and up projections; the gap embedding is a lookup.
        """
        return 2.0 * self.d_model * rank

    def exit_adapter_macs(self, rank: int) -> float:
        """Cost of one exit adapter for one token.

        A retrofit that adds these and does not charge for them reports a
        frontier its endpoints do not sit on. The amount is small — at
        ``d_model=768`` a rank-32 adapter is 49,152 MACs against a block's 8.7M,
        so about 0.57% of a block — but it lands entirely on the cheap end of the
        frontier, which is where the shallow endpoints being justified live.

        Args:
            rank: Bottleneck width.

        Returns:
            The down and up projections. One adapter runs per readout, not per
            block, so this is charged alongside :attr:`head_macs`.
        """
        return 2.0 * self.d_model * rank

    def lora_macs(self, rank: int, targets_per_block: int) -> float:
        """Cost of the low-rank updates inside one block, for one token.

        Args:
            rank: Rank of each update.
            targets_per_block: Wrapped projections in the block.

        Returns:
            Estimated multiply-accumulates. Each update is a ``d_model x rank``
            down-projection followed by a ``rank x d_model`` up-projection, so it
            scales with rank and with how many projections were wrapped. Unlike
            the exit adapter this is charged *per executed block*, so it grows
            with the routed depth.
        """
        return targets_per_block * 2.0 * self.d_model * rank

    def prefill_macs(self, depth: int, prompt_len: int) -> float:
        """Cost of running a prompt through the first ``depth`` blocks.

        Args:
            depth: Blocks executed.
            prompt_len: Prompt length in tokens.

        Returns:
            Estimated multiply-accumulates. Position ``t`` attends over ``t+1``
            positions, so the context term sums to ``d * T * (T + 1)``.
        """
        per_token = self.projection_macs + self.ffn_macs
        context = self.d_model * prompt_len * (prompt_len + 1)
        return depth * (prompt_len * per_token + context)

    def decode_macs(self, depth: int, context_len: int) -> float:
        """Cost of generating one token through the first ``depth`` blocks.

        Args:
            depth: Blocks executed.
            context_len: Positions attended to, including the new token.

        Returns:
            Estimated multiply-accumulates, excluding the vocabulary head.
        """
        return depth * self.block_macs(context_len)

    def kv_bytes(self, depth: int, seq_len: int, dtype: str = "fp32") -> int:
        """Cache memory a depth-capped request occupies.

        Args:
            depth: Layers whose keys and values are materialized.
            seq_len: Positions cached.
            dtype: Cache precision, a key of :data:`DTYPE_BYTES`.

        Returns:
            Bytes. Exactly proportional to ``depth``, which is the formal
            content of the request-level memory claim — and the reason a
            method that synthesizes entries for skipped layers saves none of
            it.

        Raises:
            KeyError: If ``dtype`` is not a known precision.
        """
        return 2 * DTYPE_BYTES[dtype] * seq_len * depth * self.kv_width

    def controller_macs(
        self,
        feature_dim: int,
        hidden_dim: int,
        n_tiers: int,
    ) -> float:
        """Cost of one controller evaluation for one request.

        Args:
            feature_dim: Width of the pooled probe features.
            hidden_dim: Bottleneck width.
            n_tiers: Number of candidate depths scored.

        Returns:
            Estimated multiply-accumulates. Compare against
            :attr:`head_macs`: this is the arithmetic reason to route on hidden
            states rather than on a vocabulary distribution.
        """
        return float(feature_dim * hidden_dim + hidden_dim * n_tiers)


@dataclass(frozen=True)
class KVAudit:
    """A cache measured against what the depth cap promised.

    The request-level memory claim is a proof, not an estimate: with no entries
    materialized above the cap, cache memory is ``2 b T d rho d_model`` and the
    saving is exactly ``1 - d/L``. What a proof cannot establish is that the
    implementation obeys it. Allocating upper layers and never writing to them,
    or writing a zero-filled placeholder, leaves the arithmetic looking correct
    while the memory saving is zero -- and the failure is invisible in every
    quality metric, because quality is unaffected by memory that is merely
    wasted.

    This is the check the derivation asks for by name. It compares three
    numbers that should agree exactly and are produced by entirely separate
    code paths: what the cache actually holds, what the analytical model says a
    depth-capped cache holds, and what the proportional law predicts.

    Attributes:
        depth: Executed blocks the cache was capped to.
        n_layers: Full depth, for the proportional law.
        seq_len: Positions cached.
        measured_bytes: What the cache holds, read from the cache itself.
        predicted_bytes: What :meth:`AnalyticalCostModel.kv_bytes` expects.
        full_depth_bytes: What an uncapped cache of the same length would hold.
        populated_layers: Indices of layers holding any entries.
    """

    depth: int
    n_layers: int
    seq_len: int
    measured_bytes: int
    predicted_bytes: int
    full_depth_bytes: int
    populated_layers: tuple[int, ...]

    @property
    def leaked_layers(self) -> tuple[int, ...]:
        """Populated layers at or above the cap, which should be none.

        A non-empty result is the failure the memory claim is exposed to: the
        request executed to its depth but the cache materialized above it, so
        the reported saving is not real.
        """
        return tuple(l for l in self.populated_layers if l >= self.depth)

    @property
    def measured_saving(self) -> float:
        """Fraction of full-depth cache memory not materialized."""
        return 1.0 - self.measured_bytes / max(self.full_depth_bytes, 1)

    @property
    def predicted_saving(self) -> float:
        """The proportional law, ``1 - d/L``."""
        return 1.0 - self.depth / max(self.n_layers, 1)

    @property
    def exact(self) -> bool:
        """Whether measurement, analytical model and law all agree.

        Byte counts are integers computed from the same integer quantities, so
        exact equality is the right test; a tolerance here would hide precisely
        the off-by-one-layer error worth catching.
        """
        return (
            not self.leaked_layers
            and self.measured_bytes == self.predicted_bytes
            and abs(self.measured_saving - self.predicted_saving) < 1e-12
        )

    def report(self) -> str:
        """Renders the audit as a line for a log.

        Returns:
            A single line, prefixed ``ok`` or ``LEAK``.
        """
        status = "ok  " if self.exact else "LEAK"
        line = (
            f"{status} depth {self.depth:>3}/{self.n_layers}  "
            f"T={self.seq_len:<6} measured {self.measured_bytes:>12,}B  "
            f"predicted {self.predicted_bytes:>12,}B  "
            f"saving {self.measured_saving:.4f} (law {self.predicted_saving:.4f})"
        )
        if self.leaked_layers:
            line += f"  layers above the cap holding entries: {self.leaked_layers}"
        elif self.measured_bytes != self.predicted_bytes:
            line += "  measurement disagrees with the analytical model"
        return line


def audit_kv_cache(
    cache,
    cost_model: AnalyticalCostModel,
    n_layers: int,
    dtype: str = "fp32",
) -> KVAudit:
    """Measures a depth-capped cache against the proportional memory claim.

    Args:
        cache: A :class:`src.modules.KVCache`, or anything exposing
            ``active_depth``, ``seq_len``, ``bytes_allocated`` and
            ``layer_presence``.
        cost_model: Model supplying the analytical prediction.
        n_layers: Full depth of the network.
        dtype: Cache precision, a key of :data:`DTYPE_BYTES`.

    Returns:
        The audit. Read :attr:`KVAudit.exact` for the verdict.
    """
    depth = cache.active_depth
    seq_len = cache.seq_len
    return KVAudit(
        depth=depth,
        n_layers=n_layers,
        seq_len=seq_len,
        measured_bytes=cache.bytes_allocated,
        predicted_bytes=cost_model.kv_bytes(depth, seq_len, dtype),
        full_depth_bytes=cost_model.kv_bytes(n_layers, seq_len, dtype),
        populated_layers=tuple(
            index for index, present in enumerate(cache.layer_presence) if present
        ),
    )


@dataclass
class CostCounters:
    """What an execution actually did, counted as it happened.

    Analytical formulas describe an intended execution; these record the real
    one. Comparing the two is how an accounting bug is caught, and
    :meth:`estimated_macs` deliberately derives its total from the counters
    rather than from the plan.

    Attributes:
        block_executions: Times each layer ran a full block, indexed by layer.
        kv_projection_tokens: Token-layer pairs that paid only for key and
            value projections, as token-level propagation does.
        adapter_tokens: Token-layer pairs that ran a propagation adapter.
        head_calls: Vocabulary projections invoked.
        head_tokens: Token positions projected to the vocabulary.
        controller_calls: Controller evaluations.
        backfill_tokens: Prompt positions replayed through upper blocks after
            an escalation.
        backfill_blocks: Blocks those positions were replayed through.
        prefill_tokens: Prompt positions processed.
        decode_tokens: Tokens generated.
        attention_position_sum: Total positions attended to, summed over every
            token-block pair. Held as one accumulator rather than a list of
            context lengths, which would grow as tokens times depth.
    """

    block_executions: dict[int, int] = field(default_factory=dict)
    kv_projection_tokens: int = 0
    adapter_tokens: int = 0
    head_calls: int = 0
    head_tokens: int = 0
    controller_calls: int = 0
    backfill_tokens: int = 0
    backfill_blocks: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    attention_position_sum: int = 0

    def record_blocks(
        self,
        depth: int,
        tokens: int = 1,
        context_len: int | None = None,
        start_depth: int = 0,
    ) -> None:
        """Records tokens passing through a range of blocks.

        Args:
            depth: One past the last block executed.
            tokens: Token positions that went through them.
            context_len: Positions each of those tokens attended to. ``None``
                skips the attention accounting, which is right only when the
                caller adds it another way.
            start_depth: First block executed, so a suffix continuation records
                only the blocks it really ran.
        """
        for layer in range(start_depth, depth):
            self.block_executions[layer] = (
                self.block_executions.get(layer, 0) + tokens
            )
        if context_len is not None:
            self.attention_position_sum += (
                (depth - start_depth) * tokens * context_len
            )

    def record_prefill(
        self,
        depth: int,
        prompt_len: int,
        rows: int = 1,
        start_depth: int = 0,
        offset: int = 0,
    ) -> None:
        """Records a whole prompt passing through a range of blocks.

        Prefill attention is not a single context length: position ``t``
        attends over ``offset + t + 1`` positions, so the sum is quadratic in
        the prompt and would be badly understated by using the final length
        alone.

        Args:
            depth: One past the last block executed.
            prompt_len: Prompt positions processed.
            rows: Requests processed.
            start_depth: First block executed.
            offset: Positions already in the cache before this prompt.
        """
        blocks = depth - start_depth
        for layer in range(start_depth, depth):
            self.block_executions[layer] = (
                self.block_executions.get(layer, 0) + prompt_len * rows
            )
        positions = sum(offset + t + 1 for t in range(prompt_len))
        self.attention_position_sum += blocks * rows * positions
        self.prefill_tokens += prompt_len * rows

    @property
    def total_block_executions(self) -> int:
        """Total token-block pairs executed."""
        return sum(self.block_executions.values())

    def merge(self, other: CostCounters) -> None:
        """Adds another counter set into this one, in place.

        Args:
            other: Counters to absorb, typically from one routing bucket.
        """
        for layer, count in other.block_executions.items():
            self.block_executions[layer] = (
                self.block_executions.get(layer, 0) + count
            )
        self.kv_projection_tokens += other.kv_projection_tokens
        self.adapter_tokens += other.adapter_tokens
        self.head_calls += other.head_calls
        self.head_tokens += other.head_tokens
        self.controller_calls += other.controller_calls
        self.backfill_tokens += other.backfill_tokens
        self.backfill_blocks += other.backfill_blocks
        self.prefill_tokens += other.prefill_tokens
        self.decode_tokens += other.decode_tokens
        self.attention_position_sum += other.attention_position_sum

    def estimated_macs(
        self,
        model: AnalyticalCostModel,
        adapter_rank: int = 0,
        controller_feature_dim: int = 0,
        controller_hidden: int = 0,
        n_tiers: int = 0,
    ) -> dict[str, float]:
        """Turns the counters into an estimated multiply-accumulate budget.

        The attention term uses the recorded context lengths where they exist,
        so a long generation is not costed as though every step attended to one
        token.

        Args:
            model: Architecture cost model.
            adapter_rank: Bottleneck width of the propagation adapters.
            controller_feature_dim: Pooled feature width.
            controller_hidden: Controller bottleneck width.
            n_tiers: Candidate depths the controller scored.

        Returns:
            A breakdown keyed by component, plus ``"total"``. Every entry is an
            estimate; none is a measurement of time.
        """
        per_token = model.projection_macs + model.ffn_macs
        blocks = self.total_block_executions * per_token

        # Context attention, summed over the positions really attended to.
        attention = 2.0 * self.attention_position_sum * model.d_model

        breakdown = {
            "blocks": blocks,
            "attention": attention,
            "kv_projections": self.kv_projection_tokens * model.kv_projection_macs,
            "adapters": self.adapter_tokens * model.adapter_macs(adapter_rank),
            "vocabulary_head": self.head_tokens * model.head_macs,
            "controller": self.controller_calls
            * model.controller_macs(
                controller_feature_dim, controller_hidden, n_tiers
            ),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown


def format_cost_table(breakdown: dict[str, float]) -> str:
    """Renders a cost breakdown with each component's share.

    Args:
        breakdown: Output of :meth:`CostCounters.estimated_macs`.

    Returns:
        A multi-line table, most expensive component first.
    """
    total = max(breakdown.get("total", 0.0), 1e-9)
    rows = sorted(
        ((k, v) for k, v in breakdown.items() if k != "total"),
        key=lambda item: -item[1],
    )

    header = f"{'component':>18}  {'est. MACs':>14}  {'share':>7}"
    lines = [header, "-" * len(header)]
    for name, value in rows:
        lines.append(f"{name:>18}  {value:>14,.0f}  {value / total:>6.1%}")
    lines.append(f"{'total':>18}  {total:>14,.0f}  {1.0:>6.1%}")
    return "\n".join(lines)


if __name__ == "__main__":
    for name, config in (
        (
            "repository toy",
            TransformerConfig(
                vocab_size=256, d_model=96, n_layers=8, n_heads=4, n_kv_heads=2,
                ff_dim=256, max_seq_len=160,
            ),
        ),
        ("repository default", TransformerConfig()),
    ):
        model = AnalyticalCostModel.from_config(config)
        print(f"{name}: d={model.d_model} f={model.ff_dim} "
              f"rho={model.gqa_ratio:.2f} V={model.vocab_size}")
        print(f"  one block (no context) {model.projection_macs + model.ffn_macs:,.0f}")
        print(f"  one vocabulary head    {model.head_macs:,.0f} "
              f"({model.head_to_block_ratio:.2f} blocks)")
        print(f"  kv bytes, depth {config.n_layers} x 512 tokens, bf16: "
              f"{model.kv_bytes(config.n_layers, 512, 'bf16'):,}")
        print()
