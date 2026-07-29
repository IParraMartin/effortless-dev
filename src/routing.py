"""Choosing how deep to run, once per request.

The decision this module makes is not "is the current answer confident?" but
"is more depth *worth buying*?". Those come apart, and the difference is the
reason a confidence threshold leaves headroom on the table. Two prompts can
produce identical output entropy at a checkpoint while deeper computation fixes
one and changes nothing for the other; a policy that sees only entropy must
treat them alike, so it cannot be optimal. The quantity to estimate is the
expected gain from continuing, compared against what continuing costs:

    continue at checkpoint k  <=>  E[V_{k+1} - R_k | H_k] > lambda * dC_k

:class:`DepthController` estimates a practical form of that. It reads a pooled
summary of the probe layer's hidden states and never touches the vocabulary
distribution — which is an arithmetic decision, not an aesthetic one. One full
vocabulary projection costs ``d_model * vocab_size`` multiply-accumulates,
which for a realistic vocabulary is several times an entire block, so a
controller that consulted the softmax at every candidate depth could easily
cost more than the depth it saved. The controller here costs
``d_model * hidden + hidden * tiers``, three orders of magnitude less.

Two output heads are available, and they answer different questions:

* **utility** predicts quality at each candidate depth. The cost trade-off is
  applied afterwards, at selection time, so one trained controller serves any
  budget and ``routing_lambda`` can change per request without retraining.
* **ordinal** predicts the probability that each depth is already sufficient,
  parameterized so the probabilities increase with depth by construction. The
  events are nested, so this structure is true rather than merely convenient,
  and it calibrates more easily — but it is tied to one operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RoutingConfig, TransformerConfig


@dataclass
class RoutingTrace:
    """A record of what routing decided, and what it cost.

    Everything is per request and in the caller's original row order, so a
    trace can be joined against the requests that produced it. Estimated and
    measured quantities are kept in separate fields throughout: block counts
    and cache bytes are counted, cost in multiply-accumulates is derived from a
    formula, and neither is a latency.

    Attributes:
        depths: Depth each request was routed to, in executed blocks.
        scores: Controller output per request per tier — predicted quality for
            the utility head, predicted sufficiency probability for the ordinal
            head.
        tiers: Candidate depths the scores are indexed by.
        probe_depth: Blocks every request ran before the decision.
        probe_blocks: Token-block pairs spent on the probe.
        endpoint_blocks: Token-block pairs spent at the chosen endpoints,
            excluding the probe.
        head_calls: Vocabulary projections performed.
        head_tokens: Token positions projected to the vocabulary.
        controller_calls: Controller evaluations.
        kv_bytes: Cache bytes materialized per request.
        boundary_bytes: Bytes of retained boundary activation per request.
        backfill_tokens: Prompt positions replayed through upper blocks.
        backfill_blocks: Blocks those positions were replayed through.
        escalations: Requests whose depth was raised after the initial choice.
        fallback_reasons: Why a request did not use the controller's choice,
            or ``None`` when it did.
    """

    depths: list[int] = field(default_factory=list)
    scores: list[list[float]] = field(default_factory=list)
    tiers: tuple[int, ...] = ()
    probe_depth: int = 0
    probe_blocks: int = 0
    endpoint_blocks: int = 0
    head_calls: int = 0
    head_tokens: int = 0
    controller_calls: int = 0
    kv_bytes: list[int] = field(default_factory=list)
    boundary_bytes: list[int] = field(default_factory=list)
    backfill_tokens: int = 0
    backfill_blocks: int = 0
    escalations: list[int] = field(default_factory=list)
    fallback_reasons: list[str | None] = field(default_factory=list)

    @property
    def mean_depth(self) -> float:
        """Average routed depth across requests."""
        return sum(self.depths) / max(len(self.depths), 1)

    @property
    def depth_distribution(self) -> dict[int, int]:
        """How many requests landed on each depth."""
        counts: dict[int, int] = {}
        for depth in self.depths:
            counts[depth] = counts.get(depth, 0) + 1
        return counts

    def summary(self) -> str:
        """Renders the route distribution and block accounting.

        Returns:
            A short multi-line string.
        """
        distribution = " ".join(
            f"d{depth}:{count}"
            for depth, count in sorted(self.depth_distribution.items())
        )
        return "\n".join(
            [
                f"routed {len(self.depths)} request(s), mean depth "
                f"{self.mean_depth:.2f}",
                f"  distribution   {distribution or 'none'}",
                f"  probe blocks   {self.probe_blocks:,} "
                f"(depth {self.probe_depth})",
                f"  endpoint blocks{self.endpoint_blocks:>8,}",
                f"  head calls     {self.head_calls:,} "
                f"({self.head_tokens:,} token positions)",
                f"  kv bytes       {sum(self.kv_bytes):,}",
            ]
        )


def pool_prompt_features(
    hidden: torch.Tensor,
    lengths: torch.Tensor | None = None,
    pooling: str = "last_mean",
    include_length: bool = False,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Reduces a probe layer's states to one feature vector per request.

    Padding is handled explicitly rather than ignored. With right padding and
    causal attention the *last real* token's state is exact whatever follows
    it, but a mean over the padded length is not: it averages in states that
    attended to nothing meaningful and shrinks by a factor that depends on how
    much padding a row happened to receive. Both reductions therefore read
    ``lengths``.

    Args:
        hidden: Probe hidden states shaped ``(batch, seq_len, d_model)``.
        lengths: Real prompt length per row, shaped ``(batch,)``. ``None``
            treats every row as full length.
        pooling: ``"last"``, ``"mean"``, or ``"last_mean"``.
        include_length: Whether to append normalized prompt length. Length
            correlates with difficulty often enough to be worth a feature, and
            it is free.
        max_seq_len: Divisor for the length feature. Required when
            ``include_length`` is set.

    Returns:
        Features shaped ``(batch, feature_dim)``.

    Raises:
        ValueError: If ``pooling`` is unknown, if a length is not positive or
            exceeds the sequence, or if the length feature is requested without
            ``max_seq_len``.
    """
    batch, seq_len, _ = hidden.shape
    if lengths is None:
        lengths = torch.full(
            (batch,), seq_len, dtype=torch.long, device=hidden.device
        )
    else:
        lengths = lengths.to(hidden.device, dtype=torch.long)
        if int(lengths.min()) < 1 or int(lengths.max()) > seq_len:
            raise ValueError(
                f"lengths must lie in [1, {seq_len}], got range "
                f"[{int(lengths.min())}, {int(lengths.max())}]."
            )

    positions = torch.arange(seq_len, device=hidden.device)
    mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    masked = hidden * mask

    last = hidden.gather(
        1,
        (lengths - 1).view(batch, 1, 1).expand(batch, 1, hidden.size(-1)),
    ).squeeze(1)
    mean = masked.sum(dim=1) / lengths.view(batch, 1).to(hidden.dtype)

    if pooling == "last":
        features = last
    elif pooling == "mean":
        features = mean
    elif pooling == "last_mean":
        features = torch.cat((last, mean), dim=-1)
    else:
        raise ValueError(
            f"pooling must be one of ('last', 'mean', 'last_mean'), got "
            f"{pooling!r}."
        )

    if include_length:
        if max_seq_len is None:
            raise ValueError(
                "max_seq_len is required when include_length is set, so the "
                "feature is comparable across models."
            )
        normalized = (lengths.to(features.dtype) / max_seq_len).unsqueeze(-1)
        features = torch.cat((features, normalized), dim=-1)

    return features


def feature_dim(
    d_model: int,
    pooling: str = "last_mean",
    include_length: bool = False,
) -> int:
    """Width of the vector :func:`pool_prompt_features` produces.

    Args:
        d_model: Residual width.
        pooling: Pooling scheme.
        include_length: Whether the length feature is appended.

    Returns:
        The feature dimension.
    """
    width = d_model * (2 if pooling == "last_mean" else 1)
    return width + (1 if include_length else 0)


class DepthController(nn.Module):
    """Predicts what more depth is worth, from a shallow probe.

    Deliberately tiny and deliberately inspectable. It is a two-layer network
    over pooled hidden states, which is enough to test whether the probe
    carries a usable signal at all — the question that decides whether
    request-level routing can work. A larger controller would confound "the
    features are insufficient" with "the model was too small", and the first is
    what needs answering first.

    Args:
        d_model: Residual width of the backbone. Used only to derive the
            feature width when ``input_dim`` is not given.
        n_tiers: Number of candidate depths.
        hidden_dim: Bottleneck width.
        pooling: How prompt positions are reduced.
        include_length: Whether normalized prompt length is a feature.
        output: ``"utility"`` or ``"ordinal"``.
        input_dim: Feature width, when it is already known — restoring a saved
            controller, or fitting on collected features whose width came from
            a pooling scheme this constructor was not told about. Given
            explicitly it is authoritative, so a mismatch between collection
            and training surfaces as a shape error rather than as silence.

    Raises:
        ValueError: If ``output`` is unrecognized or ``n_tiers`` is below one.
    """

    def __init__(
        self,
        d_model: int,
        n_tiers: int,
        hidden_dim: int = 64,
        pooling: str = "last_mean",
        include_length: bool = True,
        output: str = "utility",
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_tiers < 1:
            raise ValueError(f"n_tiers must be positive, got {n_tiers}.")
        if output not in ("utility", "ordinal"):
            raise ValueError(
                f"output must be 'utility' or 'ordinal', got {output!r}."
            )

        self.n_tiers = n_tiers
        self.pooling = pooling
        self.include_length = include_length
        self.output = output
        self.hidden_dim = hidden_dim
        self.feature_dim = input_dim or feature_dim(
            d_model, pooling, include_length
        )

        # The cost metric this controller was *selected* under. Offline
        # selection subtracts a stored per-request cost vector; the live path is
        # handed one by the caller. If the two are different quantities the
        # controller optimizes an objective nobody validated, and the mismatch
        # leaves no trace in any output. Set by the loader from the checkpoint,
        # and enforced by :meth:`select`.
        self.cost_metric: str | None = None

        # Probe features are raw residual-stream statistics whose scale depends
        # on the probe layer and the model, so they are normalized before the
        # trunk rather than left for the trunk to absorb.
        self.norm = nn.LayerNorm(self.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
        )

        if output == "utility":
            self.head = nn.Linear(hidden_dim, n_tiers)
        else:
            # Ordinal head. A single score orders the request on a difficulty
            # axis, and learned cutpoints say where each tier sits on it.
            # Storing increments and accumulating them through softplus makes
            # the cutpoints monotone by construction, so the predicted
            # probabilities can never invert -- which they must not, because
            # "sufficient by depth d" is a nested family of events.
            self.head = nn.Linear(hidden_dim, 1)
            self.cut_base = nn.Parameter(torch.zeros(1))
            self.cut_increments = nn.Parameter(torch.zeros(max(n_tiers - 1, 1)))

    def num_parameters(self) -> int:
        """Counts the controller's parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimated_macs(self) -> int:
        """Approximate multiply-accumulates for one request.

        Returns:
            Trunk plus head cost, ignoring normalization and activations. Worth
            comparing against ``d_model * vocab_size``, which is what reading
            the vocabulary at one checkpoint would cost.
        """
        trunk = self.feature_dim * self.hidden_dim
        head = self.hidden_dim * (self.n_tiers if self.output == "utility" else 1)
        return trunk + head

    def cutpoints(self) -> torch.Tensor:
        """Monotone thresholds of the ordinal head.

        Returns:
            Ascending cutpoints shaped ``(n_tiers,)``.

        Raises:
            ValueError: If the controller uses the utility head.
        """
        if self.output != "ordinal":
            raise ValueError("cutpoints exist only for the ordinal head.")
        steps = F.softplus(self.cut_increments[: self.n_tiers - 1])
        return torch.cat(
            (self.cut_base, self.cut_base + torch.cumsum(steps, dim=0))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Scores every candidate depth.

        Args:
            features: Pooled prompt features shaped ``(batch, feature_dim)``.

        Returns:
            For ``"utility"``, predicted quality per tier shaped
            ``(batch, n_tiers)``. For ``"ordinal"``, the cumulative
            probabilities ``P(D* <= tier_k)``, same shape, non-decreasing along
            the tier axis and with the deepest tier pinned to one, since full
            depth is always available as the fallback.
        """
        trunk = self.trunk(self.norm(features))

        if self.output == "utility":
            return self.head(trunk)

        difficulty = self.head(trunk)
        probabilities = torch.sigmoid(self.cutpoints().view(1, -1) - difficulty)
        # The deepest tier is sufficient by definition: there is nothing
        # further to compare it against. Pinning it removes a degree of
        # freedom that could otherwise train toward "no depth suffices".
        pinned = torch.ones_like(probabilities[:, -1:])
        return torch.cat((probabilities[:, :-1], pinned), dim=-1)

    @torch.no_grad()
    def select(
        self,
        features: torch.Tensor,
        tiers: tuple[int, ...],
        tier_costs: torch.Tensor | None = None,
        routing_lambda: float = 0.0,
        sufficiency_threshold: float = 0.5,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chooses one depth per request.

        The cost side of the trade is supplied, not predicted. Depth cost is
        deterministic given the architecture and the prompt, so predicting it
        would add variance for nothing, and keeping it external is what allows
        ``routing_lambda`` to change at inference on a frozen controller.

        Args:
            features: Pooled prompt features shaped ``(batch, feature_dim)``.
            tiers: Candidate depths, ascending.
            tier_costs: Normalized cost per tier shaped ``(n_tiers,)``.
                Defaults to depth divided by the deepest tier.
            routing_lambda: Price of a unit of normalized cost, in quality
                units.
            sufficiency_threshold: Probability at which the ordinal head calls
                a tier sufficient.
            deterministic: Whether to take the best tier rather than sample.
                Sampling exists for exploration during data collection; nothing
                reported should use it.
            generator: Random source for sampled routing.

        Returns:
            A tuple ``(depths, scores)`` where ``depths`` is shaped
            ``(batch,)`` in executed blocks and ``scores`` is the raw
            controller output shaped ``(batch, n_tiers)``.

        Raises:
            ValueError: If ``tiers`` does not match the controller's width.
        """
        if len(tiers) != self.n_tiers:
            raise ValueError(
                f"Controller was built for {self.n_tiers} tiers but was given "
                f"{len(tiers)}: {tiers}."
            )

        scores = self(features)
        tier_tensor = torch.tensor(tiers, device=features.device)

        if self.output == "utility":
            if tier_costs is None:
                if self.cost_metric not in (None, "cost_depth_fraction"):
                    raise ValueError(
                        f"this controller was selected under cost metric "
                        f"{self.cost_metric!r}, but select() was called without "
                        f"tier_costs and would fall back to depth/max_depth. "
                        f"Those are different objectives: a controller validated "
                        f"against measured MACs or cache bytes and deployed "
                        f"against a depth fraction is optimizing something "
                        f"nobody evaluated. Build the cost vector from the "
                        f"request's shape and pass it."
                    )
                tier_costs = tier_tensor.to(scores.dtype) / max(tiers)
            utility = scores - routing_lambda * tier_costs.view(1, -1).to(
                scores.device
            )
            if deterministic:
                index = utility.argmax(dim=-1)
            else:
                index = torch.multinomial(
                    F.softmax(utility, dim=-1), 1, generator=generator
                ).squeeze(-1)
        else:
            sufficient = scores >= sufficiency_threshold
            # The deepest tier is pinned to one, so something always fires and
            # argmax finds the shallowest sufficient depth without a special
            # case for "nothing was good enough".
            sufficient[:, -1] = True
            if deterministic:
                index = sufficient.float().argmax(dim=-1)
            else:
                index = torch.multinomial(
                    F.softmax(scores, dim=-1), 1, generator=generator
                ).squeeze(-1)

        return tier_tensor[index], scores


def build_controller(
    model_config: TransformerConfig,
    routing: RoutingConfig,
) -> DepthController:
    """Constructs the controller a routing configuration describes.

    Args:
        model_config: Backbone architecture.
        routing: Routing settings, already passed through
            :meth:`RoutingConfig.resolve`.

    Returns:
        An untrained controller sized for the selectable tiers.
    """
    return DepthController(
        d_model=model_config.d_model,
        n_tiers=len(routing.selectable_tiers),
        hidden_dim=routing.controller_hidden,
        pooling=routing.controller_pooling,
        include_length=routing.controller_use_length,
        output=routing.controller_output,
    )
