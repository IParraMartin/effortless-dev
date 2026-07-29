"""Configuration dataclasses for the early-exit Transformer.

Two configurations live here:

    * :class:`TransformerConfig` describes the architecture itself, including
      where the exit modules sit and how confidently they must predict before a
      token is allowed to stop.
    * :class:`TrainConfig` describes a training run: data, optimization
      schedule, precision, distribution, and checkpointing.

:func:`parse_into` turns either dataclass into a command line interface so a run
can be reconfigured without editing code.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING, dataclass, fields, replace
from typing import TypeVar

#: Confidence measures accepted by ``exit_criterion``.
EXIT_CRITERIA = ("entropy", "max_prob", "top2_margin")

#: Schemes accepted by ``exit_loss_weighting``.
EXIT_WEIGHTINGS = ("linear", "uniform", "final_only")

#: Schemes accepted by ``kv_exposure``.
KV_EXPOSURES = ("teacher", "simulated")

#: Precisions accepted by ``TrainConfig.dtype``.
DTYPES = ("bf16", "fp16", "fp32")

#: Routing granularities accepted by ``RoutingConfig.routing_mode``.
ROUTING_MODES = ("none", "fixed", "request")

#: Prompt pooling schemes accepted by ``RoutingConfig.controller_pooling``.
CONTROLLER_POOLINGS = ("last", "mean", "last_mean")

#: Controller head types accepted by ``RoutingConfig.controller_output``.
CONTROLLER_OUTPUTS = ("utility", "ordinal")


@dataclass
class TransformerConfig:
    """Hyperparameters describing a :class:`Transformer` instance.

    Attributes:
        vocab_size: Number of distinct tokens in the vocabulary.
        d_model: Width of the residual stream.
        n_layers: Number of stacked decoder blocks.
        n_heads: Number of query heads in each attention layer. Must divide
            ``d_model`` evenly.
        n_kv_heads: Number of key/value heads in each attention layer. Must
            divide ``n_heads`` evenly. Setting it equal to ``n_heads`` yields
            standard multi-head attention, ``1`` yields multi-query attention,
            and any divisor in between yields grouped-query attention.
        ff_dim: Inner dimension of the feed-forward network. When ``None`` it
            defaults to the LLaMA heuristic of ``8/3 * d_model`` rounded up to
            the nearest multiple of ``ff_multiple_of``.
        ff_multiple_of: Alignment used when ``ff_dim`` is inferred. Keeping the
            hidden size a multiple of a power of two is friendlier to tensor
            cores.
        max_seq_len: Longest sequence the rotary embedding tables are built
            for, and therefore the longest context the model can attend over.
        dropout: Dropout probability applied to attention weights and to the
            output of each sublayer. Modern LLM pretraining typically leaves
            this at zero.
        rope_theta: Base period of the rotary embedding. Larger values slow the
            frequency decay and help at long context lengths.
        norm_eps: Epsilon added to the RMSNorm denominator for numerical
            stability.
        tie_embeddings: Whether the output projection reuses the input
            embedding matrix.
        init_std: Standard deviation of the normal weight initialization.
        exit_every: Place an exit module every ``exit_every`` layers. The final
            layer always carries one, so ``1`` puts an exit on every layer.
        min_exit_layer: Earliest layer index allowed to terminate a token. The
            first layer or two rarely carry a usable prediction, and letting
            them fire mostly produces noise.
        exit_threshold: A token stops as soon as an exit's uncertainty falls
            below this value. Uncertainty is normalized to ``[0, 1]``, so ``0``
            disables early exiting entirely and ``1`` exits at the first
            opportunity.
        exit_criterion: Which uncertainty measure gates the exit. One of
            :data:`EXIT_CRITERIA`.
        exit_loss_weighting: How the per-exit losses are combined. ``"linear"``
            weights an exit proportionally to its depth, ``"uniform"`` weights
            every exit equally, and ``"final_only"`` trains just the last exit,
            which is useful as an ablation.
        self_distill_weight: Strength of the term pulling each shallow exit
            toward the final layer's distribution. Set to ``0`` to train the
            exits on the hard targets alone.
        self_distill_temperature: Softmax temperature used for the distillation
            term. Values above one expose more of the teacher's ranking among
            unlikely tokens.
        exits_per_step: Number of non-final exits to compute the loss for on
            each step, sampled fresh each time. ``None`` uses every exit.
            Lowering it trades gradient coverage for a large memory saving; see
            the note in :meth:`Transformer.forward`. Under DDP this also
            requires ``TrainConfig.find_unused_parameters``.
        full_depth_kv: When ``True``, generation runs every layer even for
            tokens that already exited, so cached keys and values are exact.
            This removes the speedup and exists to measure what the state
            propagation approximation costs.
        learned_kv_propagation: When ``True``, a small adapter per layer
            corrects an exited token's hidden state before that layer's
            key/value projections, instead of reusing it unchanged. The adapter
            is zero-initialized, so an untrained model reproduces plain
            propagation exactly.
        kv_adapter_rank: Bottleneck width of those adapters. The cost is two
            ``d_model x rank`` matrices per layer, negligible beside a
            vocabulary-sized output head.
        kv_propagation_weight: Weight of the auxiliary loss teaching the
            adapters to reconstruct the hidden state each layer would really
            have seen.
        kv_exposure: Which states the adapters are fitted on. ``"teacher"``
            uses the clean states of a full-depth pass, which is cheap but is
            not what inference produces. ``"simulated"`` runs an extra
            early-exit forward so the adapters see their own deployment
            distribution, at the cost of roughly one more backbone pass per
            step.
        refresh_every: During generation, force a full-depth token every this
            many steps, re-anchoring the residual stream and writing exact keys
            and values at that position. ``0`` disables it. Intended as a
            mitigation for drift accumulating over long generations.
    """

    vocab_size: int = 52_000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    ff_dim: int | None = None
    ff_multiple_of: int = 256
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    exit_every: int = 1
    min_exit_layer: int = 1
    exit_threshold: float = 0.3
    exit_criterion: str = "entropy"
    exit_loss_weighting: str = "linear"
    self_distill_weight: float = 0.5
    self_distill_temperature: float = 2.0
    exits_per_step: int | None = None
    full_depth_kv: bool = False
    learned_kv_propagation: bool = False
    kv_adapter_rank: int = 32
    kv_propagation_weight: float = 1.0
    kv_exposure: str = "teacher"
    refresh_every: int = 0

    def __post_init__(self) -> None:
        """Fills in derived fields and validates the configuration.

        Raises:
            ValueError: If any field is outside its permitted range, if
                ``d_model`` is not divisible by ``n_heads``, or if ``n_heads``
                is not divisible by ``n_kv_heads``.
        """
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )

        if self.ff_dim is None:
            hidden = int(8 * self.d_model / 3)
            self.ff_dim = self.ff_multiple_of * math.ceil(
                hidden / self.ff_multiple_of
            )

        if self.exit_every < 1:
            raise ValueError(f"exit_every must be positive, got {self.exit_every}.")
        if not 0 <= self.min_exit_layer < self.n_layers:
            raise ValueError(
                f"min_exit_layer must lie in [0, n_layers), got "
                f"{self.min_exit_layer} for n_layers={self.n_layers}."
            )
        if not 0.0 <= self.exit_threshold <= 1.0:
            raise ValueError(
                f"exit_threshold must lie in [0, 1], got {self.exit_threshold}."
            )
        if self.exit_criterion not in EXIT_CRITERIA:
            raise ValueError(
                f"exit_criterion must be one of {EXIT_CRITERIA}, got "
                f"{self.exit_criterion!r}."
            )
        if self.exit_loss_weighting not in EXIT_WEIGHTINGS:
            raise ValueError(
                f"exit_loss_weighting must be one of {EXIT_WEIGHTINGS}, got "
                f"{self.exit_loss_weighting!r}."
            )
        if self.self_distill_weight < 0.0:
            raise ValueError(
                "self_distill_weight must be non-negative, got "
                f"{self.self_distill_weight}."
            )
        if self.self_distill_temperature <= 0.0:
            raise ValueError(
                "self_distill_temperature must be positive, got "
                f"{self.self_distill_temperature}."
            )
        if self.kv_exposure not in KV_EXPOSURES:
            raise ValueError(
                f"kv_exposure must be one of {KV_EXPOSURES}, got "
                f"{self.kv_exposure!r}."
            )
        if self.exits_per_step is not None and self.exits_per_step < 1:
            raise ValueError(
                f"exits_per_step must be positive or None, got "
                f"{self.exits_per_step}."
            )

    @property
    def head_dim(self) -> int:
        """Dimensionality of a single attention head."""
        return self.d_model // self.n_heads

    @property
    def exit_layers(self) -> tuple[int, ...]:
        """Indices of the layers that carry an exit module.

        The final layer is always included, since it is the model's ordinary
        output head and every token that never gained confidence falls through
        to it.

        Returns:
            Ascending layer indices, always ending with ``n_layers - 1``.
        """
        layers = {
            layer
            for layer in range(self.n_layers)
            if (layer + 1) % self.exit_every == 0
        }
        layers.add(self.n_layers - 1)
        return tuple(sorted(layers))

    @property
    def exit_depths(self) -> tuple[int, ...]:
        """Executed depths that carry an exit module.

        The same exits as :attr:`exit_layers`, counted as blocks run rather
        than as zero-based indices. Request-level routing is expressed in
        depths throughout, because that is what compute and cache memory are
        proportional to, so this is the form its candidate tiers take.

        Returns:
            Ascending depths in ``[1, n_layers]``, always ending at
            ``n_layers``.
        """
        return tuple(layer + 1 for layer in self.exit_layers)

    @property
    def exit_weights(self) -> tuple[float, ...]:
        """Normalized weight of each exit in the training objective.

        Deeper exits are more accurate, so under the default ``"linear"``
        scheme they carry proportionally more of the loss. The weights sum to
        one so that the total loss stays comparable to a single-exit model's.

        Returns:
            One weight per entry of :attr:`exit_layers`, summing to ``1.0``.
        """
        layers = self.exit_layers
        if self.exit_loss_weighting == "uniform":
            raw = [1.0] * len(layers)
        elif self.exit_loss_weighting == "linear":
            raw = [float(layer + 1) for layer in layers]
        else:  # "final_only"
            raw = [0.0] * len(layers)
            raw[-1] = 1.0

        total = sum(raw)
        return tuple(value / total for value in raw)


@dataclass
class RoutingConfig:
    """How much of the backbone a request is allowed to execute.

    This describes *vertical routing*: one depth chosen per request, before
    generation starts, from a cheap probe of the prompt. It is a different
    mechanism from the per-token early exiting configured on
    :class:`TransformerConfig`, and the two are deliberately separable — token
    routing revisits the decision every step and keeps a full-depth cache,
    while request routing decides once and never materializes upper layers at
    all, which is where the memory saving comes from.

    The defaults leave routing off, so an unrouted model behaves exactly as it
    did before this existed.

    Attributes:
        routing_mode: ``"none"`` runs the full stack, ``"fixed"`` runs every
            request to :attr:`fixed_depth`, and ``"request"`` lets the
            controller choose per request.
        depth_tiers: Candidate depths, counted as executed blocks. ``None``
            derives them from the model's exit modules. Full depth must be
            reachable, because it is the fallback when routing fails.
        probe_depth: Blocks run before the routing decision. Their output is
            the controller's input, and their cost is paid by every request, so
            this trades decision quality against a floor on cost.
        fixed_depth: Depth used when ``routing_mode`` is ``"fixed"``.
        controller_hidden: Bottleneck width of the controller.
        controller_pooling: How prompt positions are reduced to one feature
            vector. One of :data:`CONTROLLER_POOLINGS`.
        controller_output: ``"utility"`` predicts quality at every tier and
            picks the best trade against cost, which lets ``routing_lambda``
            change at inference without retraining. ``"ordinal"`` predicts the
            probability that each tier is already sufficient, which is easier
            to calibrate but is tied to one operating point.
        controller_use_length: Whether normalized prompt length joins the
            pooled features.
        controller_path: Checkpoint to load controller weights from. Without
            one the controller is untrained, and routing is only as meaningful
            as its initialization.
        routing_lambda: Price of compute in the utility rule, in quality units
            per unit of normalized cost. Zero always chooses the deepest tier,
            which is the safe default.
        sufficiency_threshold: Probability at which the ordinal head calls a
            tier sufficient.
        min_depth: Shallowest depth the controller may choose. ``None`` uses
            the shallowest tier.
        safety_depth: Floor applied after the controller has chosen, for
            behaviour that shallow endpoints may simply not implement.
            Separate from ``min_depth`` so a safety floor is visible as such
            rather than hidden in the candidate set.
        deterministic_routing: Whether evaluation takes the argmax rather than
            sampling. Sampled routing exists for exploration during data
            collection, never for reporting.
        retain_boundary_state: Whether to keep the boundary activation for
            every prompt position so a request can later be escalated through
            upper blocks without recomputing the lower ones. Costs
            ``batch * prompt_len * d_model``.
    """

    routing_mode: str = "none"
    depth_tiers: tuple[int, ...] | None = None
    probe_depth: int = 1
    fixed_depth: int | None = None

    controller_hidden: int = 64
    controller_pooling: str = "last_mean"
    controller_output: str = "utility"
    controller_use_length: bool = True
    controller_path: str | None = None

    routing_lambda: float = 0.0
    sufficiency_threshold: float = 0.5
    min_depth: int | None = None
    safety_depth: int | None = None
    deterministic_routing: bool = True
    retain_boundary_state: bool = False

    def __post_init__(self) -> None:
        """Validates the fields that do not depend on a model.

        Raises:
            ValueError: If an enumerated field has an unrecognized value or a
                numeric field is out of range.
        """
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {ROUTING_MODES}, got "
                f"{self.routing_mode!r}."
            )
        if self.controller_pooling not in CONTROLLER_POOLINGS:
            raise ValueError(
                f"controller_pooling must be one of {CONTROLLER_POOLINGS}, got "
                f"{self.controller_pooling!r}."
            )
        if self.controller_output not in CONTROLLER_OUTPUTS:
            raise ValueError(
                f"controller_output must be one of {CONTROLLER_OUTPUTS}, got "
                f"{self.controller_output!r}."
            )
        if self.controller_hidden < 1:
            raise ValueError(
                f"controller_hidden must be positive, got {self.controller_hidden}."
            )
        if self.routing_lambda < 0.0:
            raise ValueError(
                f"routing_lambda must be non-negative, got {self.routing_lambda}."
            )
        if not 0.0 <= self.sufficiency_threshold <= 1.0:
            raise ValueError(
                "sufficiency_threshold must lie in [0, 1], got "
                f"{self.sufficiency_threshold}."
            )
        if self.depth_tiers is not None:
            self.depth_tiers = tuple(int(tier) for tier in self.depth_tiers)

    def resolve(self, model: TransformerConfig) -> RoutingConfig:
        """Fills in the model-dependent fields and checks the combination.

        Validation happens here rather than in ``__post_init__`` because
        almost every rule refers to the model: which depths carry an exit
        module, how deep the stack is, and whether the probe fits underneath
        the shallowest tier. A routing configuration is only well formed
        relative to a backbone.

        Args:
            model: The architecture routing will run on.

        Returns:
            A new configuration with :attr:`depth_tiers`, :attr:`min_depth`,
            and :attr:`fixed_depth` filled in. The original is unchanged.

        Raises:
            ValueError: If the tiers are unsorted, duplicated, out of range, or
                land on a depth with no exit module; if full depth is
                unreachable; if the probe is deeper than the shallowest
                selectable tier; or if a fixed-depth run names no depth.
        """
        available = model.exit_depths
        tiers = tuple(self.depth_tiers) if self.depth_tiers else available

        if not tiers:
            raise ValueError("depth_tiers resolved to an empty set.")

        if len(set(tiers)) != len(tiers):
            raise ValueError(f"depth_tiers must be unique, got {tiers}.")
        if list(tiers) != sorted(tiers):
            raise ValueError(
                f"depth_tiers must be ascending, got {tiers}. Order carries "
                f"meaning: the controller's outputs are indexed by tier."
            )

        out_of_range = [t for t in tiers if not 1 <= t <= model.n_layers]
        if out_of_range:
            raise ValueError(
                f"depth_tiers {out_of_range} lie outside [1, {model.n_layers}]. "
                f"Tiers count executed blocks, not layer indices, so full "
                f"depth is {model.n_layers}."
            )

        missing = [t for t in tiers if t not in available]
        if missing:
            raise ValueError(
                f"No exit module sits at depths {missing}. Exits are at "
                f"{available}; either choose tiers from those, or rebuild the "
                f"model with an exit_every that covers the depths you want."
            )

        if model.n_layers not in tiers:
            raise ValueError(
                f"Full depth ({model.n_layers}) must be among depth_tiers, "
                f"since it is the fallback when the controller declines to "
                f"route. Got {tiers}."
            )

        floor = tiers[0] if self.min_depth is None else int(self.min_depth)
        if floor not in tiers:
            raise ValueError(
                f"min_depth {floor} is not one of the tiers {tiers}."
            )

        selectable = tuple(t for t in tiers if t >= floor)
        if self.probe_depth < 1 or self.probe_depth > selectable[0]:
            raise ValueError(
                f"probe_depth must lie in [1, {selectable[0]}], got "
                f"{self.probe_depth}. The probe runs before the decision, so a "
                f"probe deeper than the shallowest selectable tier "
                f"({selectable[0]}) would have already spent more than that "
                f"tier costs."
            )

        if self.safety_depth is not None:
            if self.safety_depth not in tiers:
                raise ValueError(
                    f"safety_depth {self.safety_depth} is not one of the tiers "
                    f"{tiers}."
                )

        fixed = self.fixed_depth
        if self.routing_mode == "fixed":
            if fixed is None:
                fixed = model.n_layers
            if fixed not in tiers:
                raise ValueError(
                    f"fixed_depth {fixed} is not one of the tiers {tiers}."
                )

        return replace(
            self, depth_tiers=tiers, min_depth=floor, fixed_depth=fixed
        )

    @property
    def selectable_tiers(self) -> tuple[int, ...]:
        """Tiers the controller may actually choose, after the floor.

        Raises:
            ValueError: If called before :meth:`resolve`.
        """
        if self.depth_tiers is None:
            raise ValueError(
                "depth_tiers are unresolved; call resolve(model_config) first."
            )
        floor = self.min_depth or self.depth_tiers[0]
        return tuple(tier for tier in self.depth_tiers if tier >= floor)


@dataclass
class TrainConfig:
    """Settings for a training run.

    Attributes:
        dataset_name: Hugging Face dataset repository id.
        dataset_config: Configuration name within that dataset.
        text_column: Column holding the raw text. Conventions vary by corpus:
            ``"text"`` for most, ``"content"`` for code, ``"story"`` for
            TinyStories.
        streaming: Read the corpus as a stream instead of downloading it whole.
            Required for anything that will not fit on disk, and the reason
            preparation writes incrementally rather than buffering.
        tokenizer_name: Tokenizer to encode the corpus with. The model's
            vocabulary size is derived from it.
        data_dir: Directory holding the tokenized ``train.bin``/``val.bin``
            memory maps produced by ``data.py``.
        seq_len: Tokens per training example.
        max_train_docs: Cap on documents tokenized during preparation, useful
            for quick experiments. ``None`` uses the whole split. Also decides
            where a held-out set is carved from when the corpus ships only a
            training split: validation is taken from beyond the cap, so the
            two never overlap.
        overwrite_data: Re-tokenize splits that are already complete on disk.
            Preparation is skipped by default for any split whose sidecar
            matches the current settings, so a job that dies partway can be
            resubmitted without repeating hours of work.
        batch_size: Examples per micro-batch, per device.
        grad_accum_steps: Micro-batches accumulated before each optimizer step.
            The global batch is ``batch_size * grad_accum_steps * world_size``.
        max_steps: Total optimizer steps to run.
        learning_rate: Peak learning rate reached at the end of warmup.
        min_lr: Floor the cosine schedule decays to.
        warmup_steps: Steps spent linearly ramping up to ``learning_rate``.
        weight_decay: Decoupled weight decay, applied only to matrices.
        grad_clip: Global gradient-norm clip. ``0`` disables clipping.
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay. ``0.95`` is the usual LM choice, as
            ``0.999`` adapts too slowly for the loss spikes seen early on.
        dtype: Autocast precision, one of :data:`DTYPES`.
        compile_model: Whether to wrap the model in :func:`torch.compile`.
        seed: Base seed. Every stream below is derived from it when left unset,
            so a run is still specified by one number, but the streams are
            separate so that changing one does not move the others.
        model_init_seed: Parameter initialization. Deliberately *not* offset by
            rank: two arms meant to branch from a common parent cannot do so if
            their constructors consumed different random streams.
        data_order_seed: Which blocks each rank reads, in which order. The rank
            offset lives in the sampler's position formula rather than in this
            seed, so ranks read disjoint data by construction.
        dropout_seed: Stochastic regularization. Offset by rank on purpose, so
            ranks do not apply identical masks.
        exit_sampling_seed: Which exits a capped step scores. Must be identical
            across ranks or gradients diverge silently.
        num_workers: Dataloader worker processes per rank.
        ddp_backend: Collective backend. ``"nccl"`` on CUDA, ``"gloo"``
            otherwise; ``"auto"`` picks based on device availability.
        find_unused_parameters: Required when ``exits_per_step`` is set, since
            unsampled exit modules receive no gradient on that step.
        out_dir: Directory checkpoints are written to.
        save_every: Steps between checkpoints.
        resume_from: Checkpoint path to restore model, optimizer, and step from.
        eval_every: Steps between validation passes.
        eval_steps: Validation batches per pass.
        sweep_every: Steps between exit-threshold sweeps. ``0`` disables them.
        log_every: Steps between training log lines.
        wandb_project: Weights & Biases project to log to. Leaving this unset
            disables tracking entirely, and the ``wandb`` package is then not
            needed at all.
        wandb_run_name: Display name for the run. Defaults to a generated one.
        wandb_mode: ``"online"``, ``"offline"`` to buffer to disk for later
            syncing, or ``"disabled"`` to keep the calls but drop the data.
    """

    # Namespaced repository ids are required: the bare "wikitext" alias no
    # longer resolves in current huggingface_hub versions.
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    text_column: str = "text"
    streaming: bool = False
    tokenizer_name: str = "gpt2"
    data_dir: str = "data"
    seq_len: int = 512
    max_train_docs: int | None = None
    overwrite_data: bool = False

    batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    dtype: str = "bf16"
    compile_model: bool = True
    seed: int = 1337
    model_init_seed: int | None = None
    data_order_seed: int | None = None
    dropout_seed: int | None = None
    exit_sampling_seed: int | None = None
    num_workers: int = 2

    ddp_backend: str = "auto"
    find_unused_parameters: bool = False

    out_dir: str = "checkpoints"
    save_every: int = 1000
    resume_from: str | None = None
    eval_every: int = 500
    eval_steps: int = 50
    sweep_every: int = 2000
    log_every: int = 10

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"

    def __post_init__(self) -> None:
        """Validates the run settings.

        Raises:
            ValueError: If a positive-only field is zero or negative, if
                ``dtype`` is unrecognized, or if ``min_lr`` exceeds
                ``learning_rate``.
        """
        positive = {
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "eval_steps": self.eval_steps,
            "log_every": self.log_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}.")
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) cannot exceed learning_rate "
                f"({self.learning_rate})."
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}."
            )
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be non-negative, got {self.grad_clip}.")

    def seeds(self):
        """Resolves the named random streams this run consumes.

        Returns:
            A :class:`utils.provenance.Seeds` instance. Streams left unset on
            the command line are derived from :attr:`seed`, so the common case
            stays a single number while any stream remains independently
            settable.
        """
        # Imported here rather than at module scope: config.py is imported by
        # everything, and provenance pulls in subprocess and importlib.metadata
        # for facts that only a run needs.
        from utils.provenance import Seeds

        return Seeds.resolve(
            self.seed,
            model_init=self.model_init_seed,
            data_order=self.data_order_seed,
            dropout=self.dropout_seed,
            exit_sampling=self.exit_sampling_seed,
        )


_T = TypeVar("_T")


def _add_field_argument(parser: argparse.ArgumentParser, name: str, kind: object,
                        default: object) -> None:
    """Registers one dataclass field as a command line flag.

    Args:
        parser: Parser to extend.
        name: Field name, used verbatim as ``--name``.
        kind: The field's declared type annotation.
        default: Value to use when the flag is absent.
    """
    flag = f"--{name}"
    annotation = str(kind)

    if kind is bool or annotation == "bool":
        # Accepting an explicit value keeps overrides symmetric, so a config
        # default of True can be turned off with --compile_model=false.
        parser.add_argument(
            flag,
            type=lambda value: value.lower() in ("1", "true", "yes"),
            default=default,
            metavar="BOOL",
        )
    elif "int" in annotation:
        parser.add_argument(
            flag,
            type=lambda value: None if value.lower() == "none" else int(value),
            default=default,
        )
    elif "float" in annotation:
        parser.add_argument(flag, type=float, default=default)
    else:
        parser.add_argument(
            flag,
            type=lambda value: None if value.lower() == "none" else value,
            default=default,
        )


def parse_into(cls: type[_T], argv: list[str] | None = None) -> _T:
    """Builds a dataclass instance from command line arguments.

    Every field becomes a ``--field_name`` flag whose default is the field's
    own default, so an unmodified invocation reproduces the dataclass exactly.
    Booleans take an explicit value (``--compile_model=false``) and the literal
    string ``none`` maps to ``None`` for optional fields.

    Args:
        cls: A dataclass type, typically :class:`TrainConfig`.
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        An instance of ``cls`` with parsed overrides applied. The dataclass's
        own ``__post_init__`` validation still runs.

    Example:
        >>> config = parse_into(TrainConfig, ["--max_steps=100", "--dtype=fp32"])
        >>> config.max_steps
        100
    """
    parser = argparse.ArgumentParser(description=cls.__doc__)
    for field in fields(cls):
        default = field.default if field.default is not MISSING else None
        _add_field_argument(parser, field.name, field.type, default)

    namespace = parser.parse_args(argv)
    return cls(**vars(namespace))


#: Architecture fields that training derives rather than accepts: the
#: vocabulary comes from the tokenizer and the context from ``seq_len``.
_DERIVED_MODEL_FIELDS = frozenset({"vocab_size", "max_seq_len"})


def parse_configs(
    argv: list[str] | None = None,
) -> tuple[TrainConfig, dict[str, object]]:
    """Parses run settings and architecture overrides from one command line.

    Both dataclasses contribute flags to a single parser, so a run can be sized
    and scheduled in one invocation::

        python train.py --n_layers=6 --d_model=384 --exit_every=2 --max_steps=5000

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A tuple ``(train_config, model_overrides)``. The second element is
        ready to splat into :func:`tokenizer.config_from_tokenizer`, which
        supplies the vocabulary size itself.
    """
    parser = argparse.ArgumentParser(description="Train an early-exit Transformer.")

    train_names, model_names = [], []
    for cls, names in ((TrainConfig, train_names), (TransformerConfig, model_names)):
        for field in fields(cls):
            if cls is TransformerConfig and field.name in _DERIVED_MODEL_FIELDS:
                continue
            default = field.default if field.default is not MISSING else None
            _add_field_argument(parser, field.name, field.type, default)
            names.append(field.name)

    parsed = vars(parser.parse_args(argv))
    train_config = TrainConfig(**{name: parsed[name] for name in train_names})
    overrides = {name: parsed[name] for name in model_names}
    return train_config, overrides
