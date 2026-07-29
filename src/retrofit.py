"""Adding shallow endpoints to a trained parent without degrading it.

The first two training arms attacked the question from the wrong end. They
trained multi-exit models from scratch and then asked whether the full endpoint
had suffered. It had, but under an objective that had down-weighted that endpoint
by a factor of three-and-a-half, so the answer said nothing about capacity
sharing.

Retrofit inverts the order. Start from one strong parent, add endpoints, and
constrain the parent's own output not to move. The constraint can be exact or
approximate, and which one it is depends entirely on what is left trainable:

* **Exact.** With every backbone weight frozen and only exit modules trainable,
  the full-depth path is a function of unchanged parameters, so its logits are
  bit-identical before and after arbitrary exit training. There is nothing to
  measure and nothing to trade off. :func:`assert_parent_preserved` checks it.

* **Approximate.** As soon as a backbone weight can move — selective
  unfreezing, LoRA, full fine-tuning — the parent's output can drift, and
  "no regret" becomes a claim requiring a preservation term during training, a
  guardrail while it runs, and a predeclared non-inferiority test afterwards.

The ladder below runs least invasive first, so the cost of each step of
capability is visible against the step before it.

Typical use::

    parent = load_parent("checkpoints/vr-noexits/final.pt")
    model, report = retrofit(parent, RetrofitConfig(mode="frozen_exit_adapter"))
    print(report.summary())
    assert_parent_preserved(model, parent, probe_ids)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from src.config import TransformerConfig

#: Adaptation modes, ordered least to most invasive.
#:
#: ``frozen_tied_head``
#:     Backbone frozen. Only each exit's normalization is trainable; the
#:     vocabulary projection stays tied to the input embedding. The cheapest
#:     possible retrofit and the honest lower bound on what intermediate states
#:     already carry.
#: ``frozen_untied_head``
#:     Backbone frozen, each exit gets its own vocabulary projection. Far more
#:     capacity, and at ``d_model=768`` with a 50k vocabulary each head costs
#:     about 38.6M parameters — several times a block. Report the parameter
#:     count, not only the quality.
#: ``frozen_exit_adapter``
#:     Backbone frozen, each exit gains a zero-initialized residual bottleneck
#:     before its readout. Nonlinear remapping at a fraction of an untied head's
#:     cost. The recommended main lightweight baseline.
#: ``selective_unfreeze``
#:     A named subset of blocks becomes trainable. The parent is no longer exact;
#:     preservation becomes a measured constraint.
#: ``lora``
#:     Low-rank updates on named projections, backbone weights frozen. The
#:     parent is recoverable exactly by disabling the adapters, which is what
#:     makes it a usable reference rather than a memory.
#: ``qlora``
#:     LoRA over a 4-bit quantized frozen parent. Only worth its confounds when
#:     frozen weight storage is the binding constraint, which at 124M it is not.
#: ``full_finetune``
#:     Everything trainable. An upper bound on shallow quality and the mode most
#:     likely to erase the no-regret property.
RETROFIT_MODES = (
    "frozen_tied_head",
    "frozen_untied_head",
    "frozen_exit_adapter",
    "selective_unfreeze",
    "lora",
    "qlora",
    "full_finetune",
)

#: Modes in which no backbone parameter is trainable, so the parent's full-depth
#: output is preserved exactly rather than approximately.
EXACT_MODES = ("frozen_tied_head", "frozen_untied_head", "frozen_exit_adapter")

#: Default LoRA targets: the attention projections and the feed-forward
#: matrices, named as they appear in :class:`src.model.DecoderBlock`.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    Computes ``base(x) + scale * B(A(x))`` with ``B`` zero-initialized, so the
    wrapped layer is exactly the original at construction. Disabling the update
    recovers the parent's function precisely, which is the property that lets a
    LoRA retrofit keep an exact reference instead of a saved copy.

    Args:
        base: The layer to wrap. Its weight is frozen in place.
        rank: Rank of the update.
        alpha: Scaling numerator; the update is multiplied by ``alpha / rank``,
            the usual convention that keeps the effective step size roughly
            constant as the rank changes.
        dropout: Dropout applied to the input of the update path only.

    Attributes:
        base: The frozen original layer.
        lora_a: Down-projection, randomly initialized.
        lora_b: Up-projection, zero-initialized.
        enabled: Whether the update contributes. Set ``False`` to evaluate the
            exact parent.

    Raises:
        ValueError: If ``rank`` is not positive.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}.")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = alpha / rank
        self.enabled = True
        self.reset_identity()

    def reset_identity(self) -> None:
        """Restores exact parent behaviour.

        Re-applied after any model-wide initialization traversal, which would
        otherwise fill ``lora_b`` and turn a documented identity into a random
        perturbation of the parent.
        """
        # Kaiming-uniform on A and zeros on B, following the original LoRA
        # paper: the product is zero either way, but a non-degenerate A means
        # the first gradient step has a direction to move in.
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the wrapped layer and, if enabled, its update.

        Args:
            x: Input activations.

        Returns:
            The layer's output, shaped as the wrapped layer would produce.
        """
        out = self.base(x)
        if not self.enabled:
            return out
        return out + self.scale * self.lora_b(self.lora_a(self.dropout(x)))


@dataclass
class RetrofitConfig:
    """How a parent is adapted.

    Attributes:
        mode: One of :data:`RETROFIT_MODES`.
        exit_adapter_rank: Bottleneck width for ``frozen_exit_adapter``.
        untie_exit_heads: Give each exit its own vocabulary projection. Implied
            by ``frozen_untied_head``.
        train_exit_norms: Whether each exit's normalization is trainable. True
            in every mode; listed so a run's record states it.
        unfreeze_blocks: Block indices to unfreeze under
            ``selective_unfreeze``. Empty means the mode has nothing to do,
            which is rejected rather than silently equal to a frozen run.
        unfreeze_norms_only: Under ``selective_unfreeze``, restrict trainability
            to the normalization parameters of the named blocks. A much smaller
            capacity increase at much lower risk to the parent.
        lora_rank: Rank of the low-rank updates.
        lora_alpha: LoRA scaling numerator.
        lora_dropout: Dropout on the update path.
        lora_targets: Substrings matched against parameter-module names to
            select which projections are wrapped.
        lora_layers: Block indices to wrap. Empty means every block. Restricting
            to blocks below the deepest exit avoids the endpoint coupling
            described in :func:`retrofit`.
        preservation_weight: Convenience mirror of the model-config field, so a
            retrofit and its objective are specified in one place.
        parent_checkpoint: Path recorded for provenance.
    """

    mode: str = "frozen_exit_adapter"
    exit_adapter_rank: int = 32
    untie_exit_heads: bool = False
    train_exit_norms: bool = True
    unfreeze_blocks: tuple[int, ...] = ()
    unfreeze_norms_only: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    lora_layers: tuple[int, ...] = ()
    preservation_weight: float = 0.0
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validates the retrofit request.

        Raises:
            ValueError: If the mode is unknown, a rank is not positive, or the
                mode's own required fields are missing. A mode that silently
                degenerates into a different mode makes the run record wrong,
                which is worse than a failed launch.
        """
        if self.mode not in RETROFIT_MODES:
            raise ValueError(
                f"mode must be one of {RETROFIT_MODES}, got {self.mode!r}."
            )
        if self.mode == "frozen_exit_adapter" and self.exit_adapter_rank < 1:
            raise ValueError(
                "mode='frozen_exit_adapter' needs exit_adapter_rank >= 1; with "
                "zero it is frozen_tied_head, and the record should say so."
            )
        if self.mode == "selective_unfreeze" and not self.unfreeze_blocks:
            raise ValueError(
                "mode='selective_unfreeze' needs unfreeze_blocks; with none it "
                "is a frozen retrofit, and the record should say so."
            )
        if self.mode in ("lora", "qlora"):
            if self.lora_rank < 1:
                raise ValueError(f"lora_rank must be positive, got {self.lora_rank}.")
            if not self.lora_targets:
                raise ValueError("lora_targets must name at least one projection.")
        if self.mode == "frozen_untied_head":
            self.untie_exit_heads = True
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                f"lora_dropout must lie in [0, 1), got {self.lora_dropout}."
            )

    @property
    def preserves_parent_exactly(self) -> bool:
        """Whether this mode leaves the full-depth path bit-identical."""
        return self.mode in EXACT_MODES

    @property
    def parent_is_recoverable(self) -> bool:
        """Whether the parent's function can be recovered exactly at any point.

        True for the frozen modes, where it never moved, and for LoRA, where
        :func:`set_lora_enabled` switches the updates off. That makes the parent a
        computable reference rather than a remembered one, which is what lets a
        preservation claim be checked instead of asserted.
        """
        return self.mode in EXACT_MODES or self.mode in ("lora", "qlora")


@dataclass
class RetrofitReport:
    """What a retrofit actually made trainable.

    A mode name is an intention. This is the audit: the parameters that will
    receive gradients, grouped so a reader can see whether the intention was
    carried out. Reported before training rather than inferred afterwards,
    because a frozen retrofit that accidentally left a block trainable produces
    a perfectly plausible loss curve and a false no-regret claim.

    Attributes:
        mode: The mode applied.
        exact: Whether the parent's full-depth output is preserved exactly.
        trainable: Number of trainable parameters.
        frozen: Number of frozen parameters.
        trainable_groups: Trainable count per group, keyed by group name.
        trainable_names: Every trainable parameter's name, sorted.
        lora_modules: Names of the layers wrapped with a low-rank update.
        notes: Caveats worth carrying with the numbers.
    """

    mode: str
    exact: bool
    trainable: int
    frozen: int
    trainable_groups: dict[str, int] = field(default_factory=dict)
    trainable_names: tuple[str, ...] = ()
    lora_modules: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def trainable_fraction(self) -> float:
        """Trainable share of all parameters."""
        total = self.trainable + self.frozen
        return self.trainable / total if total else 0.0

    def summary(self) -> str:
        """Renders the audit for console output.

        Returns:
            A multi-line string.
        """
        lines = [
            f"mode        {self.mode}",
            f"parent      {'exactly preserved' if self.exact else 'may drift'}",
            f"trainable   {self.trainable:,} of "
            f"{self.trainable + self.frozen:,} ({self.trainable_fraction:.2%})",
        ]
        for group, count in sorted(self.trainable_groups.items()):
            lines.append(f"  {group:<22}{count:>14,}")
        if self.lora_modules:
            lines.append(f"lora        {len(self.lora_modules)} layers wrapped")
        for note in self.notes:
            lines.append(f"note        {note}")
        return "\n".join(lines)


def load_parent(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, TransformerConfig]:
    """Loads a trained checkpoint as a frozen parent.

    Args:
        path: Checkpoint written by :func:`training.train.save_checkpoint`.
        device: Device to place the model on.

    Returns:
        A tuple ``(model, config)``. The model is in eval mode with gradients
        disabled, so it cannot be trained by accident — the failure that turns a
        preservation term into a slow drift of both models toward each other.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If it carries no model configuration, which means it was not
            written by this repository and its architecture cannot be inferred.
    """
    from src.model import Transformer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in state:
        raise KeyError(
            f"{path} carries no 'model_config'. Its architecture cannot be "
            f"inferred, so it cannot be loaded as a parent."
        )

    config = state["model_config"]
    parent = Transformer(config)
    parent.load_state_dict(state["model"])
    parent.to(device).eval().requires_grad_(False)
    return parent, config


def restore(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    """Loads a retrofitted checkpoint, rebuilding the modules it was saved with.

    Wrapping a projection with :class:`LoRALinear` renames its weight from
    ``blocks.0.attn.q_proj.weight`` to ``blocks.0.attn.q_proj.base.weight``. A
    plain :class:`src.model.Transformer` therefore cannot load a LoRA checkpoint
    at all — every wrapped projection reads as a missing key, and a loader that
    waved that through with ``strict=False`` would return a model whose attention
    was left at its random initialization while reporting success.

    So the retrofit configuration has to be replayed before the weights are
    loaded. It is stored in the checkpoint for exactly this reason.

    Args:
        path: Checkpoint written by ``experiments.retrofit_parent``.
        device: Device to place the model on.

    Returns:
        The model, in eval mode with gradients disabled.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
        KeyError: If it carries no architecture.
        ValueError: If any parameter is still missing after the modules are
            rebuilt, which means the checkpoint and its recorded retrofit
            disagree.
    """
    from src.model import Transformer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in blob:
        raise KeyError(f"{path} carries no 'model_config'.")

    model = Transformer(blob["model_config"])

    recorded = blob.get("retrofit")
    if recorded:
        settings = RetrofitConfig(
            **{
                name: (
                    tuple(value) if isinstance(value, list) else value
                )
                for name, value in recorded.items()
            }
        )
        if settings.mode in ("lora", "qlora"):
            apply_lora(model, settings)

    incompatible = model.load_state_dict(blob["model"], strict=False)
    if incompatible.missing_keys:
        raise ValueError(
            f"{path} is missing {len(incompatible.missing_keys)} parameter(s) "
            f"after replaying its recorded retrofit "
            f"({(recorded or {}).get('mode', 'none')}), starting with "
            f"{incompatible.missing_keys[:3]}. The checkpoint and its retrofit "
            f"record disagree; loading anyway would leave part of the model at "
            f"its random initialization."
        )

    model.to(device).eval().requires_grad_(False)
    return model


def retrofit(
    parent: nn.Module,
    config: RetrofitConfig,
    model_config: TransformerConfig | None = None,
) -> tuple[nn.Module, RetrofitReport]:
    """Builds an elastic model from a parent, and reports what it made trainable.

    The parent's weights are copied into a model whose architecture may differ
    from it only in exit placement and exit-module contents. Backbone shapes must
    match, since the whole point is that the backbone is the parent's.

    Args:
        parent: A trained model, normally from :func:`load_parent`.
        config: How to adapt it.
        model_config: Architecture for the retrofitted model. Defaults to the
            parent's with the retrofit's exit settings applied. Supply one to
            change exit placement, which is the common case: a final-only parent
            has one exit and the retrofit wants several.

    Returns:
        A tuple ``(model, report)``.

    Raises:
        ValueError: If the parent's backbone shapes do not match the requested
            architecture, or if the mode is ``qlora`` without its dependency.

    Note:
        **Endpoint coupling under LoRA.** A request that stops at depth ``d``
        executes only the adapters up to ``d``. If full-depth quality comes to
        depend on upper-layer adapters compensating for lower-layer changes, the
        endpoints stop being independent and a shallow endpoint's quality becomes
        a function of which upper adapters exist. Restrict ``lora_layers`` to
        blocks below the deepest exit, or measure adapter ablations by depth
        before believing an endpoint's number.
    """
    from src.model import Transformer

    parent_config: TransformerConfig = parent.config
    if model_config is None:
        model_config = _derive_model_config(parent_config, config)

    _check_backbone_compatible(parent_config, model_config)

    model = Transformer(model_config)
    missing = _copy_parent_weights(parent, model)

    notes: list[str] = []
    if missing:
        notes.append(
            f"{len(missing)} module(s) had no parent counterpart and kept their "
            f"initialization: {', '.join(sorted(missing)[:4])}"
            + ("..." if len(missing) > 4 else "")
        )

    lora_modules: tuple[str, ...] = ()
    if config.mode == "full_finetune":
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
        _unfreeze_exits(model, config)
        if config.mode == "selective_unfreeze":
            _unfreeze_blocks(model, config)
        if config.mode in ("lora", "qlora"):
            lora_modules = apply_lora(model, config)
        if config.mode == "qlora":
            notes.append(
                "qlora requires a quantized parent; this build applies LoRA over "
                "the unquantized weights, so it is a BF16 LoRA control and must "
                "not be reported as QLoRA"
            )

    if not config.preserves_parent_exactly and config.preservation_weight == 0.0:
        notes.append(
            "backbone weights are trainable and preservation_weight is zero, so "
            "nothing constrains the parent's output; the no-regret claim needs "
            "a predeclared non-inferiority test on held-out data"
        )

    return model, _audit(model, config, lora_modules, tuple(notes))


def apply_lora(model: nn.Module, config: RetrofitConfig) -> tuple[str, ...]:
    """Wraps the selected projections with low-rank updates, in place.

    Args:
        model: Model to modify.
        config: Retrofit settings supplying rank, alpha, dropout, targets and
            layer range.

    Returns:
        The names of the wrapped layers, in traversal order. Exactly the set the
        run should report, so "which modules" is a recorded fact rather than an
        inference from the target list.
    """
    layers = set(config.lora_layers)
    wrapped: list[str] = []

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if not any(target in child_name for target in config.lora_targets):
                continue
            if layers:
                block = _block_index(full)
                if block is None or block not in layers:
                    continue
            setattr(
                module,
                child_name,
                LoRALinear(
                    child,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                ),
            )
            wrapped.append(full)

    return tuple(wrapped)


def set_lora_enabled(model: nn.Module, enabled: bool) -> int:
    """Turns every low-rank update on or off.

    Disabling them recovers the parent's function exactly, which is what makes a
    LoRA retrofit auditable: the reference is computable rather than remembered.

    Args:
        model: Model holding :class:`LoRALinear` layers.
        enabled: Whether the updates contribute.

    Returns:
        How many layers were switched.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled
            count += 1
    return count


def assert_parent_preserved(
    model: nn.Module,
    parent: nn.Module,
    input_ids: torch.Tensor,
    tolerance: float = 0.0,
) -> float:
    """Checks that the retrofitted model's full-depth logits match the parent's.

    Under a frozen mode this must hold to the bit, and ``tolerance=0.0`` says so.
    Anything else means a backbone weight moved, or a module that was believed
    frozen was not.

    Args:
        model: The retrofitted model.
        parent: The frozen reference.
        input_ids: Probe tokens shaped ``(batch, seq_len)``.
        tolerance: Largest permitted absolute difference. Zero demands exactness.

    Returns:
        The largest observed absolute difference.

    Raises:
        AssertionError: If the difference exceeds ``tolerance``. Raised rather
            than returned, because a caller that ignored the number would report
            an unverified no-regret result.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            left = model(input_ids).logits
            right = parent(input_ids).logits
    finally:
        model.train(was_training)

    difference = float((left - right).abs().max())
    if difference > tolerance:
        raise AssertionError(
            f"full-depth logits moved by {difference:.3e}, above the permitted "
            f"{tolerance:.3e}. Under a frozen retrofit this cannot happen unless "
            f"a backbone parameter is trainable or a module believed frozen is "
            f"not; check the retrofit report's trainable_names."
        )
    return difference


def _derive_model_config(
    parent_config: TransformerConfig, config: RetrofitConfig
) -> TransformerConfig:
    """Builds the retrofitted architecture from the parent's.

    Args:
        parent_config: The parent's architecture.
        config: Retrofit settings.

    Returns:
        A copy of the parent's configuration with the retrofit's exit settings
        applied. Exit placement is left alone; a caller wanting more exits than
        the parent has passes ``model_config`` explicitly.
    """
    from dataclasses import replace

    return replace(
        parent_config,
        exit_adapter_rank=(
            config.exit_adapter_rank
            if config.mode == "frozen_exit_adapter"
            else parent_config.exit_adapter_rank
        ),
        tie_embeddings=(
            False if config.untie_exit_heads else parent_config.tie_embeddings
        ),
        preservation_teacher_checkpoint=config.parent_checkpoint,
        preservation_weight=config.preservation_weight,
    )


def _check_backbone_compatible(
    parent_config: TransformerConfig, model_config: TransformerConfig
) -> None:
    """Rejects an architecture whose backbone is not the parent's.

    Args:
        parent_config: The parent's architecture.
        model_config: The requested architecture.

    Raises:
        ValueError: If a shape-determining field differs.
    """
    shape_fields = (
        "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ff_dim",
        "max_seq_len",
    )
    differing = {
        name: (getattr(parent_config, name), getattr(model_config, name))
        for name in shape_fields
        if getattr(parent_config, name) != getattr(model_config, name)
    }
    if differing:
        raise ValueError(
            f"the retrofitted backbone must be the parent's, but these fields "
            f"differ (parent, requested): {differing}."
        )


def _copy_parent_weights(parent: nn.Module, model: nn.Module) -> set[str]:
    """Copies every parameter and buffer the two models share.

    Exit modules are matched by *depth*, not by index: a final-only parent's sole
    exit sits at position 0, while in a six-exit model the same depth is position
    5. Matching by index would silently install the parent's full-depth readout
    on the depth-2 exit.

    Args:
        parent: Source model.
        model: Destination model.

    Returns:
        Names of destination parameters that had no source counterpart and kept
        their own initialization.
    """
    source = dict(parent.state_dict())
    destination = model.state_dict()

    remapped: dict[str, torch.Tensor] = {}
    parent_exits = {
        layer: position
        for position, layer in enumerate(parent.config.exit_layers)
    }

    missing: set[str] = set()
    for name, tensor in destination.items():
        key = name
        match = re.match(r"exit_modules\.(\d+)\.(.*)", name)
        if match:
            position, remainder = int(match.group(1)), match.group(2)
            layer = model.config.exit_layers[position]
            if layer not in parent_exits:
                missing.add(f"exit_modules.{position}")
                continue
            key = f"exit_modules.{parent_exits[layer]}.{remainder}"
        if key not in source or source[key].shape != tensor.shape:
            missing.add(name)
            continue
        remapped[name] = source[key]

    model.load_state_dict(remapped, strict=False)
    return missing


def _unfreeze_exits(model: nn.Module, config: RetrofitConfig) -> None:
    """Makes the shallow exit modules trainable, and nothing else.

    Two exclusions, and both are load-bearing rather than tidiness:

    Under a tied head the projection *is* the input embedding, so it stays
    frozen. Training it would move the parent's full-depth logits through the
    embedding, which is the one thing a frozen retrofit promises not to do.

    The **final** exit module is the parent's own output head. Its normalization
    and, when untied, its projection both feed the full-depth logits, so
    training either breaks exactness after the very first optimizer step — while
    leaving the model bit-identical at initialization, which is where a naive
    check would look. There is nothing to retrofit at the deepest endpoint: it is
    the thing being preserved.

    Args:
        model: Model whose shallow exits become trainable.
        config: Retrofit settings.
    """
    tied = model.config.tie_embeddings
    protect_final = config.mode not in ("selective_unfreeze", "full_finetune")
    last = len(model.exit_modules) - 1

    for position, exit_module in enumerate(model.exit_modules):
        if protect_final and position == last:
            continue
        if config.train_exit_norms:
            exit_module.norm.requires_grad_(True)
        if exit_module.adapter is not None:
            exit_module.adapter.requires_grad_(True)
        if not tied:
            exit_module.proj.requires_grad_(True)


def _unfreeze_blocks(model: nn.Module, config: RetrofitConfig) -> None:
    """Makes the named blocks trainable.

    Args:
        model: Model to modify.
        config: Retrofit settings naming the blocks and whether to restrict to
            normalization parameters.

    Raises:
        ValueError: If a named block does not exist, which would otherwise leave
            a run silently frozen where it believed it was adapting.
    """
    out_of_range = [
        index
        for index in config.unfreeze_blocks
        if not 0 <= index < len(model.blocks)
    ]
    if out_of_range:
        raise ValueError(
            f"unfreeze_blocks names block(s) {out_of_range}, but the model has "
            f"{len(model.blocks)}."
        )

    for index in config.unfreeze_blocks:
        block = model.blocks[index]
        if config.unfreeze_norms_only:
            block.attn_norm.requires_grad_(True)
            block.ffn_norm.requires_grad_(True)
        else:
            block.requires_grad_(True)


def _audit(
    model: nn.Module,
    config: RetrofitConfig,
    lora_modules: tuple[str, ...],
    notes: tuple[str, ...],
) -> RetrofitReport:
    """Counts what is trainable, by group.

    Args:
        model: The retrofitted model.
        config: Retrofit settings.
        lora_modules: Layers wrapped with low-rank updates.
        notes: Caveats to carry.

    Returns:
        The populated report.
    """
    trainable, frozen = 0, 0
    groups: dict[str, int] = {}
    names: list[str] = []

    # Deduplicated by identity: a tied projection is the same tensor as the
    # embedding and must not be counted twice.
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.requires_grad:
            trainable += parameter.numel()
            groups[_audit_group(name)] = (
                groups.get(_audit_group(name), 0) + parameter.numel()
            )
            names.append(name)
        else:
            frozen += parameter.numel()

    return RetrofitReport(
        mode=config.mode,
        exact=config.preserves_parent_exactly,
        trainable=trainable,
        frozen=frozen,
        trainable_groups=groups,
        trainable_names=tuple(sorted(names)),
        lora_modules=lora_modules,
        notes=notes,
    )


def _audit_group(name: str) -> str:
    """Names the audit group a parameter belongs to.

    Args:
        name: Fully qualified parameter name.

    Returns:
        ``"exit_norms"``, ``"exit_adapters"``, ``"exit_heads"``, ``"lora"``,
        ``"blocks"``, ``"embedding"``, or ``"other"``. The exit groups are kept
        apart because their costs differ by orders of magnitude: an untied head
        is roughly 38.6M parameters at ``d_model=768`` and a rank-32 adapter is
        roughly 49k.
    """
    if "lora_" in name:
        return "lora"
    if name.startswith("exit_modules"):
        if ".adapter." in name:
            return "exit_adapters"
        if ".norm." in name:
            return "exit_norms"
        return "exit_heads"
    if name.startswith("blocks"):
        return "blocks"
    if name.startswith("embed"):
        return "embedding"
    return "other"


def _block_index(name: str) -> int | None:
    """Extracts the block index from a module name.

    Args:
        name: Fully qualified module name.

    Returns:
        The block index, or ``None`` if the module is not inside a block.
    """
    match = re.match(r"blocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def trainable_parameters(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Yields the parameters an optimizer should receive.

    Args:
        model: Model to inspect.

    Yields:
        ``(name, parameter)`` for every trainable parameter, deduplicated by
        identity so a tied weight is offered once.
    """
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        yield name, parameter
