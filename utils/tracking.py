"""Experiment tracking, kept optional and rank-aware.

Training calls into :class:`RunTracker` unconditionally. The tracker decides
whether anything actually happens, so the training loop carries no
``if wandb is not None`` branches and no rank checks around logging.

Nothing here is required to train: with ``wandb_project`` unset, or with the
``wandb`` package absent, every method is a no-op.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from src.config import TrainConfig, TransformerConfig


class RunTracker:
    """Mirrors training metrics to Weights & Biases when configured.

    Args:
        train_config: Run settings. Tracking is enabled only when
            ``wandb_project`` is set.
        model_config: Architecture, recorded alongside the run settings so a
            result can be traced back to the model that produced it.
        is_main: Whether this process should log. Only rank zero writes, since
            the metrics it reports are already reduced across ranks.

    Attributes:
        enabled: Whether logging calls will do anything.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        model_config: TransformerConfig,
        is_main: bool = True,
    ) -> None:
        self._run = None
        self.enabled = False

        if not is_main or not train_config.wandb_project:
            return

        try:
            import wandb
        except ImportError:
            print(
                "wandb_project is set but wandb is not installed; "
                "continuing without tracking. Install it with 'uv add wandb'."
            )
            return

        self._run = wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            mode=train_config.wandb_mode,
            config={
                "train": asdict(train_config),
                "model": asdict(model_config),
                # Promoted out of the nested dicts so they can be used directly
                # as axes when comparing runs in the UI.
                "n_layers": model_config.n_layers,
                "d_model": model_config.d_model,
            },
        )
        self.enabled = True

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Records a group of scalars.

        Args:
            metrics: Metric names to values. Names are used verbatim, so
                callers should namespace them (``"train/loss"``).
            step: Optimizer step the metrics describe.
        """
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Closes the run, flushing anything still buffered."""
        if self._run is not None:
            self._run.finish()
            self._run = None
            self.enabled = False
