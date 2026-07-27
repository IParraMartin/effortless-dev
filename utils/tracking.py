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
    from utils.calibration import SweepPoint
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

        self._wandb = wandb
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
                "n_exits": len(model_config.exit_layers),
                "exit_criterion": model_config.exit_criterion,
                "self_distill_weight": model_config.self_distill_weight,
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

    def log_exit_losses(
        self,
        exit_losses: dict[int, float],
        step: int,
        prefix: str = "train",
    ) -> None:
        """Records one series per exit.

        Separate series are what make the depth gradient visible: a healthy run
        shows deep exits below shallow ones, converging as self-distillation
        pulls the shallow exits up.

        Args:
            exit_losses: Cross-entropy keyed by layer index.
            step: Optimizer step.
            prefix: Namespace for the series, typically ``"train"`` or
                ``"eval"``.
        """
        if self._run is None:
            return
        self._run.log(
            {f"{prefix}/exit_ce/layer_{layer}": value
             for layer, value in exit_losses.items()},
            step=step,
        )

    def log_sweep(self, points: list[SweepPoint], step: int) -> None:
        """Records the accuracy-versus-depth tradeoff curve.

        Logged both as a table and as a scatter plot, so the shape of the
        tradeoff can be read directly rather than reconstructed from scalars.

        Args:
            points: Output of :func:`calibration.sweep_thresholds`.
            step: Optimizer step the sweep was taken at.
        """
        if self._run is None or not points:
            return

        table = self._wandb.Table(
            columns=["threshold", "mean_exit_depth", "accuracy", "nll",
                     "compute_saved"],
            data=[
                [p.threshold, p.mean_exit_depth, p.accuracy, p.nll,
                 p.compute_saved]
                for p in points
            ],
        )
        self._run.log(
            {
                "sweep/table": table,
                "sweep/curve": self._wandb.plot.scatter(
                    table, "compute_saved", "accuracy",
                    title="accuracy vs compute saved",
                ),
            },
            step=step,
        )

    def finish(self) -> None:
        """Closes the run, flushing anything still buffered."""
        if self._run is not None:
            self._run.finish()
            self._run = None
            self.enabled = False
