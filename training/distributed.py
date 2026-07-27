"""Process-group helpers for single- and multi-process training.

Every function here works whether or not the script was launched under
``torchrun``. A plain ``python train.py`` sees a world of size one and takes the
same code paths as a distributed run, so there is no separate single-GPU
branch to keep correct.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """Where this process sits in the job.

    Attributes:
        rank: Global index of this process.
        local_rank: Index of this process on its own node, which selects the
            GPU.
        world_size: Total number of processes.
        device: Device this process should place tensors on.
        enabled: Whether a process group was actually initialized.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    enabled: bool

    @property
    def is_main(self) -> bool:
        """Whether this process should log, checkpoint, and print."""
        return self.rank == 0


def _select_backend(requested: str) -> str:
    """Resolves the collective backend to use.

    Args:
        requested: Backend name, or ``"auto"`` to choose by device.

    Returns:
        ``"nccl"`` when CUDA is available, otherwise ``"gloo"``. Gloo is the
        only option on CPU and macOS, which is what makes distributed code
        testable without GPUs.
    """
    if requested != "auto":
        return requested
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup(backend: str = "auto") -> DistributedContext:
    """Joins the process group if one was requested by the launcher.

    Args:
        backend: Collective backend, or ``"auto"`` to pick by device.

    Returns:
        The :class:`DistributedContext` for this process. When ``WORLD_SIZE``
        is absent or one, no process group is created and a single-process
        context is returned.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return DistributedContext(0, 0, 1, device, enabled=False)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend=_select_backend(backend))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Gloo on CPU: every rank shares the machine, so they share the device.
        device = torch.device("cpu")

    return DistributedContext(rank, local_rank, world_size, device, enabled=True)


def cleanup(context: DistributedContext) -> None:
    """Tears down the process group if one was created.

    Args:
        context: The context returned by :func:`setup`.
    """
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Averages a tensor across all processes.

    Used for logging, so that the reported loss describes the whole global
    batch rather than whatever slice rank zero happened to see.

    Args:
        value: Tensor to average. Not modified.
        context: The context returned by :func:`setup`.

    Returns:
        The average across ranks, or ``value`` unchanged in a single-process
        run.
    """
    if not context.enabled:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / context.world_size


def barrier(context: DistributedContext) -> None:
    """Blocks until every process arrives.

    Args:
        context: The context returned by :func:`setup`.
    """
    if context.enabled and dist.is_initialized():
        dist.barrier()
