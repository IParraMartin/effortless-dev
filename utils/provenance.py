"""Recording enough about a run that someone else could believe it.

Every number in this repository is produced by a script, and a number without
its provenance is not evidence. This module collects the things that decide
whether a result can be reproduced or compared: the code that ran, the machine
it ran on, the seeds, the configuration, and a schema version so a reader can
tell an old record from a new one.

The design rule is that provenance capture must never fail a run. A missing
``git`` binary, a repository with no commits, or an unreadable device name all
degrade to an explicit ``None`` rather than an exception, because losing an
experiment to a metadata error is worse than recording that one field was
unavailable.

Typical use::

    record = RunRecord.create("evaluate_vertical_routing", config=asdict(cfg),
                              seeds={"eval": 0})
    record.write(Path("results/eval.json"), payload={"rows": rows})
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Version of the record layout written by :meth:`RunRecord.write`. Bump it
#: when a field changes meaning, not when one is added.
SCHEMA_VERSION = 1


def _git(*args: str) -> str | None:
    """Runs a git command, returning ``None`` when it cannot be answered.

    Args:
        *args: Arguments following ``git``.

    Returns:
        Stripped stdout, or ``None`` if git is missing, this is not a
        repository, or the command failed — for example in a fresh repository
        with no commits, where ``rev-parse HEAD`` has no answer to give.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Describes the working tree the run executed from.

    Returns:
        A mapping with ``commit``, ``branch``, and ``dirty``. A dirty tree
        means the commit hash does not fully identify the code that ran, which
        is exactly the situation a reader needs to be warned about. Fields that
        could not be determined are ``None``.
    """
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def hardware() -> dict[str, Any]:
    """Describes the machine and software stack.

    Latency numbers are meaningless without this, and quality numbers can still
    shift with a torch version, so it is recorded for every run rather than
    only for benchmarks.

    Returns:
        A mapping describing the platform, Python, torch, and accelerator.
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = "Apple MPS"
        device_type = "mps"
    else:
        device = platform.processor() or platform.machine()
        device_type = "cpu"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device_type": device_type,
        "device_name": device,
        "device_count": torch.cuda.device_count() if device_type == "cuda" else 1,
    }


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """Hashes a file so a checkpoint can be identified by content.

    A checkpoint path is not an identifier: files get overwritten, and two runs
    quoting ``final.pt`` may have used different weights. The digest is.

    Args:
        path: File to hash.
        chunk: Bytes read per iteration, so large checkpoints do not have to
            fit in memory.

    Returns:
        A hex SHA-256 digest, or ``None`` if the file does not exist.
    """
    import hashlib

    path = Path(path)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunRecord:
    """Everything about a run except its results.

    Attributes:
        schema_version: Layout version of the written record.
        script: Name of the experiment that produced it.
        created_at: Unix timestamp of creation.
        created_at_iso: The same instant, readable.
        git: Output of :func:`git_state`.
        hardware: Output of :func:`hardware`.
        seeds: Every seed the run consumed, named by what it controls. Recorded
            individually rather than as one number, because a run that seeds
            data and initialization separately cannot be reproduced from a
            single value.
        config: Full configuration, already reduced to JSON-safe types.
        inputs: Datasets, splits, and checkpoints the run consumed, including
            content digests where available.
        notes: Free-form annotations, such as a known limitation of the run.
    """

    schema_version: int
    script: str
    created_at: float
    created_at_iso: str
    git: dict[str, Any]
    hardware: dict[str, Any]
    seeds: dict[str, int] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        script: str,
        config: dict[str, Any] | None = None,
        seeds: dict[str, int] | None = None,
        inputs: dict[str, Any] | None = None,
        notes: list[str] | None = None,
    ) -> RunRecord:
        """Captures the current environment.

        Args:
            script: Name of the experiment, normally the module name.
            config: Configuration to record. Values are passed through
                :func:`jsonable`, so dataclasses and tensors are acceptable.
            seeds: Seeds keyed by what each one controls.
            inputs: Datasets, splits, and checkpoints consumed.
            notes: Caveats worth carrying alongside the numbers.

        Returns:
            A populated record, not yet written anywhere.
        """
        now = time.time()
        return cls(
            schema_version=SCHEMA_VERSION,
            script=script,
            created_at=now,
            created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            git=git_state(),
            hardware=hardware(),
            seeds=dict(seeds or {}),
            config=jsonable(config or {}),
            inputs=jsonable(inputs or {}),
            notes=list(notes or []),
        )

    def write(
        self,
        path: str | Path,
        payload: dict[str, Any] | None = None,
    ) -> Path:
        """Writes the record, and optionally the results, as one JSON file.

        Keeping provenance and results in the same file is deliberate: two
        files drift apart, and a results file that has lost its provenance is
        no more usable than one that never had any.

        Args:
            path: Destination. Parent directories are created.
            payload: Results to store under ``"results"``.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"run": asdict(self), "results": jsonable(payload or {})}
        path.write_text(json.dumps(document, indent=2, sort_keys=False))
        return path

    def summary(self) -> str:
        """Renders a one-line-per-fact header for console output.

        Returns:
            A multi-line string suitable for printing above a results table.
        """
        commit = self.git.get("commit") or "unknown"
        dirty = self.git.get("dirty")
        marker = "" if dirty is None else ("-dirty" if dirty else "")
        seeds = ", ".join(f"{k}={v}" for k, v in sorted(self.seeds.items()))
        return "\n".join(
            [
                f"script    {self.script}",
                f"commit    {commit[:12]}{marker}",
                f"hardware  {self.hardware['device_name']} "
                f"({self.hardware['device_type']}), torch "
                f"{self.hardware['torch']}",
                f"seeds     {seeds or 'none recorded'}",
                f"time      {self.created_at_iso}",
            ]
        )


def jsonable(value: Any) -> Any:
    """Converts a value into something :mod:`json` can serialize.

    Handles the types that actually turn up in these configurations —
    dataclasses, tensors, numpy scalars, paths, sets, tuples — and falls back
    to ``repr`` rather than raising, so provenance capture cannot be the thing
    that fails a run.

    Args:
        value: Any value.

    Returns:
        A structure of dicts, lists, strings, numbers, booleans, and ``None``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)
