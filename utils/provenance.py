"""Recording enough about a run that someone else could believe it.

Every number in this repository is produced by a script, and a number without
its provenance is not evidence. This module collects the things that decide
whether a result can be reproduced or compared: the code that ran, the machine
it ran on, the seeds, the configuration, and a schema version so a reader can
tell an old record from a new one.

There are two layers. :class:`RunRecord` is the single-file form used by the
one-shot experiment scripts, where provenance and results belong in the same
JSON document. :class:`RunArtifacts` is the directory form used by long
cluster runs, which need to append metrics as they go, hold raw per-request
records too large for one document, and survive preemption with an auditable
resume chain.

The design rule for capture is that it must never fail a run: a missing ``git``
binary, a repository with no commits, or an unreadable device name all degrade
to an explicit ``None`` rather than an exception, because losing an experiment
to a metadata error is worse than recording that one field was unavailable.

The design rule for *validation* is the opposite, and applies only to
:class:`RunArtifacts`: a run that cannot describe itself completely fails
before it consumes cluster time, because the alternative is a checkpoint nobody
can place. :meth:`RunArtifacts.create` therefore raises when a field named in
``required`` came back unavailable.

Typical use::

    record = RunRecord.create("evaluate_vertical_routing", config=asdict(cfg),
                              seeds={"eval": 0})
    record.write(Path("results/eval.json"), payload={"rows": rows})

    artifacts = RunArtifacts.create(
        "runs/vr-exits", script="training.train", config=asdict(cfg),
        seeds=Seeds.derive(1337), inputs={"train": "data/train.bin"},
    )
    artifacts.log_metric({"step": 0, "loss": 10.8})
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

#: Version of the record layout written by :meth:`RunRecord.write`. Bump it
#: when a field changes meaning, not when one is added.
SCHEMA_VERSION = 1

#: Version of the directory layout written by :class:`RunArtifacts`. Separate
#: from :data:`SCHEMA_VERSION` because the two formats evolve independently.
ARTIFACT_SCHEMA_VERSION = 1

#: Substrings that mark an environment variable as a credential. Matched
#: case-insensitively against the variable name, because the value of a secret
#: is exactly what must not reach a file that gets committed or shared.
SECRET_MARKERS = (
    "token",
    "key",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth",
    "session",
    "cookie",
    "signature",
)

#: Environment variables worth recording verbatim. An allowlist rather than a
#: denylist: dumping ``os.environ`` and filtering it is one forgotten marker
#: away from writing somebody's API key into a results directory.
RECORDED_ENVIRONMENT = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS",
    "SLURM_NTASKS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_RESTART_COUNT",
    "CUDA_VISIBLE_DEVICES",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "OMP_NUM_THREADS",
    "TORCH_COMPILE_DISABLE",
)


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


def git_diff() -> str | None:
    """Captures uncommitted changes, so a dirty tree is still identifiable.

    A commit hash plus this diff reconstructs the exact source that ran. Staged
    and unstaged changes are both included; untracked files are not, since git
    cannot diff what it has never seen — :func:`git_state` reports them through
    ``dirty`` instead.

    Returns:
        A unified diff, the empty string for a clean tree, or ``None`` when git
        could not answer.
    """
    return _git("diff", "HEAD")


def hardware() -> dict[str, Any]:
    """Describes the machine the run executed on.

    Latency numbers are meaningless without this, and quality numbers can still
    shift with a torch version, so it is recorded for every run rather than
    only for benchmarks.

    Returns:
        A mapping describing the platform, host, accelerators, and the
        scheduler allocation. Accelerator entries carry the name and total
        memory of every visible device, because a run that landed on a
        heterogeneous node cannot be compared against one that did not.
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

    accelerators = []
    if device_type == "cuda":
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerators.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": f"{properties.major}.{properties.minor}",
                    "multi_processor_count": properties.multi_processor_count,
                }
            )

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "hostname": platform.node() or None,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device_type": device_type,
        "device_name": device,
        "device_count": torch.cuda.device_count() if device_type == "cuda" else 1,
        "accelerators": accelerators,
        "scheduler": scheduler_allocation(),
    }


def scheduler_allocation() -> dict[str, Any]:
    """Reads the batch scheduler's description of this allocation.

    Returns:
        The subset of :data:`RECORDED_ENVIRONMENT` that is set, with values
        verbatim. Empty when the run was launched by hand, which is itself
        worth being able to tell.
    """
    return {
        name: os.environ[name] for name in RECORDED_ENVIRONMENT if name in os.environ
    }


def environment() -> dict[str, Any]:
    """Describes the software stack, including the installed package set.

    The package lock is the part that is usually missing and usually matters:
    two runs of the same commit on the same node can still disagree because one
    of them resolved a different minor version of a dependency.

    Returns:
        A mapping with interpreter, framework, accelerator-toolkit, and package
        version information. ``packages`` maps distribution name to version and
        is ``None`` if the metadata could not be read.
    """
    return {
        "python": sys.version,
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cudnn": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "nccl": _nccl_version(),
        "driver": _driver_version(),
        "packages": installed_packages(),
    }


def _nccl_version() -> str | None:
    """Reports the NCCL version torch was built against, if it has one."""
    try:
        version = torch.cuda.nccl.version()
    except Exception:
        return None
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version)
    return str(version)


def _driver_version() -> str | None:
    """Reports the NVIDIA driver version, or ``None`` off CUDA hardware."""
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[0].strip() or None


def installed_packages() -> dict[str, str] | None:
    """Lists installed distributions and their versions.

    Returns:
        A name-to-version mapping, sorted by name, or ``None`` if the
        distribution metadata is unreadable.
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib
        return None

    try:
        found = {}
        for distribution in metadata.distributions():
            name = distribution.metadata["Name"]
            if name:
                found[name] = distribution.version or "unknown"
    except Exception:
        return None
    return dict(sorted(found.items()))


def redacted(mapping: dict[str, Any]) -> dict[str, Any]:
    """Replaces values whose keys look like credentials.

    Args:
        mapping: Any flat mapping bound for a written record.

    Returns:
        A copy in which every key matching :data:`SECRET_MARKERS` has its value
        replaced by ``"<redacted>"``. The key survives, because knowing that a
        run had an API token set is provenance; knowing the token is a leak.
    """
    result = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            result[key] = "<redacted>"
        else:
            result[key] = value
    return result


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Writes a file so that no reader ever observes it half-written.

    Long runs are killed by schedulers mid-write. A truncated
    ``resolved_config.json`` is worse than a missing one, because it looks like
    a record until it is parsed.

    Args:
        path: Destination. Parent directories are created.
        text: Complete contents.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # os.replace is atomic within a filesystem, so the destination is
        # either the old contents or the new ones and never a prefix.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(path: str | Path, document: Any) -> Path:
    """Writes a JSON document atomically.

    Args:
        path: Destination.
        document: Any value acceptable to :func:`jsonable`.

    Returns:
        The path written.
    """
    return atomic_write_text(path, json.dumps(jsonable(document), indent=2) + "\n")


def digest_text(text: str) -> str:
    """Hashes a string with SHA-256.

    Args:
        text: Content to hash.

    Returns:
        A hex digest. Used for split definitions and request-id lists, which
        are content rather than files but still need to be identifiable.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Seeds:
    """Random seeds separated by what each one controls.

    One seed per run cannot be reproduced from, because the streams interact:
    changing the number of exits changes how many draws construction consumes,
    which silently moves the data order. Naming the streams makes each one
    independently settable, and makes the two rules below expressible.

    Two rules follow from the causal design and are enforced by the caller
    rather than by this dataclass:

    - ``model_init`` must **not** be offset by rank. Two arms that are meant to
      branch from a common parent cannot do so if their constructors consumed
      different streams, and a rank offset guarantees they did.
    - ``data_order`` and ``dropout`` **should** be offset by rank, so that
      ranks walk different data and sample different masks. This is deliberate
      and recorded, not incidental.

    Attributes:
        model_init: Parameter initialization.
        data_order: Which blocks each rank reads, in which order.
        dropout: Stochastic regularization during training.
        exit_sampling: Which exits a step scores when ``exits_per_step`` caps
            them. The present implementation draws no randomness here — the
            rotation is a deterministic function of the step counter, which is
            how every rank agrees without communicating — so this stream is
            recorded and unused. It exists because a future estimator that
            samples exits instead of rotating them must not be able to borrow
            the data or dropout stream to do it.
        controller: Controller initialization and its train/report split.
        benchmark: Workload generation for the latency benchmark.
    """

    model_init: int
    data_order: int
    dropout: int
    exit_sampling: int
    controller: int
    benchmark: int

    @classmethod
    def derive(cls, base: int) -> Seeds:
        """Spreads one base seed into the six streams.

        The offsets are arbitrary but fixed, and large enough that adjacent
        base seeds do not produce overlapping streams.

        Args:
            base: A single seed, normally ``TrainConfig.seed``.

        Returns:
            Six distinct seeds, reproducible from ``base`` alone.
        """
        return cls(
            model_init=base,
            data_order=base + 1_000,
            dropout=base + 2_000,
            exit_sampling=base + 3_000,
            controller=base + 4_000,
            benchmark=base + 5_000,
        )

    @classmethod
    def resolve(cls, base: int, **overrides: int | None) -> Seeds:
        """Derives from ``base``, then applies any explicitly set stream.

        Args:
            base: Base seed for streams that were not named.
            **overrides: Field name to seed. ``None`` values are ignored, which
                is what a command-line flag left unset looks like.

        Returns:
            The resolved seeds.

        Raises:
            ValueError: If an override names a stream that does not exist.
        """
        known = {entry.name for entry in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"unknown seed stream(s) {sorted(unknown)}; expected a subset of "
                f"{sorted(known)}."
            )
        values = asdict(cls.derive(base))
        values.update(
            {name: value for name, value in overrides.items() if value is not None}
        )
        return cls(**values)


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


#: Provenance fields that a research run must be able to state. A run that
#: cannot answer one of these produces a checkpoint nobody can place later, so
#: :meth:`RunArtifacts.create` refuses to start rather than discovering it at
#: analysis time. Overridable per call, because a smoke test on a laptop
#: outside a git checkout is a legitimate use of the same code.
REQUIRED_PROVENANCE = ("git_commit", "config", "seeds", "command")


@dataclass
class RunArtifacts:
    """A directory that describes one long run as it happens.

    :class:`RunRecord` writes provenance and results together once, at the end.
    That is the wrong shape for a cluster job: it may be preempted, it produces
    metrics continuously, and its per-request output can be larger than memory.
    This class owns a directory instead, with a fixed layout::

        <run_dir>/
          resolved_config.json     complete configuration, as it was resolved
          command.txt              a command that reconstructs the run
          environment.json         interpreter, framework, package lock
          hardware.json            host, accelerators, scheduler allocation
          git_commit.txt           commit, branch, dirty flag
          git_diff.patch           uncommitted changes, or a stated absence
          parent_checkpoint.sha256 the initialization every arm branched from
          data_manifest.json       inputs consumed, with content digests
          seeds.json               every stream, named by what it controls
          resume_chain.jsonl       one line per launch, including preemptions
          metrics.jsonl            append-only metric stream
          raw_records/             per-request output too large for one file
          checkpoints/             weights and full training state

    Every file is written atomically, so a run killed mid-write leaves either
    the previous contents or the new ones.

    Attributes:
        run_dir: Root of the directory described above.
        script: Module that produced it.
        schema_version: Layout version, written into every JSON file.
    """

    run_dir: Path
    script: str
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        script: str,
        config: dict[str, Any] | None = None,
        seeds: Seeds | dict[str, int] | None = None,
        inputs: dict[str, Any] | None = None,
        command: list[str] | str | None = None,
        parent_checkpoint: str | Path | None = None,
        notes: list[str] | None = None,
        required: tuple[str, ...] = REQUIRED_PROVENANCE,
    ) -> RunArtifacts:
        """Creates the directory and writes every static provenance file.

        Args:
            run_dir: Directory to create. Existing contents are preserved, so a
                resumed run adds to its own record rather than replacing it.
            script: Module name, normally ``__spec__.name`` of the caller.
            config: Complete resolved configuration. Nested dataclasses are
                serialized recursively.
            seeds: Either a :class:`Seeds` instance or a mapping of stream name
                to seed.
            inputs: Files and datasets consumed. Values that name an existing
                path are hashed, so ``{"train": "data/train.bin"}`` becomes a
                record of which bytes were read rather than which name was
                typed.
            command: Argument list or command line reproducing the run.
                Defaults to the current process's ``sys.argv``.
            parent_checkpoint: Initialization the run branched from. Recorded as
                a digest, which is what makes "same parent" checkable rather
                than asserted.
            notes: Caveats to carry alongside the numbers.
            required: Provenance fields that must be available. See
                :data:`REQUIRED_PROVENANCE`.

        Returns:
            The initialized artifact directory.

        Raises:
            ValueError: If a field named in ``required`` is unavailable. This
                fires before training, which is the whole point: the failure
                costs a scheduler slot, not a run.
        """
        run_dir = Path(run_dir)
        (run_dir / "raw_records").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        state = git_state()
        diff = git_diff()
        resolved_seeds = (
            asdict(seeds) if isinstance(seeds, Seeds) else dict(seeds or {})
        )
        argv = command if command is not None else [sys.executable, *sys.argv]
        command_text = argv if isinstance(argv, str) else " ".join(str(a) for a in argv)
        manifest = _hash_inputs(inputs or {})
        parent_digest = (
            file_digest(parent_checkpoint) if parent_checkpoint is not None else None
        )

        available = {
            "git_commit": state.get("commit") is not None,
            "git_diff": diff is not None,
            "config": bool(config),
            "seeds": bool(resolved_seeds),
            "command": bool(command_text.strip()),
            "environment": True,
            "hardware": True,
            "data_manifest": bool(manifest),
            "parent_checkpoint": parent_digest is not None,
        }
        unknown = set(required) - set(available)
        if unknown:
            raise ValueError(
                f"required names unknown provenance field(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(available)}."
            )
        missing = [name for name in required if not available[name]]
        if missing:
            raise ValueError(
                f"cannot start: provenance field(s) {missing} are unavailable, and "
                f"this run declared them required. A result whose code, "
                f"configuration, or seeds cannot be stated is not reproducible. "
                f"Pass required=() to run anyway, and expect the record to say so."
            )

        artifacts = cls(run_dir=run_dir, script=script)

        atomic_write_json(
            run_dir / "resolved_config.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "script": script,
                "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config or {},
                "notes": list(notes or []),
            },
        )
        atomic_write_text(run_dir / "command.txt", command_text + "\n")
        atomic_write_json(
            run_dir / "environment.json",
            {"schema_version": ARTIFACT_SCHEMA_VERSION, **environment()},
        )
        atomic_write_json(
            run_dir / "hardware.json",
            {"schema_version": ARTIFACT_SCHEMA_VERSION, **hardware()},
        )
        atomic_write_json(
            run_dir / "git_commit.txt",
            {"schema_version": ARTIFACT_SCHEMA_VERSION, **state},
        )
        # An absent diff and an empty diff mean different things, and both are
        # worth stating in a file a human will open.
        atomic_write_text(
            run_dir / "git_diff.patch",
            diff
            if diff
            else (
                "# no git metadata available; the source that ran is not\n"
                "# identifiable from this record\n"
                if diff is None
                else "# working tree clean at HEAD\n"
            ),
        )
        atomic_write_json(
            run_dir / "seeds.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "seeds": resolved_seeds,
                "convention": (
                    "model_init is not offset by rank so causal arms share a "
                    "parent; data_order and dropout are offset by rank on "
                    "purpose; exit_sampling must be identical across ranks."
                ),
            },
        )
        atomic_write_json(
            run_dir / "data_manifest.json",
            {"schema_version": ARTIFACT_SCHEMA_VERSION, "inputs": manifest},
        )
        atomic_write_text(
            run_dir / "parent_checkpoint.sha256",
            f"{parent_digest}\n"
            if parent_digest
            else "# no parent checkpoint: this run initialized from its seed\n",
        )
        return artifacts

    @property
    def metrics_path(self) -> Path:
        """Path of the append-only metric stream."""
        return self.run_dir / "metrics.jsonl"

    @property
    def raw_records_dir(self) -> Path:
        """Directory for per-request output."""
        return self.run_dir / "raw_records"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for weights and training state."""
        return self.run_dir / "checkpoints"

    def log_metric(self, values: dict[str, Any]) -> None:
        """Appends one metric record.

        This is the local record that must exist whether or not a tracking
        service was reachable. A run whose only history lives in a hosted
        dashboard cannot be reanalyzed after the project ends.

        Args:
            values: Metric names to values. A ``wall_time`` field is added.
        """
        self._append(self.metrics_path, {"wall_time": time.time(), **values})

    def record_resume(self, values: dict[str, Any]) -> None:
        """Appends one launch to the resume chain.

        Args:
            values: Facts about this launch, normally the update it started
                from, the checkpoint it restored, and the scheduler job id.
                Together the lines say how a final checkpoint was actually
                reached, which a single ``step`` field cannot.
        """
        self._append(
            self.resume_chain_path,
            {
                "wall_time": time.time(),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "scheduler": scheduler_allocation(),
                **values,
            },
        )

    @property
    def resume_chain_path(self) -> Path:
        """Path of the launch history."""
        return self.run_dir / "resume_chain.jsonl"

    def read_metrics(self) -> list[dict[str, Any]]:
        """Reads the metric stream back.

        Returns:
            One dict per line, in write order. Lines that are not valid JSON
            are skipped rather than raising, because the last line of a file
            whose process was killed mid-append can be a fragment.
        """
        return _read_jsonl(self.metrics_path)

    def read_resume_chain(self) -> list[dict[str, Any]]:
        """Reads the launch history back.

        Returns:
            One dict per launch, in order.
        """
        return _read_jsonl(self.resume_chain_path)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        """Appends one JSON line, flushed so a killed process keeps it.

        Args:
            path: File to append to.
            record: Values to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(jsonable(record)) + "\n")
            handle.flush()

    def summary(self) -> str:
        """Renders the directory's identity for console output.

        Returns:
            A multi-line string suitable for printing at run start.
        """
        state = json.loads((self.run_dir / "git_commit.txt").read_text())
        commit = state.get("commit") or "unavailable"
        marker = "-dirty" if state.get("dirty") else ""
        return "\n".join(
            [
                f"run_dir   {self.run_dir}",
                f"script    {self.script}",
                f"commit    {commit[:12]}{marker}",
                f"schema    {self.schema_version}",
            ]
        )


def _hash_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Turns an input mapping into one that records content, not just names.

    Args:
        inputs: Names to values. A value that is a path to an existing file is
            replaced by a record carrying its digest and size.

    Returns:
        The same keys, with file-valued entries expanded.
    """
    manifest: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                manifest[name] = {
                    "path": str(path),
                    "sha256": file_digest(path),
                    "bytes": path.stat().st_size,
                }
                continue
        manifest[name] = jsonable(value)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads a JSON-lines file, tolerating a truncated final line.

    Args:
        path: File to read.

    Returns:
        The parsed records, or an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
