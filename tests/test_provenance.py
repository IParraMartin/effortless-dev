"""Tests for the run-artifact contract.

A result without provenance is not evidence, so these tests treat provenance as
a functional requirement rather than as logging. They check that a run can
state what it was — configuration, code, machine, seeds, inputs — that the
record cannot be left half-written by a scheduler kill, that a changed input
changes its digest, that a run which cannot describe itself refuses to start,
and that credentials never reach a file somebody may share.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest import mock

from src.config import TrainConfig, TransformerConfig
from utils.provenance import (
    ARTIFACT_SCHEMA_VERSION,
    REQUIRED_PROVENANCE,
    RunArtifacts,
    Seeds,
    atomic_write_json,
    atomic_write_text,
    environment,
    file_digest,
    hardware,
    installed_packages,
    jsonable,
    redacted,
    scheduler_allocation,
)


@dataclass
class Inner:
    """Nested dataclass used to check recursive serialization."""

    depths: tuple[int, ...] = (2, 4)
    label: str = "inner"


@dataclass
class Outer:
    """Outer dataclass holding another dataclass and a path."""

    inner: Inner = field(default_factory=Inner)
    where: Path = Path("data/train.bin")
    ratio: float = 0.5


class Serialization(unittest.TestCase):
    """Configurations must survive the round trip without losing structure."""

    def test_nested_configs_round_trip(self) -> None:
        document = jsonable(asdict(Outer()))
        restored = json.loads(json.dumps(document))

        self.assertEqual(restored["inner"]["depths"], [2, 4])
        self.assertEqual(restored["inner"]["label"], "inner")
        self.assertEqual(restored["where"], "data/train.bin")
        self.assertEqual(restored["ratio"], 0.5)

    def test_the_real_configurations_serialize_completely(self) -> None:
        """Every field, not the handful a dashboard happened to receive.

        The historical runs logged five model fields out of thirty. A reader of
        that record cannot tell what was trained.
        """
        train, model = TrainConfig(), TransformerConfig()
        document = jsonable({"train": asdict(train), "model": asdict(model)})
        json.dumps(document)

        for name in asdict(model):
            self.assertIn(name, document["model"], name)
        for name in asdict(train):
            self.assertIn(name, document["train"], name)

    def test_unserializable_values_degrade_rather_than_raise(self) -> None:
        """Provenance capture must never be the thing that fails a run."""
        self.assertIsInstance(jsonable({"handle": object()})["handle"], str)


class AtomicWrites(unittest.TestCase):
    """A killed process must leave either the old record or the new one."""

    def test_a_failed_overwrite_leaves_the_previous_document_intact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved_config.json"
            atomic_write_json(path, {"generation": 1})

            with mock.patch("utils.provenance.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"generation": 2})

            # Not truncated, not empty, not the new value: the old one.
            self.assertEqual(json.loads(path.read_text())["generation"], 1)

    def test_a_failed_overwrite_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware.json"
            with mock.patch("utils.provenance.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"device": "a40"})

            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_the_destination_is_never_opened_for_truncation(self) -> None:
        """The mechanism, not only its effect.

        Writing in place would make the file briefly parse as a prefix of valid
        JSON, which reads as a record until it is loaded.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            atomic_write_json(path, {"model_init": 1})
            opened: list[str] = []
            real_open = Path.open

            def watching(self, *args, **kwargs):
                if args and "w" in str(args[0]):
                    opened.append(str(self))
                return real_open(self, *args, **kwargs)

            with mock.patch.object(Path, "open", watching):
                atomic_write_json(path, {"model_init": 2})

            self.assertTrue(opened)
            self.assertNotIn(str(path), opened)
            self.assertEqual(json.loads(path.read_text())["model_init"], 2)

    def test_text_and_json_writers_create_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deep = Path(directory) / "a" / "b" / "command.txt"
            atomic_write_text(deep, "python -m training.train\n")
            self.assertEqual(deep.read_text(), "python -m training.train\n")


class Digests(unittest.TestCase):
    """A path is not an identifier; the bytes at it are."""

    def test_a_changed_artifact_changes_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            path.write_bytes(b"weights-v1")
            before = file_digest(path)
            path.write_bytes(b"weights-v2")

            self.assertNotEqual(before, file_digest(path))

    def test_identical_bytes_at_different_paths_hash_alike(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a.pt", Path(directory) / "b.pt"
            left.write_bytes(b"same")
            right.write_bytes(b"same")

            self.assertEqual(file_digest(left), file_digest(right))

    def test_a_missing_file_reports_absence_rather_than_raising(self) -> None:
        self.assertIsNone(file_digest("does/not/exist.pt"))

    def test_inputs_are_recorded_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.bin").write_bytes(b"0" * 64)
            artifacts = RunArtifacts.create(
                root / "run",
                script="tests",
                config={"a": 1},
                seeds=Seeds.derive(1),
                inputs={"train": str(root / "train.bin"), "note": "held out"},
                command="python -m tests",
                required=(),
            )
            manifest = json.loads(
                (artifacts.run_dir / "data_manifest.json").read_text()
            )["inputs"]

            self.assertEqual(manifest["train"]["bytes"], 64)
            self.assertEqual(len(manifest["train"]["sha256"]), 64)
            self.assertEqual(manifest["note"], "held out")


class RequiredProvenance(unittest.TestCase):
    """A run that cannot describe itself must fail before it costs anything."""

    def test_a_missing_required_field_fails_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                RunArtifacts.create(
                    Path(directory) / "run",
                    script="tests",
                    config={},  # required and absent
                    seeds=Seeds.derive(1),
                    command="python -m tests",
                )
            self.assertIn("config", str(caught.exception))
            self.assertIn("not reproducible", str(caught.exception))

    def test_absent_git_metadata_fails_when_declared_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "utils.provenance.git_state",
                return_value={"commit": None, "branch": None, "dirty": None},
            ):
                with self.assertRaises(ValueError) as caught:
                    RunArtifacts.create(
                        Path(directory) / "run",
                        script="tests",
                        config={"a": 1},
                        seeds=Seeds.derive(1),
                        command="python -m tests",
                        required=("git_commit",),
                    )
            self.assertIn("git_commit", str(caught.exception))

    def test_absent_git_metadata_is_stated_rather_than_invented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("utils.provenance._git", return_value=None):
                artifacts = RunArtifacts.create(
                    Path(directory) / "run",
                    script="tests",
                    config={"a": 1},
                    seeds=Seeds.derive(1),
                    command="python -m tests",
                    required=(),
                )
            state = json.loads((artifacts.run_dir / "git_commit.txt").read_text())
            self.assertIsNone(state["commit"])
            self.assertIn(
                "no git metadata", (artifacts.run_dir / "git_diff.patch").read_text()
            )

    def test_a_typo_in_required_is_reported_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown provenance field"):
                RunArtifacts.create(
                    Path(directory) / "run",
                    script="tests",
                    config={"a": 1},
                    seeds=Seeds.derive(1),
                    command="python -m tests",
                    required=("git_comit",),
                )

    def test_the_default_requirement_set_names_the_essentials(self) -> None:
        self.assertEqual(
            set(REQUIRED_PROVENANCE), {"git_commit", "config", "seeds", "command"}
        )


class Secrets(unittest.TestCase):
    """Credentials must not reach a directory that gets shared."""

    def test_credential_shaped_keys_are_redacted(self) -> None:
        result = redacted(
            {
                "WANDB_API_KEY": "abcd",
                "HF_TOKEN": "efgh",
                "AWS_SECRET_ACCESS_KEY": "ijkl",
                "SLURM_JOB_ID": "123",
            }
        )
        self.assertEqual(result["SLURM_JOB_ID"], "123")
        for name in ("WANDB_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"):
            with self.subTest(name=name):
                self.assertEqual(result[name], "<redacted>")

    def test_the_scheduler_record_is_an_allowlist(self) -> None:
        """A denylist is one forgotten marker away from a leak."""
        with mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": "secret-value", "SLURM_JOB_ID": "42"},
            clear=False,
        ):
            allocation = scheduler_allocation()

        self.assertEqual(allocation.get("SLURM_JOB_ID"), "42")
        self.assertNotIn("WANDB_API_KEY", allocation)

    def test_no_written_file_contains_a_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "WANDB_API_KEY": "sentinel-wandb-key",
                    "HF_TOKEN": "sentinel-hf-token",
                    "SLURM_JOB_ID": "99",
                },
                clear=False,
            ):
                artifacts = RunArtifacts.create(
                    Path(directory) / "run",
                    script="tests",
                    config={"a": 1},
                    seeds=Seeds.derive(1),
                    command="python -m tests",
                    required=(),
                )
                artifacts.log_metric({"loss": 1.0})
                artifacts.record_resume({"start_update": 0})

            written = [
                path
                for path in artifacts.run_dir.rglob("*")
                if path.is_file()
            ]
            self.assertTrue(written)
            for path in written:
                blob = path.read_text(errors="replace")
                with self.subTest(path=path.name):
                    self.assertNotIn("sentinel-wandb-key", blob)
                    self.assertNotIn("sentinel-hf-token", blob)
            joined = "".join(path.read_text(errors="replace") for path in written)
            self.assertIn("99", joined, "the allowlisted job id should be recorded")


class DirectoryContract(unittest.TestCase):
    """The layout other tooling depends on."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.artifacts = RunArtifacts.create(
            Path(self._temporary.name) / "run",
            script="training.train",
            config={"train": asdict(TrainConfig()), "model": asdict(TransformerConfig())},
            seeds=Seeds.derive(1337),
            inputs={"corpus": "fineweb-edu"},
            command=["python", "-m", "training.train", "--max_steps=10"],
            notes=["smoke run"],
            required=(),
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_every_contracted_file_exists(self) -> None:
        for name in (
            "resolved_config.json",
            "command.txt",
            "environment.json",
            "hardware.json",
            "git_commit.txt",
            "git_diff.patch",
            "parent_checkpoint.sha256",
            "data_manifest.json",
            "seeds.json",
        ):
            with self.subTest(name=name):
                self.assertTrue((self.artifacts.run_dir / name).is_file(), name)
        for name in ("raw_records", "checkpoints"):
            with self.subTest(name=name):
                self.assertTrue((self.artifacts.run_dir / name).is_dir(), name)

    def test_every_json_file_carries_a_schema_version(self) -> None:
        for path in self.artifacts.run_dir.glob("*.json"):
            with self.subTest(name=path.name):
                self.assertEqual(
                    json.loads(path.read_text())["schema_version"],
                    ARTIFACT_SCHEMA_VERSION,
                )

    def test_the_command_reconstructs_the_run(self) -> None:
        self.assertEqual(
            (self.artifacts.run_dir / "command.txt").read_text().strip(),
            "python -m training.train --max_steps=10",
        )

    def test_seeds_are_recorded_by_purpose(self) -> None:
        recorded = json.loads((self.artifacts.run_dir / "seeds.json").read_text())
        self.assertEqual(
            set(recorded["seeds"]),
            {
                "model_init",
                "data_order",
                "dropout",
                "exit_sampling",
                "controller",
                "benchmark",
            },
        )
        self.assertIn("not offset by rank", recorded["convention"])

    def test_metrics_append_and_read_back_in_order(self) -> None:
        for step in range(3):
            self.artifacts.log_metric({"step": step, "loss": 10.0 - step})
        history = self.artifacts.read_metrics()

        self.assertEqual([row["step"] for row in history], [0, 1, 2])
        self.assertTrue(all("wall_time" in row for row in history))

    def test_a_truncated_final_metric_line_does_not_break_reading(self) -> None:
        self.artifacts.log_metric({"step": 0})
        with self.artifacts.metrics_path.open("a") as handle:
            handle.write('{"step": 1, "loss":')

        self.assertEqual(len(self.artifacts.read_metrics()), 1)

    def test_the_resume_chain_records_one_line_per_launch(self) -> None:
        self.artifacts.record_resume({"start_update": 0})
        self.artifacts.record_resume({"start_update": 500, "resumed_from": "a.pt"})
        chain = self.artifacts.read_resume_chain()

        self.assertEqual([row["start_update"] for row in chain], [0, 500])
        self.assertEqual(chain[1]["resumed_from"], "a.pt")
        self.assertIn("scheduler", chain[0])

    def test_recreating_over_an_existing_directory_keeps_the_history(self) -> None:
        """A relaunch adds to its own record; it does not replace it."""
        self.artifacts.log_metric({"step": 0})
        again = RunArtifacts.create(
            self.artifacts.run_dir,
            script="training.train",
            config={"a": 1},
            seeds=Seeds.derive(1337),
            command="python -m training.train",
            required=(),
        )
        self.assertEqual(len(again.read_metrics()), 1)

    def test_a_parent_checkpoint_is_recorded_as_a_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.pt"
            parent.write_bytes(b"common-initialization")
            artifacts = RunArtifacts.create(
                Path(directory) / "run",
                script="tests",
                config={"a": 1},
                seeds=Seeds.derive(1),
                command="python -m tests",
                parent_checkpoint=parent,
                required=("parent_checkpoint",),
            )
            recorded = (artifacts.run_dir / "parent_checkpoint.sha256").read_text()

            self.assertEqual(recorded.strip(), file_digest(parent))

    def test_no_parent_is_stated_explicitly(self) -> None:
        recorded = (self.artifacts.run_dir / "parent_checkpoint.sha256").read_text()
        self.assertIn("no parent checkpoint", recorded)


class EnvironmentCapture(unittest.TestCase):
    """The fields that decide whether two runs are comparable."""

    def test_the_package_lock_is_recorded(self) -> None:
        packages = installed_packages()
        self.assertIsInstance(packages, dict)
        self.assertIn("torch", {name.lower() for name in packages})

    def test_the_software_stack_names_its_versions(self) -> None:
        captured = environment()
        for name in ("python_version", "torch", "torch_cuda", "packages"):
            with self.subTest(name=name):
                self.assertIn(name, captured)

    def test_the_hardware_record_identifies_the_machine(self) -> None:
        captured = hardware()
        for name in ("hostname", "cpu_count", "device_type", "accelerators", "scheduler"):
            with self.subTest(name=name):
                self.assertIn(name, captured)
        self.assertIsInstance(captured["accelerators"], list)


class SeedStreams(unittest.TestCase):
    """One seed per purpose, derived so the common case stays one number."""

    def test_derivation_gives_distinct_streams(self) -> None:
        seeds = Seeds.derive(1337)
        values = list(asdict(seeds).values())
        self.assertEqual(len(set(values)), len(values))
        self.assertEqual(seeds.model_init, 1337)

    def test_derivation_is_reproducible(self) -> None:
        self.assertEqual(Seeds.derive(7), Seeds.derive(7))
        self.assertNotEqual(Seeds.derive(7), Seeds.derive(8))

    def test_an_explicit_stream_overrides_only_itself(self) -> None:
        resolved = Seeds.resolve(1337, data_order=99)
        self.assertEqual(resolved.data_order, 99)
        self.assertEqual(resolved.model_init, Seeds.derive(1337).model_init)

    def test_an_unset_stream_falls_back_to_derivation(self) -> None:
        self.assertEqual(Seeds.resolve(1337, data_order=None), Seeds.derive(1337))

    def test_an_unknown_stream_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown seed stream"):
            Seeds.resolve(1337, dat_order=5)

    def test_the_train_config_resolves_its_own_streams(self) -> None:
        config = TrainConfig(seed=11, data_order_seed=4321)
        self.assertEqual(config.seeds().data_order, 4321)
        self.assertEqual(config.seeds().model_init, 11)


if __name__ == "__main__":
    unittest.main()
