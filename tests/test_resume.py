"""Tests that a preempted run resumes into the same run, not a similar one.

A checkpoint that restores weights and optimizer moments is enough to keep
training. It is not enough to keep *the* training: the data cursor, the exit
rotation and every random stream restart, so the resumed job repeats data the
model has already seen, trains a different exit schedule, and reports a token
budget it did not consume. None of that shows up in the loss curve.

The acceptance test therefore runs the real entry point in real processes:

- arm A runs 100 optimizer updates in one process;
- arm B runs to update 50, is terminated, and is relaunched from its
  checkpoint to update 100.

Every observable is then compared — the block indices consumed, the exits
scored, the parameters, the optimizer moments, the scaler, the rotation
counter, and the next draw from each random stream. The remaining tests in
:class:`OmittingOneComponent` each disable one piece of that state and assert
the comparison notices, so the acceptance test cannot pass by accident.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import TrainConfig, TransformerConfig
from src.model import Transformer
from training.data import StatelessBlockSampler
from training.train import (
    CHECKPOINT_SCHEMA_VERSION,
    build_optimizer,
    random_states,
    restore_random_states,
    save_checkpoint,
)

REPO = Path(__file__).resolve().parent.parent

#: Updates arm A runs, and arm B reaches across two launches.
TOTAL_UPDATES = 100

#: Update arm B is terminated at.
BREAK_AT = 50

#: Driver executed in a subprocess. It patches away only the tokenizer, whose
#: vocabulary would otherwise have to be downloaded, and then calls the real
#: ``training.train.main`` so that provenance, sampling, checkpointing and
#: restoration are all the production paths.
DRIVER = '''
import json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, {repo!r})

import training.train as T
from src.config import TrainConfig, TransformerConfig

DATA_DIR, OUT_DIR, REPORT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_STEPS, BREAK_AT = int(sys.argv[4]), int(sys.argv[5])
RESUME = sys.argv[6] or None

MODEL = dict(
    vocab_size=64, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, ff_dim=64,
    exit_every=2, min_exit_layer=1, exits_per_step=1, dropout=0.1,
)

T.load_tokenizer = lambda name: None
T.config_from_tokenizer = lambda tok, max_seq_len, **over: TransformerConfig(
    max_seq_len=max_seq_len, **{{**MODEL, **over}}
)

# Pinned to CPU. The comparison is exact equality of parameters, and an
# accelerator whose reductions are not bitwise reproducible across launches
# would make this test flaky for a reason that has nothing to do with resume.
_setup = T.distributed.setup
T.distributed.setup = lambda backend="auto": T.distributed.DistributedContext(
    0, 0, 1, torch.device("cpu"), enabled=False
)

# Every training forward records what it was given and which exits it scored.
# Those are the two things a resumed run silently gets wrong.
LOG = []


class Logged(T.Transformer):
    def forward(self, input_ids, targets=None, cache=None):
        out = super().forward(input_ids, targets=targets, cache=cache)
        if self.training:
            LOG.append(
                {{
                    "rotation": int(self._step_counter.item()),
                    "first_tokens": input_ids[:, 0].tolist(),
                    "checksum": int(input_ids.sum()),
                    "exits": sorted(out.exit_losses),
                }}
            )
        return out


T.Transformer = Logged
_save = T.save_checkpoint


def report(model, optimizer, scaler, updates):
    """Writes every piece of comparable state, then leaves."""
    document = {{
        "updates": updates,
        "log": LOG,
        "parameters": {{
            name: [round(float(v), 10) for v in tensor.flatten()[:8].tolist()]
            for name, tensor in sorted(model.state_dict().items())
        }},
        "parameter_norm": round(
            float(sum(p.double().pow(2).sum() for p in model.parameters()) ** 0.5), 10
        ),
        "optimizer_exp_avg_norm": round(
            float(
                sum(
                    s["exp_avg"].double().pow(2).sum()
                    for s in optimizer.state.values()
                    if "exp_avg" in s
                )
                ** 0.5
            ),
            10,
        ),
        "optimizer_steps": sorted(
            int(s["step"]) for s in optimizer.state.values() if "step" in s
        )[:4],
        "scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
        "rotation": int(model._step_counter.item()),
        # The next draw from each stream. Equal parameters with unequal next
        # draws means the run continued from the wrong point in the stream.
        "next_torch": round(float(torch.randn(1).item()), 10),
        "next_python": __import__("random").random(),
        "next_numpy": float(__import__("numpy").random.random()),
    }}
    Path(REPORT).write_text(json.dumps(document))


def stopping_save(path, model, optimizer, step, mcfg, tcfg, **kwargs):
    _save(path, model, optimizer, step, mcfg, tcfg, **kwargs)
    if BREAK_AT and step >= BREAK_AT:
        report(model, optimizer, kwargs.get("scaler"), step)
        # A real preemption, not a clean return: nothing after this point in
        # main() gets a chance to tidy up, which is the situation resume has to
        # survive.
        os._exit(0)


T.save_checkpoint = stopping_save

config = TrainConfig(
    data_dir=DATA_DIR, out_dir=OUT_DIR, seq_len=16, batch_size=2,
    grad_accum_steps=2, max_steps=MAX_STEPS, warmup_steps=5,
    learning_rate=1e-3, min_lr=1e-4, dtype="fp32", compile_model=False,
    seed=7, num_workers=0, eval_every=0, sweep_every=0,
    save_every={break_at} or MAX_STEPS, log_every=1000,
    resume_from=RESUME, wandb_project=None,
)

T.main(config)

# Reached only by the uninterrupted arm and by the resumed second launch.
model = Logged(TransformerConfig(max_seq_len=16, **MODEL))
state = torch.load(
    Path(OUT_DIR) / "final.pt", map_location="cpu", weights_only=False
)
model.load_state_dict(state["model"])
model._step_counter.fill_(state["step_counter"])
optimizer = T.build_optimizer(model, config, torch.device("cpu"))
optimizer.load_state_dict(state["optimizer"])
scaler = torch.amp.GradScaler("cpu", enabled=False)
report(model, optimizer, scaler, state["completed_updates"])
'''


def write_corpus(directory: Path, n_tokens: int = 20_000, seed: int = 3) -> None:
    """Writes a tokenized corpus without needing a tokenizer or a download.

    Args:
        directory: Data directory to populate with ``train.bin``/``val.bin``.
        n_tokens: Tokens in the training split.
        seed: Seed for the token values, which only need to be reproducible.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for name, count in (("train.bin", n_tokens), ("val.bin", n_tokens // 8)):
        rng.integers(0, 64, size=count, dtype=np.uint16).tofile(directory / name)


def run_driver(
    data_dir: Path,
    out_dir: Path,
    report: Path,
    max_steps: int,
    break_at: int,
    resume: str = "",
) -> dict:
    """Runs one launch of the driver in its own process.

    Args:
        data_dir: Prepared corpus.
        out_dir: Where checkpoints go.
        report: Where the state report is written.
        max_steps: ``max_steps`` for this launch. Held constant across arm B's
            two launches, because the cosine schedule reads it and a launch
            that shortened it would decay differently over the same updates.
        break_at: Update to terminate at, or ``0`` to run to completion.
        resume: Checkpoint to resume from, or ``""``.

    Returns:
        The parsed state report.

    Raises:
        AssertionError: If the process failed.
    """
    script = out_dir.parent / f"driver_{out_dir.name}.py"
    script.write_text(DRIVER.format(repo=str(REPO), break_at=break_at))
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(data_dir),
            str(out_dir),
            str(report),
            str(max_steps),
            str(break_at),
            resume,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
    )
    assert report.exists(), (
        f"driver wrote no report (returncode {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads(report.read_text())


class ExactResume(unittest.TestCase):
    """One hundred updates, uninterrupted, must equal fifty plus fifty."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        data = root / "data"
        write_corpus(data)

        cls.uninterrupted = run_driver(
            data, root / "armA", root / "armA.json", TOTAL_UPDATES, 0
        )
        # Arm B, first launch: same schedule, terminated at BREAK_AT.
        run_driver(
            data, root / "armB", root / "armB-first.json", TOTAL_UPDATES, BREAK_AT
        )
        cls.resumed = run_driver(
            data,
            root / "armB",
            root / "armB-second.json",
            TOTAL_UPDATES,
            0,
            resume=str(root / "armB" / f"step-{BREAK_AT:06d}.pt"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_both_arms_completed_the_same_updates(self) -> None:
        self.assertEqual(self.uninterrupted["updates"], TOTAL_UPDATES)
        self.assertEqual(self.resumed["updates"], TOTAL_UPDATES)

    def test_the_resumed_arm_reads_the_next_unseen_batches(self) -> None:
        """The failure a stateful loader produces: the corpus restarts."""
        forwards_before = BREAK_AT * 2  # grad_accum_steps
        expected = self.uninterrupted["log"][forwards_before:]
        actual = self.resumed["log"]
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(
            [entry["first_tokens"] for entry in actual],
            [entry["first_tokens"] for entry in expected],
        )
        self.assertEqual(
            [entry["checksum"] for entry in actual],
            [entry["checksum"] for entry in expected],
        )

    def test_the_resumed_arm_trains_the_same_exit_rotation(self) -> None:
        forwards_before = BREAK_AT * 2
        expected = self.uninterrupted["log"][forwards_before:]
        self.assertEqual(
            [entry["exits"] for entry in self.resumed["log"]],
            [entry["exits"] for entry in expected],
        )
        self.assertEqual(
            [entry["rotation"] for entry in self.resumed["log"]],
            [entry["rotation"] for entry in expected],
        )

    def test_parameters_agree(self) -> None:
        self.assertEqual(
            self.resumed["parameter_norm"], self.uninterrupted["parameter_norm"]
        )
        self.assertEqual(
            self.resumed["parameters"], self.uninterrupted["parameters"]
        )

    def test_optimizer_state_agrees(self) -> None:
        self.assertEqual(
            self.resumed["optimizer_exp_avg_norm"],
            self.uninterrupted["optimizer_exp_avg_norm"],
        )
        self.assertEqual(
            self.resumed["optimizer_steps"], self.uninterrupted["optimizer_steps"]
        )

    def test_the_rotation_counter_agrees(self) -> None:
        self.assertEqual(self.resumed["rotation"], self.uninterrupted["rotation"])

    def test_the_next_draw_from_every_stream_agrees(self) -> None:
        """Equal weights with unequal next draws is still an inexact resume."""
        for stream in ("next_torch", "next_python", "next_numpy"):
            with self.subTest(stream=stream):
                self.assertEqual(
                    self.resumed[stream], self.uninterrupted[stream], stream
                )


def config(**updates) -> TransformerConfig:
    """Builds a small architecture for the omission tests."""
    values = dict(
        vocab_size=64, d_model=32, n_layers=6, n_heads=4, n_kv_heads=2,
        ff_dim=64, max_seq_len=32, exit_every=2, min_exit_layer=1,
        exits_per_step=1, dropout=0.1,
    )
    values.update(updates)
    return TransformerConfig(**values)


class OmittingOneComponent(unittest.TestCase):
    """Each state component, dropped on its own, changes an observable.

    Without these, :class:`ExactResume` could pass while a component was dead
    code — two runs can agree for reasons unrelated to the state being
    restored. Each test here is the counterfactual for one saved field.
    """

    def test_dropping_the_data_cursor_repeats_the_corpus(self) -> None:
        """The cursor is not stored; it is derived from the update count."""
        sampler = StatelessBlockSampler(n_blocks=500, batch_size=4, seed=11)
        resumed = StatelessBlockSampler(
            n_blocks=500, batch_size=4, seed=11, start_micro_batch=50
        )
        restarted = StatelessBlockSampler(n_blocks=500, batch_size=4, seed=11)

        self.assertEqual(resumed.blocks_for(50), sampler.blocks_for(50))
        # A loader restarted at zero serves batch 50's slot from batch 0's data.
        self.assertNotEqual(list(restarted.__iter__().__next__() for _ in range(1)),
                            [])
        self.assertNotEqual(restarted.blocks_for(0), sampler.blocks_for(50))

    def test_dropping_the_rotation_counter_changes_which_exits_train(self) -> None:
        model = Transformer(config())
        n_exits = len(model.config.exit_layers)

        model._step_counter.fill_(37)
        continued = model._select_exits(n_exits)
        model._step_counter.fill_(0)
        restarted = model._select_exits(n_exits)

        self.assertNotEqual(
            continued,
            restarted,
            "exit rotation must depend on the counter, or restoring it is "
            "pointless and losing it is undetectable",
        )

    def test_dropping_the_random_states_changes_the_next_draw(self) -> None:
        torch.manual_seed(5)
        for _ in range(13):
            torch.randn(4)
        captured = random_states()
        expected = torch.randn(4)

        # A resume that reseeds from the config rather than restoring the
        # stream: same seed, wrong position.
        torch.manual_seed(5)
        self.assertFalse(torch.equal(torch.randn(4), expected))

        restore_random_states(captured)
        self.assertTrue(torch.equal(torch.randn(4), expected))

    def test_dropping_the_optimizer_moments_changes_the_next_update(self) -> None:
        torch.manual_seed(0)
        model = Transformer(config())
        train_config = TrainConfig(dtype="fp32", compile_model=False)
        optimizer = build_optimizer(model, train_config, torch.device("cpu"))

        ids = torch.randint(0, 64, (2, 16))
        for _ in range(4):
            model(ids, targets=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.pt"
            save_checkpoint(
                path, model, optimizer, 4, model.config, train_config
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(state["schema_version"], CHECKPOINT_SCHEMA_VERSION)

            def advance(load_moments: bool) -> torch.Tensor:
                torch.manual_seed(0)
                fresh = Transformer(config())
                fresh.load_state_dict(state["model"])
                fresh_optimizer = build_optimizer(
                    fresh, train_config, torch.device("cpu")
                )
                if load_moments:
                    fresh_optimizer.load_state_dict(state["optimizer"])
                fresh(ids, targets=ids).loss.backward()
                fresh_optimizer.step()
                return fresh.blocks[0].attn.q_proj.weight.detach().clone()

            self.assertFalse(
                torch.equal(advance(True), advance(False)),
                "AdamW moments must affect the next update, or saving them is "
                "decoration",
            )

    def test_the_checkpoint_carries_every_component(self) -> None:
        """The contract, asserted directly, so a field cannot quietly vanish."""
        torch.manual_seed(0)
        model = Transformer(config())
        train_config = TrainConfig(dtype="fp32", compile_model=False, seed=99)
        optimizer = build_optimizer(model, train_config, torch.device("cpu"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                12,
                model.config,
                train_config,
                tokens=12 * 4096,
                lineage=[{"start_update": 0}],
            )
            state = torch.load(path, map_location="cpu", weights_only=False)

        for key in (
            "schema_version", "model", "optimizer", "scaler", "step",
            "completed_updates", "completed_tokens", "model_config",
            "train_config", "seeds", "step_counter", "random_states", "lineage",
        ):
            with self.subTest(key=key):
                self.assertIn(key, state)

        for stream in ("python", "numpy", "torch_cpu", "torch_cuda"):
            with self.subTest(stream=stream):
                self.assertIn(stream, state["random_states"])

        self.assertEqual(state["seeds"]["model_init"], 99)
        self.assertEqual(state["completed_tokens"], 12 * 4096)


class StatelessSampling(unittest.TestCase):
    """Properties of the sampler that resume exactness rests on."""

    def test_ranks_read_disjoint_blocks(self) -> None:
        ranks = [
            StatelessBlockSampler(
                n_blocks=64, batch_size=4, world_size=4, rank=rank, seed=2
            )
            for rank in range(4)
        ]
        blocks = [set(sampler.blocks_for(0)) for sampler in ranks]
        union = set().union(*blocks)
        self.assertEqual(len(union), 16, "a rank read a block another rank had")

    def test_an_epoch_covers_the_corpus_exactly_once(self) -> None:
        sampler = StatelessBlockSampler(n_blocks=40, batch_size=4, seed=4)
        seen = [sampler.block_at(position) for position in range(40)]
        self.assertEqual(sorted(seen), list(range(40)))

    def test_consecutive_epochs_reshuffle(self) -> None:
        sampler = StatelessBlockSampler(n_blocks=40, batch_size=4, seed=4)
        first = [sampler.block_at(position) for position in range(40)]
        second = [sampler.block_at(40 + position) for position in range(40)]
        self.assertEqual(sorted(second), list(range(40)))
        self.assertNotEqual(first, second)

    def test_order_is_a_pure_function_of_seed_and_position(self) -> None:
        left = StatelessBlockSampler(n_blocks=97, batch_size=3, seed=8)
        right = StatelessBlockSampler(
            n_blocks=97, batch_size=3, seed=8, start_micro_batch=500
        )
        self.assertEqual(left.blocks_for(500), right.blocks_for(500))

    def test_a_different_seed_gives_a_different_order(self) -> None:
        left = StatelessBlockSampler(n_blocks=97, batch_size=3, seed=8)
        right = StatelessBlockSampler(n_blocks=97, batch_size=3, seed=9)
        self.assertNotEqual(left.blocks_for(0), right.blocks_for(0))

    def test_unshuffled_order_is_the_file_order(self) -> None:
        sampler = StatelessBlockSampler(
            n_blocks=40, batch_size=4, seed=4, shuffle=False
        )
        self.assertEqual(sampler.blocks_for(0), [0, 1, 2, 3])
        self.assertEqual(sampler.blocks_for(3), [12, 13, 14, 15])

    def test_length_refuses_to_answer(self) -> None:
        sampler = StatelessBlockSampler(n_blocks=40, batch_size=4)
        with self.assertRaisesRegex(TypeError, "unbounded"):
            len(sampler)

    def test_invalid_arguments_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_blocks"):
            StatelessBlockSampler(n_blocks=0, batch_size=4)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            StatelessBlockSampler(n_blocks=4, batch_size=0)
        with self.assertRaisesRegex(ValueError, "start_micro_batch"):
            StatelessBlockSampler(n_blocks=4, batch_size=4, start_micro_batch=-1)


if __name__ == "__main__":
    unittest.main()
