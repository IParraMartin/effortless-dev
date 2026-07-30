"""Tests for the Slurm job scripts.

These exist because two bugs shipped in the same batch of scripts and neither was
catchable by the checks that were being run. `bash -n` finds syntax errors only,
and the whole suite of 429 Python tests never touches a `.sh` file, so both
failures were found by submitting to a cluster and reading a stack trace.

The two:

1. ``"${EXTRA_FLAGS[@]}"`` on an empty array raises *unbound variable* under
   ``set -u`` on bash before 4.4. Savio runs RHEL. Every affected script died on
   its first command, and since none of them takes extra flags in normal use the
   array was always empty.
2. ``source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"``. Slurm copies the batch
   script to ``/var/spool/slurmd/job<id>/``, so the sibling-file assumption that
   works when running a script directly is wrong under ``sbatch``. Every
   pre-existing script already had a ``_find_env`` locator with a comment naming
   this exact cause; the new scripts did not use it.

Both are properties of *text*, so they are checkable without a cluster.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent.parent / "jobs"

#: Every script here except the sourced environment. ``check.sh`` and
#: ``follow.sh`` are sourced into an interactive shell.
ALL_SCRIPTS = sorted(
    path
    for path in JOBS.glob("*.sh")
    if path.name not in {"_env.sh", "check.sh", "follow.sh"}
)

#: Scripts actually submitted with ``sbatch``, identified by their directives.
#: The rest run on a login node, where a failure is already on the terminal, so
#: the requirements below about traps and log paths do not apply to them.
BATCH_SCRIPTS = [
    path for path in ALL_SCRIPTS if "#SBATCH" in path.read_text()
]


class ScriptsParse(unittest.TestCase):
    """The check that was already being run, kept so its limits are visible."""

    def test_every_script_parses(self) -> None:
        for path in ALL_SCRIPTS + [JOBS / "_env.sh"]:
            with self.subTest(script=path.name):
                result = subprocess.run(
                    ["bash", "-n", str(path)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_there_are_scripts_to_check(self) -> None:
        """Guards against either glob silently matching nothing."""
        self.assertGreater(len(ALL_SCRIPTS), 5)
        self.assertGreater(len(BATCH_SCRIPTS), 3)


class EmptyArrayExpansion(unittest.TestCase):
    """Bug 1: unguarded array expansion under ``set -u``."""

    def test_the_failure_reproduces_on_this_bash(self) -> None:
        """Calibrates the test: unguarded really does fail here."""
        result = subprocess.run(
            ["bash", "-c", 'set -euo pipefail; a=(); echo "${a[@]}"'],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(
            result.returncode,
            0,
            "this bash tolerates unguarded empty-array expansion, so this test "
            "cannot detect the bug and needs a different mechanism",
        )
        self.assertIn("unbound variable", result.stderr)

    def test_the_guarded_form_survives(self) -> None:
        result = subprocess.run(
            ["bash", "-c", 'set -euo pipefail; a=(); echo "${a[@]+"${a[@]}"}"'],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_script_expands_an_array_unguarded(self) -> None:
        """The bug, as a property of the text.

        Any ``"${NAME[@]}"`` whose array can be empty must be written
        ``${NAME[@]+"${NAME[@]}"}``. Arrays built from a literal list are always
        non-empty and are allowed.
        """
        # Arrays assembled from literals in the script itself cannot be empty.
        allowed = {"ARCH", "ARCH_FLAGS", "COMMON", "PY", "MANIFEST_FLAGS",
                   "RESUME_FLAGS"}
        # The guarded form contains the unguarded form as a substring:
        #   ${NAME[@]+"${NAME[@]}"}
        # so guarded expansions are removed before scanning. Without this the
        # test reports every correct usage as a violation -- which it did, on the
        # first run, and the report was convincing enough to nearly send me
        # editing a script that was already right.
        guarded = re.compile(r'\$\{([A-Z_]+)\[@\]\+"\$\{\1\[@\]\}"\}')
        pattern = re.compile(r'"\$\{([A-Z_]+)\[@\]\}"')

        for path in ALL_SCRIPTS:
            # Comments are stripped as well as guarded forms. train.sh documents
            # this very rule in prose, quoting the unguarded form to explain what
            # not to write -- and the scan flagged the explanation.
            lines = [
                "" if line.lstrip().startswith("#") else line
                for line in path.read_text().splitlines()
            ]
            text = guarded.sub("<guarded>", "\n".join(lines))
            for match in pattern.finditer(text):
                name = match.group(1)
                if name in allowed:
                    continue
                line = text[: match.start()].count("\n") + 1
                self.fail(
                    f"{path.name}:{line} expands ${{{name}[@]}} unguarded. On bash "
                    f"before 4.4 this raises 'unbound variable' under set -u when "
                    f"the array is empty. Write "
                    f'${{{name}[@]+"${{{name}[@]}}"}} instead.'
                )

    def test_scripts_taking_passthrough_flags_guard_them(self) -> None:
        """The specific array that caused the outage."""
        for path in ALL_SCRIPTS:
            text = path.read_text()
            if "EXTRA_FLAGS=(" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertIn(
                    'EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"',
                    text,
                    f"{path.name} builds EXTRA_FLAGS but never expands it safely",
                )


class EnvLocation(unittest.TestCase):
    """Bug 2: the sibling-file assumption, wrong under sbatch."""

    def test_no_script_assumes_env_is_a_sibling(self) -> None:
        """Slurm copies the batch script away from the repository."""
        for path in ALL_SCRIPTS:
            text = path.read_text()
            with self.subTest(script=path.name):
                self.assertNotIn(
                    'dirname "${BASH_SOURCE[0]}")/_env.sh',
                    text,
                    f"{path.name} sources _env.sh as a sibling of the script. "
                    f"Under sbatch, BASH_SOURCE points at "
                    f"/var/spool/slurmd/job<id>/, which holds only the copied "
                    f"script. Use the _find_env locator.",
                )

    def test_every_submitted_script_locates_env(self) -> None:
        for path in ALL_SCRIPTS:
            text = path.read_text()
            if "_env.sh" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertIn("_find_env", text, f"{path.name} has no locator")
                self.assertIn('source "$(_find_env)"', text)

    def test_the_locator_checks_the_submit_directory_first(self) -> None:
        """SLURM_SUBMIT_DIR is the only reliable pointer back to the repo."""
        for path in ALL_SCRIPTS:
            text = path.read_text()
            if "_find_env" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertIn("SLURM_SUBMIT_DIR", text)


class FailureReporting(unittest.TestCase):
    """A job that dies must say so; an empty log is undiagnosable."""

    def test_env_defines_the_failure_report(self) -> None:
        text = (JOBS / "_env.sh").read_text()
        self.assertIn("report_failure()", text)
        for status in ("127", "137", "139", "143"):
            with self.subTest(status=status):
                self.assertIn(status, text)

    def test_every_submitted_script_installs_a_trap(self) -> None:
        for path in BATCH_SCRIPTS:
            text = path.read_text()
            if "_env.sh" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertRegex(
                    text,
                    r"trap (report_failure|_report_exit) EXIT",
                    f"{path.name} can fail without saying why",
                )

    def test_env_creates_the_log_directory(self) -> None:
        """Slurm opens the output file before the job body runs."""
        self.assertIn('mkdir -p "$REPO_DIR/logs"', (JOBS / "_env.sh").read_text())


class SbatchDirectives(unittest.TestCase):
    """Directives are only read before the first executable line."""

    def test_directives_precede_the_first_command(self) -> None:
        for path in BATCH_SCRIPTS:
            text = path.read_text().splitlines()
            first_command = next(
                (
                    index
                    for index, line in enumerate(text)
                    if line.strip() and not line.lstrip().startswith("#")
                ),
                len(text),
            )
            late = [
                index
                for index, line in enumerate(text)
                if line.startswith("#SBATCH") and index > first_command
            ]
            with self.subTest(script=path.name):
                self.assertEqual(
                    late,
                    [],
                    f"{path.name} has #SBATCH at line(s) "
                    f"{[i + 1 for i in late]}, after the first command on line "
                    f"{first_command + 1}. Slurm ignores those silently.",
                )

    def test_submitted_scripts_declare_output_paths(self) -> None:
        for path in BATCH_SCRIPTS:
            text = path.read_text()
            if "#SBATCH" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertIn("--output=logs/", text)
                self.assertIn("--error=logs/", text)


class CorpusSafety(unittest.TestCase):
    """No submitted job may collect from the synthetic corpus by default."""

    def test_collection_always_names_its_corpus(self) -> None:
        """The default is synthetic, which is a mechanism test, never evidence."""
        for path in ALL_SCRIPTS:
            text = path.read_text()
            if "collect_depth_trajectories" not in text:
                continue
            with self.subTest(script=path.name):
                self.assertIn(
                    "--corpus=real_text",
                    text,
                    f"{path.name} collects trajectories without naming a corpus, "
                    f"so it would use mixed_difficulty_corpus and label a "
                    f"synthetic token pattern as a routing result",
                )
                self.assertIn(
                    "--eos_id",
                    text,
                    f"{path.name} collects real text without --eos_id, so "
                    f"document boundaries are unknown and every interval it "
                    f"produces is unclustered and too narrow",
                )


if __name__ == "__main__":
    unittest.main()
