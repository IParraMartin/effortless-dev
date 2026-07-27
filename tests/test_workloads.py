"""Tests for the diagnostic corpus.

A workload is an experimental instrument, and a broken one produces numbers
that look fine. Three of these tests exist because a specific construction bug
already got through and was only caught by measuring the model afterwards:
padding by repetition destroyed the induction structure, and a fixed repeat
distance let a positional shortcut stand in for the lookup. Both showed up as
"the model did not learn", not as a corpus error.
"""

from __future__ import annotations

import unittest

import torch

from experiments.workloads import (
    FIRST_CONTENT_TOKEN,
    TAG_EASY,
    TAG_HARD,
    Workload,
    mixed_difficulty_corpus,
    split_by_source,
)


class Structure(unittest.TestCase):
    """The corpus contains the structure it claims to."""

    def setUp(self) -> None:
        self.workload = mixed_difficulty_corpus(n_requests=64, seed=0)

    def test_both_halves_share_one_prompt_length(self) -> None:
        """Unequal lengths would force padding, which broke this before."""
        self.assertEqual(self.workload.prompts.size(1), 1 + 12 + 6)
        self.assertEqual(len(set(self.workload.difficulty)), 2)

    def test_tags_match_difficulty(self) -> None:
        for row, kind in enumerate(self.workload.difficulty):
            expected = TAG_HARD if kind == "hard" else TAG_EASY
            self.assertEqual(int(self.workload.prompts[row, 0]), expected)

    def test_content_tokens_never_collide_with_tags(self) -> None:
        body = self.workload.prompts[:, 1:]
        self.assertGreaterEqual(int(body.min()), FIRST_CONTENT_TOKEN)
        self.assertGreaterEqual(
            int(self.workload.references.min()), FIRST_CONTENT_TOKEN
        )

    def test_easy_continuation_is_the_previous_token(self) -> None:
        for row, kind in enumerate(self.workload.difficulty):
            if kind != "easy":
                continue
            last = self.workload.prompts[row, -1]
            self.assertTrue(bool((self.workload.references[row] == last).all()))

    def test_hard_continuation_is_recoverable_by_induction(self) -> None:
        """The answer must follow an earlier occurrence of the last token."""
        for row, kind in enumerate(self.workload.difficulty):
            if kind != "hard":
                continue
            prompt = self.workload.prompts[row]
            answer = int(self.workload.references[row][0])

            matches = [
                position
                for position in range(1, prompt.numel() - 1)
                if int(prompt[position]) == int(prompt[-1])
                and int(prompt[position + 1]) == answer
            ]
            self.assertTrue(
                matches, f"row {row}: no earlier occurrence explains the answer"
            )

    def test_repeat_distance_varies(self) -> None:
        """A constant distance is solvable by position alone, with one layer."""
        distances = set()
        for row, kind in enumerate(self.workload.difficulty):
            if kind != "hard":
                continue
            prompt = self.workload.prompts[row]
            for position in range(1, prompt.numel() - 1):
                if int(prompt[position]) == int(prompt[-1]):
                    distances.add(prompt.numel() - 1 - position)
        self.assertGreater(len(distances), 1)

    def test_hard_fraction_is_respected(self) -> None:
        workload = mixed_difficulty_corpus(n_requests=100, hard_fraction=0.25)
        hard = sum(1 for kind in workload.difficulty if kind == "hard")
        self.assertEqual(hard, 25)

    def test_shape_arguments_are_rejected_when_impossible(self) -> None:
        with self.assertRaises(ValueError):
            mixed_difficulty_corpus(n_requests=4, vocab_size=2)
        with self.assertRaises(ValueError):
            mixed_difficulty_corpus(
                n_requests=4, block_len=6, repeat_len=4, continuation_len=4
            )
        with self.assertRaisesRegex(ValueError, "position alone"):
            mixed_difficulty_corpus(
                n_requests=4, block_len=6, repeat_len=1, continuation_len=4
            )

    def test_spec_round_trips(self) -> None:
        """Training resamples from this, so it has to reproduce the shape."""
        again = mixed_difficulty_corpus(
            n_requests=8, seed=1, **self.workload.spec
        )
        self.assertEqual(again.prompts.size(1), self.workload.prompts.size(1))
        self.assertEqual(
            again.references.size(1), self.workload.references.size(1)
        )


class Splitting(unittest.TestCase):
    """Splits are made by source, so structure cannot leak across them."""

    def test_no_source_appears_on_both_sides(self) -> None:
        workload = mixed_difficulty_corpus(n_requests=64, seed=0)
        train, validation = split_by_source(workload, 0.25, seed=0)
        self.assertFalse(
            set(train.source_ids) & set(validation.source_ids)
        )
        self.assertEqual(len(train) + len(validation), len(workload))

    def test_both_sides_are_non_empty(self) -> None:
        workload = mixed_difficulty_corpus(n_requests=8, seed=0)
        train, validation = split_by_source(workload, 0.25, seed=0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(validation), 0)

    def test_select_preserves_alignment(self) -> None:
        workload = mixed_difficulty_corpus(n_requests=16, seed=0)
        subset = workload.select([3, 1, 7])
        self.assertEqual(subset.source_ids, [3, 1, 7])
        self.assertTrue(
            torch.equal(subset.prompts[0], workload.prompts[3])
        )
        self.assertEqual(subset.difficulty[0], workload.difficulty[3])

    def test_grouped_sources_stay_together(self) -> None:
        """The reason splitting is by source rather than by row."""
        workload = mixed_difficulty_corpus(n_requests=16, seed=0)
        paired = Workload(
            prompts=workload.prompts,
            references=workload.references,
            difficulty=workload.difficulty,
            source_ids=[index // 2 for index in range(16)],
            spec=workload.spec,
        )
        train, validation = split_by_source(paired, 0.5, seed=0)
        for source in set(train.source_ids):
            self.assertNotIn(source, set(validation.source_ids))


if __name__ == "__main__":
    unittest.main()
