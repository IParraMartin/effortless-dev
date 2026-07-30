"""Tests for the tokenizer glue.

The two functions with real logic — vocabulary sizing and next-token batch
construction — are exercised against a fake tokenizer so the suite needs no
network. ``load_tokenizer`` is thin glue over ``AutoTokenizer`` and is left to
integration use.
"""

from __future__ import annotations

import unittest

import torch

from src.tokenizer import IGNORE_INDEX, config_from_tokenizer, make_training_batch


class FakeTokenizer:
    """Minimal stand-in exposing only what the functions under test call."""

    eos_token = "<|eos|>"

    def __init__(self, size: int = 50_257) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __call__(self, texts, **kwargs):
        # Returns fixed tensors regardless of text: the logic under test is the
        # shift and the pad-masking, not the encoding.
        input_ids = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ConfigFromTokenizerTests(unittest.TestCase):
    def test_rounds_up_to_multiple(self) -> None:
        config = config_from_tokenizer(FakeTokenizer(50_257), pad_vocab_to_multiple_of=64)
        self.assertEqual(config.vocab_size % 64, 0)
        self.assertGreaterEqual(config.vocab_size, 50_257)

    def test_exact_when_multiple_is_one(self) -> None:
        config = config_from_tokenizer(FakeTokenizer(1234), pad_vocab_to_multiple_of=1)
        self.assertEqual(config.vocab_size, 1234)

    def test_overrides_pass_through(self) -> None:
        config = config_from_tokenizer(
            FakeTokenizer(100), d_model=64, n_heads=4, n_layers=2
        )
        self.assertEqual(config.d_model, 64)
        self.assertEqual(config.n_layers, 2)

    def test_vocab_size_override_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_from_tokenizer(FakeTokenizer(100), vocab_size=100)

    def test_bad_multiple_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_from_tokenizer(FakeTokenizer(100), pad_vocab_to_multiple_of=0)


class MakeTrainingBatchTests(unittest.TestCase):
    def test_shift_and_pad_masking(self) -> None:
        inputs, targets = make_training_batch(
            FakeTokenizer(), ["a", "b"], max_length=4, append_eos=False
        )
        # input_ids[:, :-1] and labels[:, 1:] of the fake's fixed return.
        self.assertTrue(torch.equal(inputs, torch.tensor([[5, 6, 7], [8, 9, 0]])))
        # Padded positions (attention_mask == 0) become IGNORE_INDEX, then shift.
        self.assertTrue(
            torch.equal(targets, torch.tensor([[6, 7, IGNORE_INDEX], [9, IGNORE_INDEX, IGNORE_INDEX]]))
        )

    def test_empty_texts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_training_batch(FakeTokenizer(), [], max_length=8)

    def test_too_short_max_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_training_batch(FakeTokenizer(), ["a"], max_length=1)


if __name__ == "__main__":
    unittest.main()
