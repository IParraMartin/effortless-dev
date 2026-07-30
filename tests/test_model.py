"""Tests for the decoder-only Transformer.

Everything here runs on CPU in fp32 without a tokenizer or the network, so the
suite stays fast and hermetic. The load-bearing test is
:meth:`CacheTests.test_cached_generation_matches_full_forward`: it checks that
incremental decoding through the KV cache reproduces a single full-depth pass,
which is the property that makes ``generate`` correct.
"""

from __future__ import annotations

import unittest

import torch

from src.config import TransformerConfig
from src.model import ModelOutput, Transformer
from src.modules import KVCache


def tiny_config(**updates) -> TransformerConfig:
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.0,
    )
    values.update(updates)
    return TransformerConfig(**values)


class ConstructionTests(unittest.TestCase):
    def test_tied_head_shares_the_embedding_matrix(self) -> None:
        model = Transformer(tiny_config(tie_embeddings=True))
        self.assertIs(model.lm_head.weight, model.embed.weight)

    def test_untied_head_is_a_separate_matrix(self) -> None:
        model = Transformer(tiny_config(tie_embeddings=False))
        self.assertIsNot(model.lm_head.weight, model.embed.weight)
        tied = Transformer(tiny_config(tie_embeddings=True))
        self.assertGreater(
            model.num_parameters(), tied.num_parameters()
        )

    def test_grouped_query_shapes(self) -> None:
        # n_heads not divisible by n_kv_heads must be rejected at config time.
        with self.assertRaises(ValueError):
            tiny_config(n_heads=4, n_kv_heads=3)

    def test_ff_dim_inferred_as_multiple(self) -> None:
        config = tiny_config(d_model=256, ff_dim=None, ff_multiple_of=64)
        self.assertEqual(config.ff_dim % 64, 0)


class ForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = tiny_config()
        self.model = Transformer(self.config).eval()
        self.tokens = torch.randint(0, self.config.vocab_size, (2, 16))

    def test_logits_shape(self) -> None:
        out = self.model(self.tokens)
        self.assertEqual(
            tuple(out.logits.shape), (2, 16, self.config.vocab_size)
        )
        self.assertIsNone(out.loss)

    def test_loss_is_finite_and_backprops(self) -> None:
        model = Transformer(self.config)  # train mode
        out = model(self.tokens[:, :-1], targets=self.tokens[:, 1:])
        self.assertIsInstance(out, ModelOutput)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.any(g != 0) for g in grads))

    def test_ignore_index_excludes_positions(self) -> None:
        targets = self.tokens.clone()
        targets[:, :] = -100
        out = self.model(self.tokens, targets=targets)
        # All positions ignored -> cross_entropy over an empty set is NaN.
        self.assertTrue(torch.isnan(out.loss))

    def test_rejects_sequences_beyond_context(self) -> None:
        too_long = torch.zeros(1, self.config.max_seq_len + 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(too_long)


class GenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = tiny_config()
        self.model = Transformer(self.config).eval()

    def test_output_length(self) -> None:
        prompt = torch.randint(0, self.config.vocab_size, (2, 4))
        out = self.model.generate(prompt, max_new_tokens=8, temperature=0.0)
        self.assertEqual(tuple(out.shape), (2, 12))

    def test_greedy_is_deterministic(self) -> None:
        prompt = torch.randint(0, self.config.vocab_size, (2, 4))
        a = self.model.generate(prompt, max_new_tokens=8, temperature=0.0)
        b = self.model.generate(prompt, max_new_tokens=8, temperature=0.0)
        self.assertTrue(torch.equal(a, b))

    def test_eos_pads_after_stopping(self) -> None:
        prompt = torch.randint(0, self.config.vocab_size, (1, 3))
        eos = 0
        out = self.model.generate(
            prompt, max_new_tokens=10, temperature=0.0, eos_token_id=eos
        )
        tail = out[0, 3:].tolist()
        if eos in tail:
            first = tail.index(eos)
            self.assertTrue(all(t == eos for t in tail[first:]))


class CacheTests(unittest.TestCase):
    def test_cached_generation_matches_full_forward(self) -> None:
        torch.manual_seed(0)
        config = tiny_config()
        model = Transformer(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 12))

        with torch.no_grad():
            full = model(tokens).logits

            cache = KVCache(config.n_layers)
            step_logits = [model(tokens[:, :1], cache=cache).logits[:, -1]]
            for pos in range(1, tokens.size(1)):
                step_logits.append(
                    model(tokens[:, pos:pos + 1], cache=cache).logits[:, -1]
                )
            incremental = torch.stack(step_logits, dim=1)

        self.assertTrue(torch.allclose(full, incremental, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
