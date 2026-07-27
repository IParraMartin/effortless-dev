"""Hugging Face tokenizer integration for the from-scratch Transformer.

The model itself is tokenizer-agnostic: it consumes integer token ids and knows
nothing about how text was segmented. This module supplies the three pieces of
glue that connect a pretrained Hugging Face tokenizer to it:

    * :func:`load_tokenizer` fetches the tokenizer and guarantees it has the
      padding and end-of-sequence tokens the training loop relies on.
    * :func:`config_from_tokenizer` derives a :class:`TransformerConfig` whose
      embedding table matches the tokenizer's vocabulary.
    * :func:`make_training_batch` turns raw strings into the shifted
      ``(input_ids, targets)`` pair expected by :meth:`Transformer.forward`.
"""

from __future__ import annotations
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import TransformerConfig

#: Ignore index understood by ``F.cross_entropy`` in :meth:`Transformer.forward`.
IGNORE_INDEX = -100


def load_tokenizer(
    name_or_path: str = "gpt2",
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    """Loads a pretrained tokenizer and normalizes its special tokens.

    Many causal language-model tokenizers ship without a padding token because
    their original training used packed sequences rather than padded batches.
    When one is missing it is aliased to the end-of-sequence token, which is
    harmless here since padded positions are excluded from the loss anyway.

    Args:
        name_or_path: Hub repository id or local directory, for example
            ``"gpt2"``, ``"Qwen/Qwen3-8B"``, or ``"./my-tokenizer"``.
        use_fast: Whether to prefer the Rust-backed fast tokenizer. Fast
            tokenizers are substantially quicker on large corpora and are
            required for offset mapping.

    Returns:
        The loaded tokenizer, guaranteed to expose non-``None``
        ``pad_token_id`` and ``eos_token_id``.

    Raises:
        ValueError: If the tokenizer defines neither a padding nor an
            end-of-sequence token, leaving nothing to pad batches with.
    """
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=use_fast)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer {name_or_path!r} defines neither a pad token nor "
                "an eos token; one is required to build padded batches."
            )
        tokenizer.pad_token = tokenizer.eos_token

    # Right padding keeps real tokens contiguous at the start of each row,
    # which is what the causal-only attention in this model assumes.
    tokenizer.padding_side = "right"
    return tokenizer


def config_from_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    *,
    pad_vocab_to_multiple_of: int = 64,
    **overrides: object,
) -> TransformerConfig:
    """Builds a model config whose vocabulary matches a tokenizer.

    The embedding table is sized to at least ``len(tokenizer)``, which counts
    any tokens added after pretraining, and is then rounded up to a hardware
    friendly multiple. The surplus rows are never emitted by the tokenizer and
    simply learn to be improbable, but the aligned matrix shape measurably
    speeds up the output projection on tensor cores.

    Args:
        tokenizer: Tokenizer whose vocabulary the model must cover.
        pad_vocab_to_multiple_of: Alignment for the final vocabulary size. Pass
            ``1`` to use the exact tokenizer size.
        **overrides: Any other :class:`TransformerConfig` field, such as
            ``d_model`` or ``n_layers``.

    Returns:
        A configuration with ``vocab_size`` set from the tokenizer and every
        other field taken from ``overrides`` or its default.

    Raises:
        ValueError: If ``pad_vocab_to_multiple_of`` is not positive, or if
            ``vocab_size`` is passed in ``overrides``, since it would conflict
            with the size implied by the tokenizer.
    """
    if pad_vocab_to_multiple_of < 1:
        raise ValueError(
            "pad_vocab_to_multiple_of must be positive, got "
            f"{pad_vocab_to_multiple_of}."
        )
    if "vocab_size" in overrides:
        raise ValueError(
            "vocab_size is derived from the tokenizer and cannot be overridden."
        )

    size = len(tokenizer)
    multiple = pad_vocab_to_multiple_of
    vocab_size = -(-size // multiple) * multiple

    return TransformerConfig(vocab_size=vocab_size, **overrides)


def make_training_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    append_eos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes raw text into a next-token-prediction batch.

    Each string is tokenized, truncated, and right-padded to a common length,
    then split into the input and target halves of the causal objective: the
    model sees positions ``0..n-1`` and predicts positions ``1..n``. Padding
    positions in the targets are replaced with :data:`IGNORE_INDEX` so they
    contribute nothing to the loss.

    No attention mask is needed even though the batch is padded. Padding sits
    at the end of every row, and the model's attention is strictly causal, so a
    real token can never attend to a pad that follows it.

    Args:
        tokenizer: Tokenizer produced by :func:`load_tokenizer`.
        texts: Raw strings to encode. Must be non-empty.
        max_length: Maximum number of tokens per example before the shift.
            Sequences longer than this are truncated; the returned tensors have
            a sequence dimension of at most ``max_length - 1``.
        append_eos: Whether to terminate each example with the end-of-sequence
            token, teaching the model where documents stop.

    Returns:
        A tuple ``(input_ids, targets)``, both shaped
        ``(len(texts), seq_len)`` and ready to pass straight to
        :meth:`Transformer.forward`.

    Raises:
        ValueError: If ``texts`` is empty or ``max_length`` is below two, which
            would leave nothing to predict after the shift.
    """
    if not texts:
        raise ValueError("texts must contain at least one string.")
    if max_length < 2:
        raise ValueError(
            f"max_length must be at least 2 to form a shifted pair, got "
            f"{max_length}."
        )

    if append_eos:
        texts = [text + tokenizer.eos_token for text in texts]

    encoded = tokenizer(
        list(texts),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = token_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
    return token_ids[:, :-1], labels[:, 1:]


if __name__ == "__main__":
    from src.model import Transformer

    torch.manual_seed(0)

    tokenizer = load_tokenizer("gpt2")
    print(f"tokenizer: {type(tokenizer).__name__}  vocab: {len(tokenizer)}")

    config = config_from_tokenizer(
        tokenizer, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=256
    )
    model = Transformer(config)
    print(f"vocab_size: {config.vocab_size}  parameters: {model.num_parameters() / 1e6:.2f}M")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Rotary embeddings encode position as a rotation.",
    ]
    input_ids, targets = make_training_batch(tokenizer, texts, max_length=32)
    print(f"input_ids: {tuple(input_ids.shape)}  ignored targets: {int((targets == IGNORE_INDEX).sum())}")

    out = model(input_ids, targets=targets)
    print(f"logits: {tuple(out.logits.shape)}  loss: {out.loss.item():.4f}")

    prompt = tokenizer("The quick brown", return_tensors="pt").input_ids
    result = model.generate(
        prompt,
        max_new_tokens=12,
        temperature=0.8,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"sample: {tokenizer.decode(result.sequences[0])!r}")
    print(f"mean exit layer: {result.mean_exit_layer:.2f} of {config.n_layers}")
