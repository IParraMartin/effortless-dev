# A decoder-only Transformer

A small, readable implementation of a modern decoder-only language model, with
the training loop and data pipeline needed to pretrain it from scratch on a
Hugging Face corpus.

## The model

`src/model.py` is a standard modern LLM stack:

- **Pre-norm blocks** with [RMSNorm](https://arxiv.org/abs/1910.07467) —
  normalization on each sublayer's input, leaving a clean residual identity path.
- **Rotary position embeddings** (RoPE), so attention depends on relative
  distance and no learned position table is stored.
- **Grouped-query attention**, run through
  `F.scaled_dot_product_attention` (a fused FlashAttention kernel where one is
  available). `n_kv_heads == n_heads` is multi-head, `1` is multi-query, and
  anything between is GQA.
- **SwiGLU** feed-forward blocks.
- **Tied embeddings**: the output projection reuses the input embedding matrix.
- **Scaled initialization**: residual projections are shrunk by
  `1/sqrt(2·n_layers)` so activation variance stays ~unit at any depth.

Training is ordinary next-token cross-entropy; generation is autoregressive with
a key/value cache. The model consumes integer token ids and is otherwise
tokenizer-agnostic.

## Layout

```
src/
  config.py     TransformerConfig, TrainConfig, and command-line parsing
  modules.py    KVCache, RMSNorm, RotaryEmbedding
  model.py      Attention, SwiGLU, DecoderBlock, Transformer (forward + generate)
  tokenizer.py  Hugging Face tokenizer glue and next-token batch construction
training/
  data.py       corpus tokenization, packed memmaps, resumable block sampling
  train.py      the training loop: DDP, cosine schedule, checkpoint + resume, W&B
  distributed.py single- and multi-process helpers
utils/
  tracking.py   optional Weights & Biases logging
jobs/           Slurm scripts for the Savio cluster (see jobs/README.md)
tests/          CPU-only unit tests
```

## Commands

Run everything from the repository root. The examples use `uv run`; drop it if
you have the environment on your `PATH`.

```bash
# Tests (CPU, a few seconds)
uv run python -m unittest discover -s tests -t .

# 1. Tokenize a corpus into data/{train,val}.bin
uv run python -m training.data \
    --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
    --tokenizer_name gpt2 --streaming=true

# 2. Train (single process)
uv run python -m training.train --n_layers=12 --d_model=768 --max_steps=20000

# ...or on multiple GPUs on one node
uv run torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 \
    -m training.train --n_layers=12 --d_model=768 --max_steps=20000
```

Architecture and run settings share one command line — `python -m training.train
--help` lists every flag. `vocab_size` and `max_seq_len` are derived from the
tokenizer and `--seq_len`, so they are not passed directly.

A quick end-to-end sanity check builds a tiny model, trains one batch, and
samples from it:

```bash
uv run python -m src.tokenizer
```

## Data

Text is tokenized once, ahead of training, into a flat array of ids stored as a
memory map (`uint16`, or `uint32` past a 65,536-token vocabulary — recorded in a
`.meta.json` sidecar). Documents are concatenated with an end-of-sequence token
between them ("packing"), so nothing is padded and the strictly causal attention
needs no mask. When a corpus ships only a training split, a small held-out set is
carved from it without overlap.

## Reproducibility

`training.data.StatelessBlockSampler` makes block order a pure function of the
seed and the global position, so there is no data cursor to serialize:
[resuming](training/train.py) reconstructs the sampler from the completed-update
count and continues on the next unseen batch. Checkpoints also carry the
optimizer moments, the gradient scaler, and every RNG stream, so a run restarted
from `--resume_from` continues rather than restarting.
