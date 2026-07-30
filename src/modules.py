"""Building blocks shared by the Transformer.

The pieces that are independent of how the decoder stack is assembled: a
per-layer key/value cache for incremental decoding, RMS normalization, and
rotary position embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KVCache:
    """Growable per-layer cache of attention keys and values.

    During autoregressive decoding every generated token attends to all of its
    predecessors. Recomputing their keys and values at each step is quadratic
    work, so they are stored here and extended one step at a time. Every layer
    holds the same number of positions, so :meth:`__len__` can read layer zero
    alone.

    Args:
        n_layers: Number of decoder layers in the model.

    Raises:
        ValueError: If ``n_layers`` is not positive.
    """

    def __init__(self, n_layers: int) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def __len__(self) -> int:
        """Number of tokens currently cached."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].size(-2)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends a new key/value chunk and returns the full history.

        Args:
            layer_idx: Index of the layer whose cache is being updated.
            key: Keys for the incoming tokens, shaped
                ``(batch, n_kv_heads, new_len, head_dim)``.
            value: Values for the incoming tokens, same shape as ``key``.

        Returns:
            A tuple ``(keys, values)`` covering every token seen so far by this
            layer, each shaped ``(batch, n_kv_heads, total_len, head_dim)``.
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat((self.keys[layer_idx], key), dim=-2)
            self.values[layer_idx] = torch.cat(
                (self.values[layer_idx], value), dim=-2
            )
        return self.keys[layer_idx], self.values[layer_idx]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    RMSNorm rescales activations by their root mean square without subtracting
    the mean, which removes the centering term of LayerNorm at essentially no
    quality cost while being cheaper to compute.

    Args:
        dim: Size of the feature dimension being normalized.
        eps: Epsilon added to the mean square before the reciprocal square
            root, guarding against division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes and rescales the final dimension of ``x``.

        Args:
            x: Input tensor shaped ``(..., dim)``.

        Returns:
            A tensor with the same shape and dtype as ``x``.

        Note:
            The statistics are accumulated in float32 even under autocast, then
            cast back, so that low-precision training stays stable.
        """
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding tables.

    RoPE encodes absolute positions as a rotation applied to pairs of channels
    in the query and key vectors. Because the dot product of two rotated
    vectors depends only on the difference of their angles, attention scores
    become a function of relative distance.

    Args:
        head_dim: Per-head feature size. Must be even.
        max_seq_len: Number of positions to precompute rotations for.
        theta: Base period controlling how fast the rotation frequencies decay
            across channel pairs.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}.")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = torch.outer(positions, inv_freq)

        # Duplicated so the tables line up with the split-half rotation below.
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the two halves of the feature dimension into each other.

        Args:
            x: Tensor shaped ``(..., head_dim)``.

        Returns:
            The tensor ``[-x2, x1]`` where ``x1`` and ``x2`` are the first and
            second halves of the final dimension of ``x``.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def rotate(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotates a single query or key tensor.

        Args:
            x: Tensor shaped ``(batch, heads, seq_len, head_dim)``.
            offset: Absolute position of the first token in ``x``. During
                cached decoding this is the number of tokens already processed.

        Returns:
            The rotated tensor, with the shape and dtype of ``x``.

        Raises:
            ValueError: If ``offset + seq_len`` exceeds the precomputed table.
        """
        seq_len = x.size(-2)
        end = offset + seq_len
        if end > self.cos.size(0):
            raise ValueError(
                f"Position {end} exceeds the precomputed RoPE table of length "
                f"{self.cos.size(0)}."
            )

        cos = self.cos[offset:end].to(x.dtype)
        sin = self.sin[offset:end].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the rotation to a batch of queries and keys.

        Args:
            q: Queries shaped ``(batch, n_heads, seq_len, head_dim)``.
            k: Keys shaped ``(batch, n_kv_heads, seq_len, head_dim)``.
            offset: Absolute position of the first token in ``q`` and ``k``.

        Returns:
            A tuple ``(q_rotated, k_rotated)`` with the input shapes and dtypes.
        """
        return self.rotate(q, offset), self.rotate(k, offset)
