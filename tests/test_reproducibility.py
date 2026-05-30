"""Reproducibility tests.

Two seeded runs of the same code path MUST produce bit-identical
tensors.  If these tests ever fail on CPU, every numeric claim in
the report becomes unverifiable.
"""

from __future__ import annotations

import numpy as np
import torch

from src.utils.seeding import set_global_seed


def _build_lstm() -> torch.nn.LSTM:
    """Small LSTM matching the state-encoder shape used in §7."""
    return torch.nn.LSTM(input_size=12, hidden_size=8, num_layers=1, batch_first=True)


def _forward_once(seed: int) -> torch.Tensor:
    """Seed, build a fresh LSTM, forward a deterministic input."""
    set_global_seed(seed)
    lstm = _build_lstm()
    # Fixed input — same seed must give same init weights AND same input.
    x = torch.arange(1 * 5 * 12, dtype=torch.float32).reshape(1, 5, 12) / 60.0
    out, _ = lstm(x)
    return out


def test_two_seeded_runs_produce_identical_tensors() -> None:
    """The headline guarantee: same seed → bit-identical LSTM output."""
    out_a = _forward_once(seed=42)
    out_b = _forward_once(seed=42)
    assert torch.equal(out_a, out_b), (
        "Two runs seeded with 42 produced different tensors; reproducibility plumbing is broken."
    )


def test_seed_zero_supported() -> None:
    """Edge case: seed=0 must be accepted (not treated as falsy)."""
    out_a = _forward_once(seed=0)
    out_b = _forward_once(seed=0)
    assert torch.equal(out_a, out_b)
    # And seed=0 must produce a different result than seed=1.
    out_c = _forward_once(seed=1)
    assert not torch.equal(out_a, out_c), (
        "seed=0 and seed=1 produced identical tensors; seeding has no effect."
    )


def test_seed_affects_numpy_too() -> None:
    """set_global_seed must cover numpy, not just torch."""
    set_global_seed(123)
    a = np.random.rand(10)
    set_global_seed(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)
