"""LSTM transition model: f_φ(s_t, a_t, h_t) → ŝ_{t+1}  (brief §7.3 eq. 14).

Architecture: input layer concatenates state (12-dim) with an action embedding
(8-dim) per timestep. An LSTM (configurable hidden_size + num_layers) consumes
the resulting (batch, seq, STATE_DIM + action_embed_dim) tensor; a linear head
on the *last* LSTM output produces the 12-dim next-state prediction.

Frozen during the RL phase (Phase 4/5) — `freeze()` sets requires_grad=False
on every param.
"""

from __future__ import annotations

import torch
from torch import nn

from src.env.state import ACTION_COUNT, STATE_DIM

_ACTION_EMBED_DIM = 8
_EXPECTED_STATE_SEQ_NDIM = 3  # (batch, seq, STATE_DIM)
_EXPECTED_ACTION_SEQ_NDIM = 2  # (batch, seq)


class LSTMWorldModel(nn.Module):
    """Stateless-per-call LSTM world model. Hidden state is recomputed each forward."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.action_embedding = nn.Embedding(num_embeddings=ACTION_COUNT, embedding_dim=_ACTION_EMBED_DIM)
        # PyTorch warns if dropout > 0 with num_layers == 1 — guard explicitly.
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=STATE_DIM + _ACTION_EMBED_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, STATE_DIM)

    def forward(self, state_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """state_seq: (batch, seq, STATE_DIM); action_seq: (batch, seq) int64.

        Returns predicted next_state (batch, STATE_DIM) — float32.
        """
        if state_seq.dim() != _EXPECTED_STATE_SEQ_NDIM or state_seq.size(-1) != STATE_DIM:
            raise ValueError(f"state_seq must be (batch, seq, {STATE_DIM}); got {tuple(state_seq.shape)}")
        if action_seq.dim() != _EXPECTED_ACTION_SEQ_NDIM or action_seq.shape != state_seq.shape[:2]:
            raise ValueError(
                f"action_seq must be (batch, seq) matching state_seq; got {tuple(action_seq.shape)}"
            )

        action_embed = self.action_embedding(action_seq)  # (batch, seq, embed_dim)
        x = torch.cat([state_seq, action_embed], dim=-1)  # (batch, seq, STATE_DIM + embed_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden_size)
        last = lstm_out[:, -1, :]  # (batch, hidden_size) — last timestep
        return self.head(last)  # (batch, STATE_DIM)

    def freeze(self) -> None:
        """Set requires_grad=False on every param (Phase 4+ RL phase)."""
        for p in self.parameters():
            p.requires_grad = False

    def is_frozen(self) -> bool:
        """True iff every parameter has requires_grad=False."""
        return all(not p.requires_grad for p in self.parameters())

    @property
    def num_parameters(self) -> int:
        """Total parameter count (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
