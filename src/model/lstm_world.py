"""LSTM transition model: f_φ(s_t, a_t, h_t) → ŝ_{t+1}  (brief §7.3 eq. 14).

Theory note (see docs/THEORY.md §7.3). The workout-trainee environment is
formally a POMDP [2]: the trainee's evolution depends on history h_t, not
only on the current observation s_t. We recover a Markovian transition by
learning a recurrent world model [7] whose hidden state acts as a
sufficient statistic of h_t — turning the POMDP back into an MDP whose
transition kernel is this frozen network. REINFORCE (Phase 4) and A2C
(Phase 5) then roll out against f_φ as a learned simulator.

Architecture: input layer concatenates state (12-dim) with an action embedding
(``action_embed_dim``, default 8) per timestep. An LSTM (configurable
hidden_size + num_layers) consumes the resulting
(batch, seq, STATE_DIM + action_embed_dim) tensor; a linear head on the *last*
LSTM output produces the 12-dim next-state prediction.

Frozen during the RL phase (Phase 4/5) — `freeze()` sets requires_grad=False
on every param.
"""

from __future__ import annotations

import torch
from torch import nn

from src.env.state import ACTION_COUNT, STATE_DIM

_DEFAULT_ACTION_EMBED_DIM = 8  # backward-compat default; config-tunable
_EXPECTED_STATE_SEQ_NDIM = 3  # (batch, seq, STATE_DIM)
_EXPECTED_ACTION_SEQ_NDIM = 2  # (batch, seq)


class LSTMWorldModel(nn.Module):
    """Stateless-per-call LSTM world model. Hidden state is recomputed each forward."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        action_embed_dim: int = _DEFAULT_ACTION_EMBED_DIM,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.action_embed_dim = int(action_embed_dim)

        self.action_embedding = nn.Embedding(num_embeddings=ACTION_COUNT, embedding_dim=self.action_embed_dim)
        # PyTorch warns if dropout > 0 with num_layers == 1 — guard explicitly.
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=STATE_DIM + self.action_embed_dim,
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
        if not torch.isfinite(state_seq).all():
            raise ValueError(
                "state_seq contains non-finite values (NaN/Inf); inspect upstream env / data pipeline"
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
