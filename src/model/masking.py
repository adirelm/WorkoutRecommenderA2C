"""Shared invalid-action masking helper (Huang & Ontañón 2022).

ADR-004 mandates that illegal actions receive probability 0 under the
policy. We implement this by setting their *pre-softmax logits* to
``-inf`` so that ``softmax`` (inside ``torch.distributions.Categorical``
or anywhere else downstream) drives those positions to 0 and the
policy-gradient term ``∇log π(a|s)`` ignores them entirely. See
"A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"
(Huang & Ontañón 2022): https://arxiv.org/abs/2006.14171.

Two duplicates of this logic used to live in :mod:`src.model.policy_net`
and :mod:`src.model.actor_critic`, with a third near-duplicate inside
:meth:`src.services.base_trainer.BaseTrainer.sample_action`. They are now
all routed through :func:`apply_mask` (DRY — CLAUDE.md §5).
"""

from __future__ import annotations

import torch


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of illegal actions to ``-inf`` BEFORE softmax.

    Args:
        logits: shape ``(..., n_actions)``. Raw network output.
        mask:   shape ``(..., n_actions)``. bool-castable.
                ``True`` = legal, ``False`` = illegal.

    Returns:
        Same shape as ``logits`` with illegal positions set to ``-inf``
        so softmax assigns them probability 0 and the policy gradient
        ignores them.

    Notes:
        Callers retain responsibility for shape/None validation — this
        helper is intentionally minimal so it can be reused from
        networks, trainers, and test harnesses without bringing along
        domain-specific assertions (e.g. ``ACTION_COUNT``).
    """
    mask_bool = mask if mask.dtype == torch.bool else mask.to(dtype=torch.bool)
    return logits.masked_fill(~mask_bool, float("-inf"))
