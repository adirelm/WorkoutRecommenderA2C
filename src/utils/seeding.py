"""Global seeding for bit-reproducible runs.

Reproducibility is a precondition for any honest numeric comparison
between agents.  Without it, a "Double-Q beat REINFORCE by 3%" claim
is indistinguishable from RNG noise.  This module centralises every
source of stochasticity we touch:

* ``random``                 — Python stdlib RNG (used by stdlib shuffles).
* ``numpy.random``           — array sampling, dataset shuffles, noise.
* ``torch.manual_seed``      — CPU tensor init, dropout masks.
* ``torch.cuda.manual_seed_all`` — every visible CUDA device.
* ``PYTHONHASHSEED``         — dict/set ordering, critical for set-based
  state hashing and any cache key derived from ``hash()``.
* ``cudnn.deterministic``    — forces deterministic convolution algos.
* ``cudnn.benchmark = False``— disables the autotuner (its choice
  depends on wall-clock heuristics → non-reproducible).
* ``use_deterministic_algorithms(True, warn_only=True)`` — flips PyTorch
  into "raise/warn on any non-deterministic kernel".  ``warn_only`` is
  required because several ops we rely on (``scatter_add``, ``index_add``,
  some LSTM kernels on MPS) have no deterministic CUDA path.

Caveats the caller MUST know about:

1. ``scatter_add`` / ``index_add`` are non-deterministic on CUDA — runs
   that touch them will differ at the last few bits.  We default to CPU.
2. Apple MPS LSTM kernels are not bit-deterministic; tests that assert
   tensor equality must run on CPU.
3. Mixed precision (``torch.cuda.amp``) adds drift; we report fp32.
4. Multi-worker ``DataLoader`` shuffle order depends on worker scheduling
   — we use ``num_workers=0`` for the headline curves.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every RNG we touch.  Call once at program start.

    Args:
        seed: Non-negative integer.  ``0`` is supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
