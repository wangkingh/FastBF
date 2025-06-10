from __future__ import annotations

"""Split multi‑channel time‑series data into overlapping windows and return
rich timestamp metadata **without allocating the data blocks**.

Key changes vs. previous revision
---------------------------------
1. **`winlen` (seconds)** replaces the old `n_samples` argument.
   The function converts it to an integer number of samples via `dt`.
2. Explicit sanity check for `hop_ratio` → must satisfy `0 < hop_ratio ≤ 1`.
3. Additional guardrails: `winlen` must be positive and not longer than the
   full record length.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np

###############################################################################
# Data class
###############################################################################

@dataclass
class Window:
    """Metadata of a single window."""

    index: int                 # 0‑based window id
    start_idx: int             # inclusive sample index
    end_idx: int               # exclusive sample index
    start_time: datetime       # UTC timestamp at start_idx
    end_time: datetime         # UTC timestamp at end_idx
    duration: float            # window length in seconds

    def __repr__(self) -> str:  # noqa: D401
        return (
            "Window("  # noqa: E501
            f"idx={self.index}, "
            f"start={self.start_time.isoformat(timespec='milliseconds')}Z, "
            f"end={self.end_time.isoformat(timespec='milliseconds')}Z, "
            f"len={self.duration:.3f}s)"
        )

###############################################################################
# Helpers
###############################################################################

def _sliding_indices(npts: int, winlen_samples: int, hop: int) -> List[Tuple[int, int]]:
    """Generate `(start, end)` index pairs for sliding windows."""
    pairs: List[Tuple[int, int]] = []
    i = 0
    while i + winlen_samples <= npts:
        pairs.append((i, i + winlen_samples))
        i += hop
    return pairs

###############################################################################
# Main function
###############################################################################

def build_windows(
    *,
    start_epoch: datetime,          # UTC timestamp of sample index 0
    dt: float,                      # sampling interval in seconds
    npts: int,                      # total number of samples per channel
    winlen: float,                  # window length in seconds (≥ dt)
    hop_ratio: float = 0.5,         # hop / window length ratio (0–1]
    skip_win_idx: Sequence[int] | None = None,
) -> List[Window]:
    """Generate a list of `Window` metadata for a sliding‑window traversal.

    Parameters
    ----------
    start_epoch
        UTC `datetime` corresponding to the very first sample (index 0).
    dt
        Sampling interval in seconds.
    npts
        Total number of samples per channel.
    winlen
        Desired window length **in seconds**. Internally converted to an
        integer number of samples = ``round(winlen / dt)``.
    hop_ratio
        Fraction of the window length to advance per step.
        Must satisfy ``0 < hop_ratio ≤ 1``. Default is 0.5 (50 % overlap).
    skip_win_idx
        Optional iterable of window indices (0‑based) to skip.

    Returns
    -------
    List[Window]
        Metadata for every retained window.
    """

    # ---------------- Sanity checks ----------------
    if not isinstance(start_epoch, datetime):
        raise TypeError("start_epoch must be a datetime object in UTC")

    if winlen <= 0:
        raise ValueError("winlen must be positive")

    n_samples = int(round(winlen / dt))
    if n_samples <= 0:
        raise ValueError("winlen too short given dt — rounded to 0 samples")
    if n_samples > npts:
        raise ValueError("winlen exceeds total record length")

    if not (0 < hop_ratio <= 1):
        raise ValueError("hop_ratio must satisfy 0 < hop_ratio ≤ 1")
    hop = int(round(n_samples * hop_ratio))
    if hop == 0:
        raise ValueError("hop_ratio too small: resulting hop = 0 samples")

    if skip_win_idx is None:
        skip_win_idx = []
    skip_set = set(skip_win_idx)
    if any(i < 0 for i in skip_set):
        raise ValueError("skip_win_idx may not contain negative indices")

    # ---------------- Index generation ----------------
    idx_pairs = _sliding_indices(npts, n_samples, hop)
    idx_pairs = [p for k, p in enumerate(idx_pairs) if k not in skip_set]

    # ---------------- Metadata assembly ----------------
    meta: List[Window] = []
    for w, (s_idx, e_idx) in enumerate(idx_pairs):
        start_time = start_epoch + timedelta(seconds=s_idx * dt)
        end_time = start_epoch + timedelta(seconds=e_idx * dt)
        meta.append(
            Window(
                index=w,
                start_idx=s_idx,
                end_idx=e_idx,
                start_time=start_time,
                end_time=end_time,
                duration=(e_idx - s_idx) * dt,
            )
        )

    return meta
