from __future__ import annotations
"""Split multi‑channel time‑series data into overlapping windows **without
embedding absolute timestamps**.

Re‑design highlights
--------------------
1. **`Window` now stores only sample indices and duration.**  Absolute UTC
   timestamps are computed later, when the task knows the true record start
   time.  This keeps the object lightweight and reusable.
2. `build_windows()` is therefore _purely index‑based_; it no longer needs
   the record's start time.  The helper `attach_times()` is provided to turn
   a list of relative windows into absolute‑time windows when required
   (e.g. right before calling the beamforming kernel).
3. Input validation remains identical to the previous implementation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np

###############################################################################
# Data classes
###############################################################################

@dataclass
class Window:
    """Lightweight window description — index space only."""

    index: int      # 0‑based window id
    start_idx: int  # inclusive sample index
    end_idx: int    # exclusive sample index

    @property
    def duration(self) -> int:
        """Window length in samples."""
        return self.end_idx - self.start_idx

    def __repr__(self) -> str:  # noqa: D401
        return f"Window(idx={self.index}, start={self.start_idx}, end={self.end_idx}, len={self.duration})"


@dataclass
class WindowAbs(Window):
    """Window with absolute UTC timestamps attached (for downstream algorithms)."""

    start_time: datetime  # UTC timestamp at start_idx
    end_time: datetime    # UTC timestamp at end_idx

    def __post_init__(self):
        # keep parent invariants
        if self.end_idx <= self.start_idx:
            raise ValueError("end_idx must be greater than start_idx")

    def __repr__(self) -> str:  # noqa: D401
        t0 = self.start_time.isoformat(timespec="milliseconds") + "Z"
        t1 = self.end_time.isoformat(timespec="milliseconds") + "Z"
        return (
            f"WindowAbs(idx={self.index}, start={t0}, end={t1}, len={self.duration})"
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
# Main public helpers
###############################################################################

def build_windows(
    *,
    dt: float,                      # sampling interval in seconds
    npts: int,                      # total number of samples per channel
    winlen: float,                  # window length in seconds (≥ dt)
    hop_ratio: float = 0.5,         # hop / window length ratio (0–1]
    skip_win_idx: Sequence[int] | None = None,
) -> List[Window]:
    """Return **relative** windows (indices only).

    Parameters
    ----------
    dt  : sampling interval (s)
    npts: number of samples
    winlen: desired window length (s)
    hop_ratio: fraction of window length to advance
    skip_win_idx: optional iterable of indices to drop
    """

    # ---------------- Sanity checks ----------------
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

    # ---------------- Assembly ----------------
    return [Window(index=w, start_idx=s, end_idx=e) for w, (s, e) in enumerate(idx_pairs)]


def attach_times(
    windows: List[Window],
    *,
    start_epoch: datetime,
    dt: float,
) -> List[WindowAbs]:
    """Convert *relative* windows to *absolute‑time* windows.

    This function leaves the original list untouched and returns a new list of
    :class:`WindowAbs` instances.
    """

    out: List[WindowAbs] = []
    for w in windows:
        t0 = start_epoch + timedelta(seconds=w.start_idx * dt)
        t1 = start_epoch + timedelta(seconds=w.end_idx * dt)
        out.append(
            WindowAbs(
                index=w.index,
                start_idx=w.start_idx,
                end_idx=w.end_idx,
                start_time=t0,
                end_time=t1,
            )
        )
    return out
