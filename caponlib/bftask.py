from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List
import logging
import numpy as np

from .utils     import array_xy,load_stream
from .windows import attach_times, Window

ComputeFn = Callable[..., Dict[str, Any]]

@dataclass
class BfTask:
    sac_txt   : Path
    windows   : List[Window]
    bands     : List[tuple[float, float]]
    slows     : List[float]
    stack_size: int
    out_dir   : Path
    compute_fn: ComputeFn | None = None     # None → dry-run

    # ------------------------------------------------------------------
    def __post_init__(self):
        # 用 sac-list 文件名保证唯一
        tag = self.sac_txt.stem          # 2025-03-28-13-00
        self.mat_out = self.out_dir / f"{tag}.mat"
        self.mat_out.parent.mkdir(parents=True, exist_ok=True)
        
    # ------------------------------------------------------------------
    def run(self, *, logger: logging.Logger) -> None:
        
        stream = load_stream(self.sac_txt)
        tr0    = stream[0]
        start_epoch = tr0.stats.starttime.datetime
        dt          = float(tr0.stats.delta)

        # ---------- 相对 → 绝对窗口 -----------------------------------
        windows_abs = attach_times(
            self.windows,
            start_epoch=start_epoch,
            dt=dt,
        )
        
        if self.compute_fn is None:   # dry-run
            logger.info(
                "[Dry-run] %s | windows=%d | bands=%s | max_s=%.2f | "
                "stack=%d → %s",
                self.sac_txt.name, len(self.windows), self.bands,
                np.max(self.slows), self.stack_size, self.mat_out.name
            )
            return

        # ---- 实际计算 ---------------------------------------------------
        stream = load_stream(self.sac_txt)
        rx, ry = array_xy(stream)  # ref station = trace 0

        self.compute_fn(
            stream     = stream,
            rx         = rx,
            ry         = ry,
            windows    = windows_abs,
            bands      = self.bands,
            s_range    = self.slows,
            stack_size = self.stack_size,
            save_path  = self.mat_out,     # ← 交给 ias_capon 链式写入
        )

        logger.debug("Finished: %s → %s", self.sac_txt.name, self.mat_out.name)
