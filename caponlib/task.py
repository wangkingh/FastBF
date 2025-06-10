from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List
import logging
import numpy as np

from .utils     import array_xy,load_stream
from .windows import Window

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
    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows list is empty")

        # 以第一个 Window 的 UTC 起始时间命名：YYYY-MM-DD-HH-MM.mat
        ts = self.windows[0].start_time.strftime("%Y-%m-%d-%H-%M")
        self.mat_out: Path = self.out_dir / f"{ts}.mat"
        self.mat_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self, *, logger: logging.Logger) -> None:
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
            windows    = self.windows,
            bands      = self.bands,
            s_range    = self.slows,
            stack_size = self.stack_size,
            save_path  = self.mat_out,     # ← 交给 ias_capon 链式写入
        )

        logger.debug("Finished: %s → %s", self.sac_txt.name, self.mat_out.name)
