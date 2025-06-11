"""
beamrunner – 驱动层：解析配置、准备数据、并行调度
"""

# ── 配置 & SAC-list ─────────────────────────────────────────
from .config_parser  import parse_and_validate, print_array_summary
from .sac_refactor   import organise_sac_files, write_sac_lists_by_time

# ── 数据准备 (位于本包 io.py / geometry.py) ────────────────
from .utils       import array_xy,load_stream

# ── 任务 & 调度 ────────────────────────────────────────────
from .bftask          import BfTask              # 先导入 Task，防循环依赖
from .scheduler      import run_parallel
from .windows import Window, build_windows

from .ias_capon import ias_capon
__all__ = [
    # 配置
    "parse_and_validate",
    "print_array_summary",
    # SAC-list
    "organise_sac_files",
    "write_sac_lists_by_time",
    # 数据
    "load_stream",
    "array_xy",
    # 任务 / 调度
    "BfTask",
    "run_parallel",
    "Window",
    "windows",
    "ias_capon",
]
