"""
台阵几何计算 – 把经纬度转换成平面 x, y (单位：度)，
依赖 obspy 的大圆工具，避免自己写球面三角。
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from obspy import Stream, read
from obspy.geodetics import gps2dist_azimuth

# ------------------------------------------------------------------
# logging
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())        # 避免“未配置 handler”警告

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
DEG2KM = 1.0 / 1000.0                         # m → km
EARTH_RADIUS_DEG = 180.0 / np.pi / 6371.0     # km → degree


def _one_pair(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> Tuple[float, float]:
    """返回 (dx, dy) – 参考台 (0) 指向 (1) 的投影坐标（单位：度）"""
    dist_m, az, _ = gps2dist_azimuth(lat0, lon0, lat1, lon1)
    dist_deg = dist_m * DEG2KM * EARTH_RADIUS_DEG          # km → degree
    # 方位角 az=0 指北，变换到数学坐标系：x=E, y=N
    dx = dist_deg * np.sin(np.deg2rad(az))
    dy = dist_deg * np.cos(np.deg2rad(az))
    return dx, dy


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def array_xy(st: Stream, ref_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    st       : obspy.Stream (要求每 trace 有 SAC stla/stlo)
    ref_idx  : 参考台索引，默认第 0 条 trace

    Returns
    -------
    rx, ry   : 长度 = nr 的 1-D ndarray (degree)
    """
    nr = len(st)
    lat0 = float(st[ref_idx].stats.sac.stla)
    lon0 = float(st[ref_idx].stats.sac.stlo)

    rx = np.empty(nr, dtype=np.float32)
    ry = np.empty(nr, dtype=np.float32)

    for i, tr in enumerate(st):
        lat1 = float(tr.stats.sac.stla)
        lon1 = float(tr.stats.sac.stlo)
        rx[i], ry[i] = _one_pair(lat0, lon0, lat1, lon1)

    return rx, ry


def load_stream(txt_path: Path) -> Stream:
    """Read a sac-list txt where each line is an absolute SAC path."""
    with txt_path.open() as fh:
        files: List[str] = [ln.strip() for ln in fh if ln.strip()]

    if len(files) < 2:                       # too few stations → early abort
        raise RuntimeError(f"{txt_path} contains <2 SAC files")

    st = Stream()
    for fp in files:
        st += read(fp, format="SAC")

    logger.debug("Loaded %d traces from %s", len(st), txt_path.name)
    return st
