"""
ias_capon.py – modular IAS‑Capon implementation
==============================================

五个核心部件
------------
1. **extract_window_data** —— 按 `Window` 切片波形并做去均值 / Hann taper。
2. **band_slice_fft**——GPU-FFT 后裁剪频带，返回实部 / 虚部及频带参数。
3. **CaponSolver**  ——  协方差 → 求逆 → Capon FK 的 GPU 包装器。
4. **WindowStacker**    —— 负责 `stack_size` 个窗口平均，记录第一段时间戳。
5. **MatWriter**    ——  增量 merge 写 `.mat` 文件，`key` 对应 FK，`key_t0` 对应首窗口时间戳。

顶层 `ias_capon()`：外层扫频段，内层扫窗口，调用上述组件完成计算与持久化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import logging
import numpy as np
from obspy import Stream  # type: ignore
from scipy.io import savemat, loadmat

from cfk import compute_fk, fft_cuda, compute_sm, matrix_invert  # type: ignore
from .windows import Window

###############################################################################
# logging & constant
###############################################################################
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # avoid warnings if user code lacks handlers

DL_FACTOR: float = 1.0  # diagonal‑loading multiplier


###############################################################################
# Step‑level helpers
###############################################################################

def _band_params(fmin: float, fmax: float, nsamp: int, dt: float) -> Tuple[int, int, float]:
    """Return (find, fave, fmid)."""
    fmid = 0.5 * (fmin + fmax)
    df   = 0.5 * (fmax - fmin)
    find = int(round(fmid * nsamp * dt))
    fave = int(round(df * nsamp * dt))
    if fave < 1:
        raise ValueError(f"Bandwidth too narrow: {fmin}-{fmax} Hz → fave=0")
    return find, fave, fmid


def extract_window_data(st: Stream, win: Window) -> np.ndarray:
    """Return (nr, nsamp) float64 array for one window."""
    nr = len(st)
    nsamp = win.end_idx - win.start_idx
    out = np.empty((nr, nsamp), dtype=np.float64)
    for i, tr in enumerate(st):
        seg = tr.data[win.start_idx:win.end_idx] - tr.data[win.start_idx:win.end_idx].mean()
        seg *= np.hanning(nsamp)
        out[i] = seg
    return out


def band_slice_fft(x: np.ndarray, fmin: float, fmax: float, dt: float) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """GPU FFT + band slice → real32, imag32, fave, find, fmid."""
    nr, nsamp = x.shape
    find, fave, fmid = _band_params(fmin, fmax, nsamp, dt)
    ft = fft_cuda(x, nsamp, nr)  # complex64 (nr, 2*(nsamp//2+1))
    band = ft[:, (find - fave):(find + fave + 1)]
    return band.real.astype(np.float32), band.imag.astype(np.float32), fave, find, fmid

###############################################################################
# GPU Capon solver
###############################################################################

class CaponSolver:
    def __init__(self, rx: np.ndarray, ry: np.ndarray, s_vals: Sequence[float], *, dl: float = DL_FACTOR, log: logging.Logger | None = None):
        if len(s_vals) < 2:
            raise ValueError("s_range must have ≥2 points")
        self.rx = rx.astype(np.float32)
        self.ry = ry.astype(np.float32)
        self.s_vals = np.asarray(s_vals, dtype=np.float32)
        if not np.allclose(np.diff(self.s_vals), self.s_vals[1]-self.s_vals[0]):
            raise ValueError("s_range must be evenly spaced")
        self.nk   = self.s_vals.size
        self.smin = float(self.s_vals[0])
        self.sinc = float(self.s_vals[1]-self.s_vals[0])
        self.dl   = float(dl)
        self.lg   = log or logging.getLogger(__name__)

    # ---------- private reusable core ---------------------------------
    def _solve_core(
        self,
        smr: np.ndarray,         # shape (2*fave+1, nr, nr)  float32
        smi: np.ndarray,
        *, nsamp: int, fave: int, find: int,
        dt: float, fmid: float
    ) -> np.ndarray:
        nr = smr.shape[-1]
        # ① 对角加载
        I = np.identity(nr, dtype=np.float32)
        for fi in range(2*fave+1):
            smr[fi] += I * smr[fi].trace() / nsamp * self.dl

        # ② batched inverse
        tx  = (smr + 1j*smi).astype(np.complex64).ravel()
        itx = matrix_invert(tx, nr, 2*fave+1).reshape(2*fave+1, nr, nr)
        ismr, ismi = itx.real.astype(np.float32).ravel(), itx.imag.astype(np.float32).ravel()

        # ③ Capon FK
        kinc = 2*np.pi*self.sinc*fmid
        kmin = 2*np.pi*self.smin*fmid
        tfk = np.zeros(self.nk*self.nk, dtype=np.float32)
        compute_fk(ismr, ismi, tfk, self.nk, self.rx, self.ry,
                np.float32(kinc), np.float32(kmin),
                nr, nsamp, np.float32(dt),
                np.float32(self.smin), np.float32(self.sinc),
                fave, find)
        return tfk.reshape(self.nk, self.nk)

    # ---------- 新接口：直接接平均协方差 -------------------------
    def solve_from_cov(
        self,
        smr: np.ndarray,
        smi: np.ndarray,
        *, nsamp: int, fave: int, find: int,
        dt: float, fmid: float
    ) -> np.ndarray:
        # 注意：smr/smi 已经是“ΣS_i / N”，此处不再平均
        return self._solve_core(smr.copy(), smi.copy(),
                                nsamp=nsamp, fave=fave, find=find,
                                dt=dt, fmid=fmid)

###############################################################################
# Covairance stacker
###############################################################################
class CovStacker:
    """累加协方差矩阵 (实、虚) 到够 stack_size。"""
    def __init__(self, stack_size: int, fave: int, nr: int):
        self.N = stack_size
        self.fave = fave
        self.nr = nr
        self.count = 0
        self.smr = np.zeros((2*fave+1, nr, nr), np.float32)
        self.smi = np.zeros_like(self.smr)
        self.first_iso: str | None = None
        self.group_id = 0

    def add(self, smr_i: np.ndarray, smi_i: np.ndarray, t0_iso: str):
        if self.count == 0:
            self.first_iso = t0_iso
        self.smr += smr_i
        self.smi += smi_i
        self.count += 1
        return self.count == self.N          # True → 可以 flush

    def flush(self):
        if self.count == 0:
            return None
        smr_avg = self.smr / self.count
        smi_avg = self.smi / self.count
        used = self.count              # ← 备份！
        t0   = self.first_iso or ""
        gid  = self.group_id       # ② 备份当前组号
        self.group_id += 1         # ③ 自增，为下一组做准备
        # reset 缓冲
        self.smr.fill(0.0); self.smi.fill(0.0)
        self.count = 0; self.first_iso = None
        return smr_avg, smi_avg, t0, used,gid
    

##########################################################################
#                             Mat Writer
##########################################################################
class MatWriter:
    """
    增量写入 .mat 文件，针对每张 FK 图同时保存：
        <key>             :   功率谱 (float32, nk×nk)
        <key>_t0          :   ISO-8601 首窗 UTC (char row-vector)
        <key>_step_sec    :   相邻窗中心间隔 (float32, 秒)
        <key>_nwin        :   累积窗口数 (int16)
    其余一次性元数据仍写入 meta 变量。
    """
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.meta_written = False
        path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    def write(
        self,
        key: str,
        arr: np.ndarray,
        t0_iso: str,
        *,
        step_sec: float,
        nwin: int,
    ) -> None:
        """
        Parameters
        ----------
        key       : 主数据变量名，例如 'fk_b00_g0003'
        arr       : (nk, nk) float32 Capon FK
        t0_iso    : 首窗口 UTC, ISO-8601 字符串
        step_sec  : 窗复制步长 (秒)。无重叠 = 窗长；50% overlap = 窗长/2
        nwin      : 本次累加的窗口个数
        """
        # ---------- 1. 读已有内容 ----------
        db: dict[str, object] = {}
        if self.path.exists():
            try:
                raw = loadmat(self.path)
                db.update({k: v for k, v in raw.items() if not k.startswith("__")})
            except Exception:
                logger.warning("Unreadable MAT %s – overwriting", self.path.name)

        # ---------- 2. 处理键名冲突 ----------
        base = key
        dup = 0
        while key in db:
            dup += 1
            key = f"{base}_dup{dup}"

        # ---------- 3. 写本批数据 ----------
        db[key]                = arr.astype(np.float32)
        db[f"{key}_t0"]        = np.bytes_(t0_iso)
        db[f"{key}_step_sec"]  = np.float32(step_sec)
        db[f"{key}_nwin"]      = np.int16(nwin)

        # ---------- 4. 首次写 meta ----------
        if not self.meta_written:
            db["meta"] = self.meta
            self.meta_written = True

        savemat(self.path, db, do_compression=True)


###############################################################################
# Public orchestrator
###############################################################################

def ias_capon(
    stream: Stream,
    rx: np.ndarray,
    ry: np.ndarray,
    windows: Sequence[Window],
    bands: List[Tuple[float, float]],
    s_range: Sequence[float],
    stack_size: int = 1,
    save_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    IAS-Capon beamforming.

    Parameters
    ----------
    stream : obspy.Stream
        原始波形；每条 trace 需带 SAC 头字段 `stla`/`stlo`。
    rx, ry : np.ndarray (float32, shape=(n_r,))
        台阵平面坐标 (degree)。x 指东、y 指北，常用
        ``rx, ry = array_xy(stream)`` 生成。
    windows : Sequence[Window]
        每个 `Window` 应含字段
        ┌ index:int ─ 0-based id
        ├ start_idx:int ─ 包含
        ├ end_idx:int   ─ 不含
        ├ start_time:datetime ─ UTC
        ├ end_time:datetime   ─ UTC
        └ duration:float ─ 秒
    bands : list[tuple[float, float]]
        频带列表 ``[(fmin, fmax), …]`` (Hz)，顺序保留，可重叠。
    s_range : Sequence[float]
        倾斜范围 (km/s)。
    stack_size : int, default 1
        叠加窗口数。
    save_path : pathlib.Path | None
        若给定，则保存输出文件。
    Returns
    -------
    dict[str, np.ndarray]
        关键结果数组（功率谱、波数图等），键名见实现。
    """
    
    lg = logger or logging.getLogger(__name__)
    
    if save_path is not None and save_path.exists():
        logging.getLogger(__name__).info("Remove existing MAT %s", save_path.name)
        save_path.unlink()          # ← 彻底删除旧文件
        
    if len(stream) == 0:
        raise ValueError("Empty ObsPy Stream")
    dt = float(stream[0].stats.delta)

    max_stack = stack_size if stack_size < len(windows) else len(windows)
    
    # ───────────────── ① 计算窗口步长（秒） ──────────────────
    win_len_sec = (windows[0].end_idx - windows[0].start_idx) * dt          # 一个窗长度
    if len(windows) > 1:                                                    # 若有重叠，用相邻首样本差
        step_sec = (windows[1].start_idx - windows[0].start_idx) * dt
    else:                                                                   # 只有 1 窗
        step_sec = win_len_sec
    
        
    solver = CaponSolver(rx, ry, s_range, log=lg)
    writer = MatWriter(save_path, meta=dict(nr=len(stream), dt=dt, slows=np.asarray(s_range, np.float32), bands=np.asarray(bands, np.float32))) if save_path else None

    fk_out: Dict[str, np.ndarray | str] = {}

    nr = len(stream)        # 台阵通道数，外层已经算过的话可删
    
    # ──────────────────── 频带循环 ─────────────────────
    for b_idx, (fmin, fmax) in enumerate(bands):
        lg.info("Band %d/%d (%.2f-%.2f Hz)", b_idx + 1, len(bands), fmin, fmax)

        # ---------- 先算出本 band 的 fave，用来分配缓存 ----------
        nsamp0 = windows[0].end_idx - windows[0].start_idx
        _, fave0, _ = _band_params(fmin, fmax, nsamp0, dt)

        # ★ 预分配：max_stack 个窗口 × (2 · nr) × (2 · fave+1)
        ft_buf  = np.empty((max_stack, 2 * nr, 2 * fave0 + 1), np.float32)
        sm_buf  = np.empty((2 * fave0 + 1) * 2 * nr * nr,       np.float32)

        count       = 0               # 已暂存的窗口数
        group_id    = 0               # 本频带第几个 FK 图
        first_iso   = ""              # 当前累计段的首窗时间戳

        # -------------------- 时间窗循环 --------------------
        for idx, win in enumerate(windows):
            # ① 取窗 → FFT → 裁频带
            x_win = extract_window_data(stream, win)
            rft, ift, fave, find, fmid = band_slice_fft(x_win, fmin, fmax, dt)

            # ★ ② 把实部/虚部按 kernel 需要的排布放进 ft_buf
            ft_buf[count, :nr,       :] = rft
            ft_buf[count,  nr:,      :] = ift

            if count == 0:            # 记录这一组首窗时间
                first_iso = win.start_time.isoformat()
            count += 1

            # ★ ③ 是否触发一次 GPU 协方差计算？
            flush_now = (count == max_stack) or (idx == len(windows) - 1)
            if not flush_now:
                continue

            # ---------- GPU 端一次性 Σ / nwin 平均 ----------
            sm_buf.fill(0.0)          # 清零旧结果
            compute_sm(ft_buf[:count].ravel(),     # ← nwin = count
                    count,                       # ← nwin 参数
                    nr, fave, sm_buf)
            sm = sm_buf.reshape(2 * fave + 1, 2 * nr, nr)
            smr, smi = sm[:, :nr, :], sm[:, nr:, :]

            # ---------- Capon FK ----------
            tfk = solver.solve_from_cov(
                smr, smi,
                nsamp = win.end_idx - win.start_idx,
                fave  = fave,
                find  = find,
                dt    = dt,
                fmid  = fmid,
            )

            key = f"fk_b{b_idx:02d}_g{group_id:04d}"
            fk_out[key]        = tfk
            fk_out[f"{key}_t0"] = first_iso
            if writer:
                writer.write(
                    key, tfk, first_iso,
                    step_sec = step_sec,
                    nwin     = count,          # 本次实际累加的窗口数
                )

            # ---------- 为下一组重置计数器 ----------
            group_id  += 1
            count      = 0
            first_iso  = ""

    lg.info("ias_capon completed: %d FK slices", sum(1 for k in fk_out if not k.endswith("_t0")))
    return fk_out
