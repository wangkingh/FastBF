# bf_fk_peaks_pdf.py
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Any


def loadmat_simplified(path: str | os.PathLike):
    try:
        return loadmat(path, simplify_cells=True)  # SciPy ≥1.8
    except TypeError:
        return loadmat(path)


def chunk_list(lst, chunk_size):
    """
    将一个列表按给定大小 chunk_size 分割，生成器方式返回。
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def list_fk_keys(mat_dict):
    """列出所有功率谱主键，如 ['fk_b00_g0000', ...]"""
    for k in mat_dict.keys():
        print(k)
    return [
        k
        for k in mat_dict.keys()
        if k.startswith("fk_") and not k.endswith(("_t0", "_step_sec", "_nwin"))
    ]


def read_header_from_mat(mat_dict: dict) -> tuple[float, float]:
    """返回 (smin, smax)。假定 meta.slows 是 1-D 升序数组。"""
    slows = np.asarray(mat_dict["meta"]["slows"]).squeeze()
    return float(slows[0]), float(slows[-1])


def key_to_first_time(mat_dict: dict, key: str) -> str:
    """
    读取 <key>_t0 并返回 ISO-8601 字符串。

    - 对 loadmat(..., simplify_cells=True)：
        可能得到 str / bytes / np.ndarray 标量。
    - 若是 ndarray，则 .item() 抽出标量。
    - 若是 bytes，则 .decode() 转成 str。
    """
    iso = mat_dict[f"{key}_t0"]

    if isinstance(iso, np.ndarray):  # 标量数组 or char array
        iso = iso.item()  # → bytes or str

    if isinstance(iso, bytes):
        iso = iso.decode()

    return str(iso)


# ---- 以下是峰值检测和可视化相关的辅助函数 (可以直接复用你已有的) ----

from scipy.ndimage import maximum_filter


def find_all_local_maxima(tfk_array, neighborhood_size=3):
    """
    Identify all local maxima in a 2D array using a maximum filter.

    Returns
    -------
    max_coords : np.ndarray, shape (P, 2)
        The (row, col) coordinates of all local maxima.
    max_values : np.ndarray, shape (P,)
        The amplitude of each local maximum.
    median_tfk : float
        The median value of the entire tfk_array.
    median_max : float
        The median value of all local maxima.
    """
    filtered = maximum_filter(tfk_array, size=neighborhood_size)
    local_max_mask = tfk_array == filtered

    max_coords = np.argwhere(local_max_mask)
    max_values = tfk_array[local_max_mask]
    median_tfk = np.median(tfk_array.ravel())
    median_max = np.median(max_values)

    return max_coords, max_values, median_tfk, median_max


def rc_to_slowness(row, col, smin, smax, Nrows, Ncols, swap_xy=True):
    """
    Convert (row, col) indices to slowness coordinates (x, y).
    """
    if Ncols > 1:
        x_coord = smin + col * (smax - smin) / (Ncols - 1)
    else:
        x_coord = smin
    if Nrows > 1:
        y_coord = smin + row * (smax - smin) / (Nrows - 1)
    else:
        y_coord = smin

    if swap_xy:
        return (y_coord, x_coord)
    else:
        return (x_coord, y_coord)


# peak_utils.py ------------------------------------------------------------
def select_peaks_adaptive(
    max_coords: np.ndarray,
    max_values: np.ndarray,
    *,
    smin: float,
    smax: float,
    Nrows: int,
    Ncols: int,
    alpha: float = 3.7,
    swap_xy: bool = True,
) -> tuple[list[tuple[tuple[float, float], float]], float]:
    """
    用自适应阈值挑峰值；返回 peaks, threshold
        peaks = [ ((x, y), val), ... ]  已按 val 降序
    """
    median_val = np.median(max_values)
    threshold = median_val + alpha * np.std(max_values)

    keep = max_values >= threshold
    coords = max_coords[keep]
    values = max_values[keep]

    order = np.argsort(values)[::-1]
    coords = coords[order]
    values = values[order]

    peaks: list[tuple[tuple[float, float], float]] = []
    for (r, c), v in zip(coords, values):
        x, y = rc_to_slowness(r, c, smin, smax, Nrows, Ncols, swap_xy=swap_xy)
        peaks.append(((x, y), float(v)))

    return peaks, float(threshold)


def visualize_tfk_with_peaks(
    fk_array: np.ndarray,
    smin: float,
    smax: float,
    peaks: Sequence[tuple[tuple[float, float], float]],
    ax: plt.Axes,
    *,
    title: str | None = None,
    set_x: bool = False,
    set_y: bool = False,
    vmin: float = -20.0,
    vmax: float = 0.0,
    cmap: str = "gist_stern_r",
    scale_to_max: bool = True,
) -> Any:
    """
    显示 TF-K 图并叠加峰值。

    Parameters
    ----------
    fk_array : 2-D power map (float32/64)
    smin, smax : slowness 边界 (s/deg)
    peaks : [ ((x, y), amp_db), ... ]
    ax : matplotlib.Axes
    title : str | None
    set_x / set_y : 是否给当前子图加 xlabel / ylabel
    vmin, vmax : dB 色标范围
    cmap : imshow colormap
    scale_to_max : 若 True，先除以 `fk_array.max()` 再转 dB，使最高点为 0 dB

    Returns
    -------
    im : AxesImage 方便外层加 colorbar
    """

    eps = np.finfo(fk_array.dtype).eps
    fk_safe = np.maximum(fk_array, eps)

    if scale_to_max:
        fk_safe = fk_safe / fk_safe.max()

    fk_db = 10.0 * np.log10(fk_safe)

    im = ax.imshow(
        fk_db.T,
        extent=[smin, smax, smax, smin],  # 注意转置后 y 轴翻转
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # im = ax.imshow(
    #     fk_db.T,
    #     extent=[smin, smax, smax, smin],   # 注意转置后 y 轴翻转
    #     interpolation="none",
    #     cmap=cmap,
    # )

    # 坐标轴 & 标签
    ax.set_xlim(smin, smax)
    ax.set_ylim(smin, smax)
    if set_x:
        ax.set_xlabel("E–W slowness [s/deg]")
    if set_y:
        ax.set_ylabel("N–S slowness [s/deg]")
    if title:
        ax.set_title(title)

    # 圆环 (固定震源方位参考，可按需改半径)
    for r in (3.312, 1.870):
        ax.add_artist(plt.Circle((0, 0), r, color="w", fill=False, alpha=0.35, lw=1.0))

    # 高亮峰值
    for (x, y), _ in peaks:
        ax.plot(x, y, "y*", markersize=6, mec="k", mew=0.3)

    return im


# ---- 关键：主函数，批量处理多个 .mat 文件，每个文件只有一个 tfk_**** ----
def plot_files_to_pdf(
    mat_files: list[str],
    pdf_path: str | Path,
    *,
    alpha: float = 3.7,
    neighborhood_size: int = 3,
    vmin_db: float = -2.5,
    vmax_db: float = 0,
) -> None:
    """
    批量读取 .mat (每个仅含 1 个 fk_* 数组) → 峰值检测 → 3×4 分页写入 PDF。
    Parameters
    ----------
    mat_files : list[str]
    pdf_path  : 输出 PDF 路径
    alpha : float
        select_peaks_adaptive 的阈值系数
    neighborhood_size : int
        maximum_filter 的邻域大小
    vmin_db, vmax_db : float
        图像统一 dB 范围，便于跨页对比
    """

    if not mat_files:
        raise ValueError("No mat files provided")

    nrows, ncols, per_page = 3, 4, 12
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf, plt.rc_context(
        {"font.family": "serif", "font.size": 11, "axes.linewidth": 1.2}
    ):
        d = pdf.infodict()
        d["Title"] = f"Beamforming {pdf_path.stem}"
        d["Author"] = "bf_fk_peaks_pdf"
        d["Subject"] = f"Beamforming result for {pdf_path.stem}"
        d["CreationDate"] = datetime.now()

        # ───────── 分页循环 ───────────────────────────────────────────────
        for chunk in tqdm(chunk_list(mat_files, per_page), desc="Plot → PDF"):
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(16, 12),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            # 先关闭所有子图，后面遇到空位就保持关闭
            for ax in axs.ravel():
                ax.axis("off")

            for idx, mfile in enumerate(chunk):
                i, j = divmod(idx, ncols)
                ax = axs[i, j]
                ax.axis("on")

                # ① 读取 .mat（展开结构体）
                mat = loadmat_simplified(mfile)
                (fk_key,) = list_fk_keys(mat)  # 断言唯一
                fk = np.asarray(mat[fk_key], dtype=np.float32)
                Nrows, Ncols = fk.shape

                # ② 本文件自己的 smin/smax
                smin, smax = read_header_from_mat(mat)

                # ③ 峰值检测
                max_rc, max_val, *_ = find_all_local_maxima(
                    fk, neighborhood_size=neighborhood_size
                )
                peaks, _ = select_peaks_adaptive(
                    max_rc,
                    max_val,
                    smin=smin,
                    smax=smax,
                    Nrows=Nrows,
                    Ncols=Ncols,
                    alpha=alpha,
                    swap_xy=True,
                )

                # ④ 绘图
                time_str = key_to_first_time(mat, fk_key)
                visualize_tfk_with_peaks(
                    fk,
                    smin,
                    smax,
                    peaks,
                    ax,
                    title=time_str,
                    set_x=(i == nrows - 1),
                    set_y=(j == 0),
                    vmin=vmin_db,
                    vmax=vmax_db,
                )

            # ⑤ 每页共用 colorbar
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            sm = plt.cm.ScalarMappable(
                cmap="gist_stern_r", norm=plt.Normalize(vmin=vmin_db, vmax=vmax_db)
            )
            fig.colorbar(sm, cax=cax, label="dB")

            pdf.savefig(fig)
            plt.close(fig)
    print(f"[OK] PDF saved → {pdf_path}")


def find_peaks_from_big_mat(
    mat_file: str | os.PathLike,
    *,
    alpha: float = 8.0,  # ← 与绘图保持一致
    neighborhood_size: int = 3,
    swap_xy: bool = True,
) -> dict[str, list[tuple[float, float]]]:
    """
    扫描“大 mat”文件中所有 FK 图并返回每张图的峰值坐标。

    Returns
    -------
    dict
        { "<文件名>_<ISO时间>": [(x1, y1), (x2, y2), ...], ... }
    """
    # 1) 载入 .mat（展开结构体）
    mat_dict = loadmat_simplified(mat_file)

    # 2) 公共头信息：slowness 范围
    smin, smax = read_header_from_mat(mat_dict)

    # 3) 列出功率谱键
    fk_keys = sorted(list_fk_keys(mat_dict))

    # 4) 结果
    result: dict[str, list[tuple[float, float]]] = {}
    day_info = Path(mat_file).stem

    # 5) 逐张提峰值
    for fk_key in fk_keys:
        time_str = key_to_first_time(mat_dict, fk_key)  # ISO-8601
        fk_array = np.asarray(mat_dict[fk_key], dtype=np.float32)
        Nrows, Ncols = fk_array.shape

        max_rc, max_val, *_ = find_all_local_maxima(
            fk_array, neighborhood_size=neighborhood_size
        )
        peaks, _ = select_peaks_adaptive(
            max_rc,
            max_val,
            smin=smin,
            smax=smax,
            Nrows=Nrows,
            Ncols=Ncols,
            alpha=alpha,
            swap_xy=swap_xy,
        )

        result[f"{day_info}_{time_str}"] = [coord for coord, _ in peaks]

    return result


def worker(args):
    mat_file, alpha, neighborhood_size = args
    return find_peaks_from_big_mat(
        mat_file, alpha=alpha, neighborhood_size=neighborhood_size
    )


if __name__ == "__main__":
    import glob
    import os
    import pickle
    from multiprocessing import Pool

    # =========== 参数区（直接在代码中写死） ===========
    mat_dir = "/home/wjx/bf_work/temp_output/bf_mat"  # .mat 文件所在目录
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = f"{output_dir}/bf_x2_fk_peaks.pdf"  # 画图输出PDF路径
    output_pkl = f"{output_dir}/bf_x2_peaks.pkl"  # 并行提取峰值后的结果存储PKL文件
    alpha = 8.0  # select_peaks_adaptive 中的阈值系数
    neighborhood_size = 3  # find_all_local_maxima中的邻域大小
    processes = 20  # 并行进程数

    # =========== 1) 批量搜集 .mat 文件 ===========
    all_mat = glob.glob(os.path.join(mat_dir, "*.mat"))
    all_mat.sort()

    # =========== 2) 并行提取峰值并存储到 pkl ===========
    # big_result_dict: { "some_label": [ (x1,y1), (x2,y2), ...], ... }
    big_result_dict = {}
    tasks = [(mat_file, alpha, neighborhood_size) for mat_file in all_mat]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    with Pool(processes=processes) as pool:
        for local_dict in tqdm(
            pool.imap_unordered(worker, tasks),
            total=len(tasks),
            desc="[Extracting peaks]",
        ):
            # local_dict 形如: { label: [ (x,y), (x,y), ... ], ... }
            big_result_dict.update(local_dict)

    # 将字典写入 pickle
    with open(output_pkl, "wb") as f:
        pickle.dump(big_result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle saved at: {output_pkl}")

    # =========== 3) 调用 plot_files_to_pdf 输出 PDF ===========
    # 同样传入 alpha, neighborhood_size
    plot_files_to_pdf(
        mat_files=all_mat,
        pdf_path=output_pdf,
        alpha=alpha,
        neighborhood_size=neighborhood_size,
    )

    print("[INFO] All tasks finished.")
