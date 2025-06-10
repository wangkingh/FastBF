# bf_fk_peaks_pdf.py
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm

plt.rcParams.update({"font.family": "serif", "font.size": 12})
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.minor.width"] = 1.5


def chunk_list(lst, chunk_size):
    """
    将一个列表按给定大小 chunk_size 分割，生成器方式返回。
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def parse_time_label(var_name):
    """
    var_name e.g. 'tfk_0030' or 'tfk_1200'
    """
    # var_name = 'tfk_0030'
    time_str = var_name.replace("tfk_", "")  # '0030'
    if len(time_str) == 4:
        hour = time_str[0:2]
        minute = time_str[2:4]
        return f"{hour}:{minute}"
    else:
        return var_name


def read_header_from_mat(mat_dict):
    """
    从 loadmat() 后的字典 mat_dict 中读取公共头段变量: smin, smax, freq, df。
    返回 (smin, smax, freq, df)。
    """
    smin = float(mat_dict["smin"][0][0])
    smax = float(mat_dict["smax"][0][0])
    freq = float(mat_dict["freq"][0][0])
    df = float(mat_dict["df"][0][0])
    return smin, smax, freq, df


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


def infer_array_shape(max_coords):
    """
    给定局部最大值坐标，推断原始数组形状 (nrows, ncols)。
    """
    max_row = np.max(max_coords[:, 0])
    max_col = np.max(max_coords[:, 1])
    return (max_row + 1, max_col + 1)


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


def select_peaks_adaptive(max_coords, max_values, smin, smax, alpha=3.7, swap_xy=True):
    """
    用自适应阈值挑选峰值: threshold = median + alpha * std。
    返回 peaks, threshold:
        peaks 是 [((x, y), val), ...]，并按 val 降序排序。
    """
    median_val = np.median(max_values)
    measure_val = np.std(max_values)
    threshold = median_val + alpha * measure_val

    keep_mask = max_values >= threshold
    coords_filtered = max_coords[keep_mask]
    values_filtered = max_values[keep_mask]

    # sort descending
    descending_idx = np.argsort(values_filtered)[::-1]
    coords_sorted = coords_filtered[descending_idx]
    values_sorted = values_filtered[descending_idx]

    # convert (row, col) to (x, y)
    Nrows, Ncols = infer_array_shape(max_coords)
    peaks = []
    for (r, c), val in zip(coords_sorted, values_sorted):
        x, y = rc_to_slowness(r, c, smin, smax, Nrows, Ncols, swap_xy=swap_xy)
        peaks.append(((x, y), val))

    return peaks, threshold


def visualize_tfk_with_peaks(
    fk_array,
    smin,
    smax,
    peaks,
    ax,
    title=None,
    set_x=False,
    set_y=False,
):
    """
    - Convert `fk_array` to dB scale.
    - Plot it on the given ax.
    - Overlay local maxima (peaks) in yellow star.
    - Plot circles at fixed radius.
    """
    fk_max = np.max(fk_array)
    if fk_max <= 0:
        raise ValueError("Max of fk_array is non-positive; can't convert to dB.")
    fk_dB = 10 * np.log10(fk_array / fk_max)

    im = ax.imshow(
        fk_dB.T,
        extent=[smin, smax, smax, smin],
        cmap="gist_stern_r",
        interpolation="none",
    )
    if set_x:
        ax.set_xlabel("E-W [s/deg]")
    if set_y:
        ax.set_ylabel("N-S [s/deg]")
    ax.set_xlim([smin, smax])
    ax.set_ylim([smin, smax])

    # plot circles
    circle1 = plt.Circle((0, 0), 3.312, color="w", fill=False, alpha=0.4)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), 1.870, color="w", fill=False, alpha=0.4)
    ax.add_artist(circle2)

    # mark peaks
    for (x, y), val in peaks:
        ax.plot(x, y, "y*", markersize=6)

    if title:
        ax.set_title(title)


# ---- 关键：主函数，批量处理多个 .mat 文件，每个文件只有一个 tfk_**** ----


def plot_files_to_pdf(
    mat_files,
    pdf_path,
    alpha=3.7,
    neighborhood_size=3,
):
    """
    1) 假设每个文件都只有且仅有一个 tfk_**** 数组。
    2) 从每个文件提取 (smin, smax, freq, df, fk_array)。
    3) 对 fk_array 做峰值检测后可视化，分页存入同一个 PDF (3x4)。
    4) alpha: select_peaks_adaptive中的系数
    5) neighborhood_size: find_all_local_maxima中的最大滤波邻域大小
    """

    if not mat_files:
        raise ValueError("No mat files provided!")

    # 读取公共头信息
    first_data = loadmat(mat_files[0])
    smin, smax, freq, df = read_header_from_mat(first_data)

    # 为了分页显示: 每页 3 行 x 4 列 => 12 个图
    nrows, ncols = 3, 4
    per_page = nrows * ncols

    # 确保输出目录存在
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        pdf_basename = os.path.basename(pdf_path)
        date_label = os.path.splitext(pdf_basename)[0]
        d = pdf.infodict()
        d["Title"] = f"Beamforming {date_label}"
        d["Author"] = "My Script"
        d["Subject"] = f"Beamforming result for {date_label}"
        d["CreationDate"] = datetime.now()

        # 分页处理
        for chunk in tqdm(chunk_list(mat_files, per_page), desc="Plotting to PDF"):
            fig, axs = plt.subplots(
                nrows, ncols, figsize=(16, 12), sharex=True, sharey=True
            )

            idx = 0
            for i in range(nrows):
                for j in range(ncols):
                    ax = axs[i, j]
                    if idx < len(chunk):
                        mfile = chunk[idx]
                        mat_dict = loadmat(mfile)
                        tfk_keys = [k for k in mat_dict.keys() if k.startswith("tfk_")]
                        if len(tfk_keys) != 1:
                            raise ValueError(
                                f"{mfile} should have exactly 1 tfk_****, found {len(tfk_keys)}."
                            )
                        var_name = tfk_keys[0]
                        fk_array = mat_dict[var_name]

                        # 做个文件名标签
                        basename = os.path.basename(mfile)
                        time_str = os.path.splitext(basename)[0]

                        # 找局部峰值
                        max_coords, max_values, median_tfk, median_max = (
                            find_all_local_maxima(
                                fk_array, neighborhood_size=neighborhood_size
                            )
                        )
                        peaks, threshold = select_peaks_adaptive(
                            max_coords,
                            max_values,
                            smin,
                            smax,
                            alpha=alpha,  # 使用传入的 alpha
                            swap_xy=True,
                        )

                        # 绘图
                        visualize_tfk_with_peaks(
                            fk_array,
                            smin,
                            smax,
                            peaks,
                            ax,
                            title=time_str,
                            set_x=(i == nrows - 1),
                            set_y=(j == 0),
                        )
                        idx += 1
                    else:
                        ax.axis("off")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved: {pdf_path}")


def find_peaks_from_big_mat(mat_file, alpha=20, neighborhood_size=3, swap_xy=True):
    """
    从给定的 .mat 文件中读取并提取各时刻的峰值 (x,y)，返回一个字典：
    {time_str: [(x1, y1), (x2, y2), ...], ...}
    """

    # 1) 读入 mat 文件
    mat_dict = loadmat(mat_file)

    mat_fname = os.path.basename(mat_file)
    day_info = os.path.basename(mat_fname)
    # 2) 读取公共头段信息：smin, smax, freq, df
    smin, smax, freq, df = read_header_from_mat(mat_dict)

    # 3) 获取所有 'tfk_****' 变量
    tfk_keys = [k for k in mat_dict.keys() if k.startswith("tfk_")]
    tfk_keys.sort()

    # 4) 用于保存结果的字典
    #    键：时间标签（"00:30" 等）；值：[(x1, y1), (x2, y2), ...]
    result_dict = {}

    # 5) 逐个处理 tfk_**** 变量
    for tfk_key in tfk_keys:
        # 解析时间标签，比如 "tfk_0030" -> "00:30"
        time_str = parse_time_label(tfk_key)

        # 取得二维数组
        tfk_array = mat_dict[tfk_key]

        # 查找局部最大值
        max_coords, max_values, median_tfk, median_max = find_all_local_maxima(
            tfk_array, neighborhood_size=neighborhood_size
        )

        # 用自适应阈值筛选
        peaks, threshold = select_peaks_adaptive(
            max_coords, max_values, smin, smax, alpha=alpha, swap_xy=swap_xy
        )
        # peaks 是 [((x, y), val), ...] 形式

        # 只保留 (x, y)，不需要 val
        xy_coords = [coord for coord, val in peaks]

        combined_key = f"{day_info}_{time_str}"

        # 加入结果字典
        result_dict[combined_key] = xy_coords

    return result_dict


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
    mat_dir = "/home/wjx/bf_work/temp_output/bf_mat_script"  # .mat 文件所在目录
    output_dir = "./output_old"
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
