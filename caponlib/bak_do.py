from cfk import (
    compute_fk,
    fft_cuda,
    compute_sm,
    matrix_invert,
)

print("fft_cuda ptr ok:", fft_cuda)  # None 或 <built-in>?
from scipy.io import savemat
from obspy import read, Stream
import numpy as np
from numba import jit
import multiprocessing
from tqdm import tqdm
import os
import glob
import math as m
import time


def grt(r1, r2, s1, s2):
    slat = s1 * np.pi / 180.0
    slon = s2 * np.pi / 180.0
    elat = r1 * np.pi / 180.0
    elon = r2 * np.pi / 180.0

    slat = m.atan(0.996647 * m.tan(slat))
    elat = m.atan(0.996647 * m.tan(elat))

    slat = np.pi / 2.0 - slat
    elat = np.pi / 2.0 - elat

    if slon < 0.0:
        slon += 2.0 * np.pi
    if elon < 0.0:
        elon += 2.0 * np.pi

    a = m.sin(elat) * m.cos(elon)
    b = m.sin(elat) * m.sin(elon)
    c = m.cos(elat)
    a1 = m.sin(slat) * m.cos(slon)
    b1 = m.sin(slat) * m.sin(slon)
    c1 = m.cos(slat)

    cd = a * a1 + b * b1 + c * c1

    if cd > 1.0:
        cd = 1.0
    if cd < -1.0:
        cd = -1.0
    decl = m.acos(cd) * 180.0 / m.pi
    dist = decl * np.pi * 6371.0 / 180.0

    tmp1 = m.cos(elon) * m.cos(slon) + m.sin(elon) * m.sin(slon)
    tmp2a = 1.0 - cd * cd

    if tmp2a <= 0.0:
        tmp2 = 0.0
        tmp3 = 1.0
    else:
        tmp2 = m.sqrt(tmp2a)
        tmp3 = (m.sin(elat) * m.cos(slat) - m.cos(elat) * m.sin(slat) * tmp1) / tmp2

    if tmp3 > 1.0:
        tmp3 = 1.0
    if tmp3 < -1.0:
        tmp3 = -1.0
    z = m.acos(tmp3)

    if (m.sin(slon) * m.cos(elon) - m.cos(slon) * m.sin(elon)) < 0.0:
        z = 2.0 * m.pi - z

    az = 180.0 * z / m.pi

    tmp1 = m.cos(slon) * m.cos(elon) + m.sin(slon) * m.sin(elon)
    tmp2a = 1.0 - cd * cd
    if tmp2a <= 0.0:
        tmp2 = 0.0
        tmp3 = 1.0
    else:
        tmp2 = m.sqrt(tmp2a)
        tmp3 = (m.sin(slat) * m.cos(elat) - m.cos(slat) * m.sin(elat) * tmp1) / tmp2

    if tmp3 > 1.0:
        tmp3 = 1.0
    if tmp3 < -1.0:
        tmp3 = -1.0

    bz = m.acos(tmp3)

    if (m.sin(elon) * m.cos(slon) - m.cos(elon) * m.sin(slon)) < 0.0:
        bz = 2.0 * m.pi - bz

    baz = 180.0 * bz / m.pi

    return decl, dist, az, baz


def get_metadata2(st, nr):
    d = {}
    for i in range(nr):
        d[st[i].stats.station] = st[i].stats.sac.stla, st[i].stats.sac.stlo
    return d


def metric_mseed(st, d, nr):
    rx_0, ry_0 = d[st[0].stats.station]
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        rx_i, ry_i = d[st[i].stats.station]
        decl, dist, az, baz = grt(float(rx_0), float(ry_0), float(rx_i), float(ry_i))
        rx[i] = decl * np.cos(0.017453 * (90.0 - az))
        ry[i] = decl * np.sin(0.017453 * (90.0 - az))
    return rx, ry


def martix_add(smr, smi):
    tx = smr + smi * 1j
    return tx


# @jit(nopython=True)
def martix_invert_numba(tx, fave):
    itx = np.empty_like(tx)
    for m in range(2 * fave + 1):  # 计算tx的逆矩阵
        itx[m] = np.linalg.inv(tx[m])
    return itx


# nsamp（每个窗口的采样数）、nr（台站数）、rx和ry（台站的相对坐标）、st（波形数据的Stream对象）、smin、smax、sinc（DOA搜索范围和间隔）、find（感兴趣频率）
# fave（平均频率）、delta（采样间隔）、dl（对角负载因子）、overlap（是否使用重叠窗口）和taper（是否应用汉宁窗）
# nsamp*delta(采样率)
def IAS_Capon(
    nsamp,
    nr,
    rx,
    ry,
    st,
    smin,
    smax,
    sinc,
    freq,
    df,
    delta,
    dl,
    fk_file,
    skiplist,
    overlap,
    taper,
    divided=False,
):
    find = int(freq * nsamp * delta)
    fave = int(df * nsamp * delta)

    # print("nsamp = ", nsamp, "delta = ", delta, "find = ", find, "fave = ", fave)
    # print("IAS Capon DOA estimation is performed at:", "freq", freq, "+-", df)
    kmin = 2 * np.pi * smin * freq
    kmax = 2 * np.pi * smax * freq
    kinc = 2 * np.pi * sinc * freq
    nk = int(((kmax - kmin) / kinc + 0.5) + 1)
    period_num = int(np.array(st[0].stats.npts / nsamp))
    skip_num = len(skiplist)

    nwin = period_num - skip_num
    rft = np.zeros((nwin, nr, 2 * fave + 1), dtype=np.float32)
    ift = np.zeros((nwin, nr, 2 * fave + 1), dtype=np.float32)
    for i in range(nwin):
        xt = np.zeros((nr, nsamp), dtype=np.float64)
        for j in range(nr):
            xt[j] = st[j][i * nsamp : (i + 1) * nsamp]
            xt[j] -= np.mean(xt[j])
            if taper:
                xt[j] *= np.hanning(nsamp)
        ft = fft_cuda(xt, nsamp, nr)
        band = ft[:, (find - fave) : (find + fave + 1)]
        rft[i] = band.real
        ift[i] = band.imag

    smr = np.zeros((2 * fave + 1, nr, nr))
    smi = np.zeros((2 * fave + 1, nr, nr))

    sm = np.zeros(((2 * fave + 1) * 2 * nr * nr), dtype=np.float32)
    if divided == True:
        divided_nwin = nwin
    else:
        divided_nwin = 1

    # prepare dict for saving the results
    collected_data = {
        "smin": smin,
        "smax": smax,
        "freq": freq,
        "df": df,
    }

    # start the main loops
    # for j in tqdm(range(divided_nwin), desc="Processing", ncols=80):
    for j in range(divided_nwin):
        if divided == True:
            # Modified by Wang Jingxi, 2025-03-01, another way to concatenate the rft and ift
            new_ft_2d = np.concatenate((rft[j], ift[j]), axis=0)  # 2*nr * 2*fave+1
            # => shape: (2, nr, 2*fave+1)
            new_ft_3d = new_ft_2d[None, ...]
            # => shape: (1, 2, nr, 2*fave+1)
            new_ft = new_ft_3d.ravel()
        else:
            new_ft = np.zeros((nwin, 2 * nr, 2 * fave + 1), dtype=np.float32)
            new_ft = np.concatenate((rft, ift), axis=1)
            new_ft = np.ravel(new_ft)

        # sm = compute_sm_so_py(new_ft, nwin // divided_nwin, nr, fave, sm)
        compute_sm(new_ft, nwin // divided_nwin, nr, fave, sm)
        sm = sm.reshape((2 * fave + 1, 2 * nr, nr))
        smr = sm[:, :nr, :]
        smi = sm[:, nr:, :]

        mi = np.identity(nr)
        for i in range(2 * fave + 1):
            smr[i] += mi * smr[i].trace() / (nsamp) * dl

        tx = martix_add(smr.astype(np.float32), smi.astype(np.float32))
        tx = np.ascontiguousarray(tx, dtype=np.complex64)  # 给 CUDA
        itx = np.empty_like(tx)

        ##CUDA
        tx = np.ravel(tx)
        itx = np.ravel(itx)
        # martix_invert(tx, itx, nr, 2 * fave + 1)
        itx = matrix_invert(tx, nr, 2 * fave + 1)
        itx = itx.reshape((2 * fave + 1, nr, nr))

        ismr = itx.real
        ismi = itx.imag

        # 定义输入数据和参数
        # tfk = np.zeros((nk * nk), dtype=np.float32)
        kinc = np.float32(kinc)
        kmin = np.float32(kmin)
        delta = np.float32(delta)
        smin = np.float32(smin)
        sinc = np.float32(sinc)

        ismr = np.ravel(ismr)
        ismi = np.ravel(ismi)
        rx = np.ravel(rx)
        ry = np.ravel(ry)
        ismr = np.ascontiguousarray(itx.real, dtype=np.float32).ravel()
        ismi = np.ascontiguousarray(itx.imag, dtype=np.float32).ravel()
        rx = np.ascontiguousarray(rx, dtype=np.float32).ravel()
        ry = np.ascontiguousarray(ry, dtype=np.float32).ravel()
        tfk = np.zeros(nk * nk, dtype=np.float32)  # out buffer
        # 调用CUDA核函数进行并行计算
        tfk = compute_fk(
            ismr,  # ndarray float32
            ismi,  # ndarray float32
            tfk,  # 预分配输出 (nk*nk,) float32
            nk,
            rx,  # ndarray float32
            ry,  # ndarray float32
            kinc,
            kmin,
            nr,
            nsamp,
            delta,
            smin,
            sinc,
            fave,
            find,
        )
        ##func for 4 gpu
        # tfk = compute_fk_mulgpu(ismr,ismr.size,ismi,ismi.size,tfk,tfk.size,nk,rx,rx.size,ry,ry.size,kinc,kmin,nr,nsamp,delta,smin,sinc,fave,find)
        tfk = tfk.reshape((nk, nk))
        total_seconds = j * nsamp * delta
        hours = int((total_seconds // 3600) % 24)
        minutes = int((total_seconds % 3600) // 60)
        hm_info = f"{hours:02}{minutes:02}"

        # 准备保存数据
        var_name = f"tfk_{hm_info}"
        collected_data[var_name] = tfk.real

    savemat(fk_file, collected_data)
    print(f"[INFO] Data saved to {fk_file}")


def single_saclist_process(arg_list):
    sac_list_file = arg_list[0]
    print(f"[INFO]:Processing {sac_list_file}")
    fk_file = arg_list[1]
    skiplist = arg_list[2]
    nsamp = arg_list[3]
    smin = arg_list[4]
    smax = arg_list[5]
    sinc = arg_list[6]
    freq = arg_list[7]
    df = arg_list[8]
    dl = arg_list[9]
    overlap_flag = arg_list[10]
    taper_flag = arg_list[11]
    devided_flag = arg_list[12]
    st = Stream()
    all_sac_files = []
    with open(sac_list_file, "r") as f:
        for line in f:
            file_path = line.strip()
            all_sac_files.append(file_path)
    if len(all_sac_files) < 10:
        return
    for file_path in all_sac_files:
        st += read(file_path)
    nr = st.count()
    dic_meta = get_metadata2(st, nr)
    dt = st[0].stats.delta
    rx, ry = metric_mseed(st, dic_meta, nr)
    IAS_Capon(
        nsamp,
        nr,
        rx,
        ry,
        st,
        smin,
        smax,
        sinc,
        freq,
        df,
        dt,
        dl,
        fk_file,
        skiplist=skiplist,
        overlap=overlap_flag,
        taper=taper_flag,
        divided=devided_flag,
    )


# ============ 尝试在多GPU设备上执行任务 ============
def worker(arg_list, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for arg in arg_list:
        single_saclist_process(arg)


if __name__ == "__main__":
    # ==== USER INPUT PARAMETER ===
    nsamp = 9000
    smin = -12.0
    smax = 12.0
    sinc = 0.2  # s 的步长

    # 计算频带
    freq = 0.15
    df = 0.05
    dl = 1
    overlap_flag = False
    taper_flag = True

    devided_flag = False

    data_delta = 0.2  # set by users, for verbose purpose
    find = int(freq * nsamp * data_delta)
    fave = int(df * nsamp * data_delta)
    print("nsamp = ", nsamp, "delta = ", data_delta, "find = ", find, "fave = ", fave)
    print("IAS Capon DOA estimation is performed at:", "freq", freq, "+-", df)

    skip_list = []
    sac_list_dir = "/home/denglin/bf_work/temp_output/saclist"
    output_dir = "/home/denglin/bf_work/temp_output/bf_mat_script"
    num_gpus = 1
    processes_per_gpu = 6
    old_tt = time.time()
    os.makedirs(output_dir, exist_ok=True)

    arg_list_list = []

    for sac_list in glob.glob(sac_list_dir + "/*.txt"):
        sac_list_file_name = os.path.basename(sac_list)
        label = sac_list_file_name.split(".")[0]
        output_file = os.path.join(output_dir, label + ".mat")
        arg_list = [
            sac_list,
            output_file,
            skip_list,
            nsamp,
            smin,
            smax,
            sinc,
            freq,
            df,
            dl,
            overlap_flag,
            taper_flag,
            devided_flag,
        ]
        arg_list_list.append(arg_list)

    # ========== 串行测试（单进程）==========
    print("\n[INFO] 开始串行测试……")
    serial_t0 = time.time()

    # 不开多进程、不改 GPU 设定，直接逐个任务跑
    for arg in arg_list_list:
        single_saclist_process(arg)

    serial_cost = time.time() - serial_t0
    print(f"[INFO] 串行测试完成，用时 {serial_cost:.2f} 秒")
