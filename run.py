#!/usr/bin/env python
"""
run.py
===================

1. 解析 INI
2. 生成 *.txt sac-list
3. 多 GPU 并行执行 IAS-Capon, 生成 *.mat

新增：
-----
使用全局变量 ``VERBOSE`` 控制是否打印关键信息。
- ``VERBOSE = True``  → 输出 INFO 级及以上日志，并显示数组概要等信息。
- ``VERBOSE = False`` → 仅输出 WARNING 及以上日志，静默运行。
"""

from __future__ import annotations
from pathlib import Path
import logging, sys
from obspy import read
from caponlib.config_parser import (
    parse_and_validate,
    print_array_summary,
    logger as cfg_logger,
)
from caponlib.sac_refactor import (
    organise_sac_files,
    write_sac_lists_by_time,
)
from caponlib.scheduler import run_parallel, run_serial
from caponlib.windows import build_windows,attach_times
from caponlib.ias_capon import ias_capon  # ← 显式算法注入

# ---------------- 全局控制开关 ----------------
VERBOSE: bool = False  # ← 关键信息输出开关（True 打印 / False 静默）
TEST_MODE: bool = False  # ← True 时仅打印 beam_cfg，不跑计算

# ---------------- 固定配置文件 ----------------
INI_FILE = Path("configs/default.ini")  # ← 只改这一行即可换配置
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---------- logging ----------------------------------------------
    log_level = logging.INFO if VERBOSE else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ini_file = Path(INI_FILE)

    # ---------- 0) 解析配置 ------------------------------------------
    try:
        array, window, slow, bands, device, output, _ = parse_and_validate(ini_file)

        if VERBOSE:
            print_array_summary(array)  # 仅在 VERBOSE 时打印阵列概要

        cfg_logger.debug("win_len = %d", window["win_len"])
        cfg_logger.debug("GPU list : %s", device["ids"])
        cfg_logger.debug("CPU used : %s", device["cpu_num"])
        sac_list_dir = Path(output["dir"]) / "saclist"
    except Exception:
        cfg_logger.exception("Failed to parse config")
        sys.exit(1)

    # ---------- 1) 生成 sac-list (*.txt) -----------------------------
    try:
        files_group = organise_sac_files(array, cpu_count=device["cpu_num"])
        cfg_logger.info("Writing sac-list files → %s", sac_list_dir)
        write_sac_lists_by_time(files_group, sac_list_dir, window["min_stations"])
        cfg_logger.info("Wrote %d sac-list files → %s", len(files_group), sac_list_dir)
    except Exception:
        cfg_logger.exception("SAC-list generation failed")
        sys.exit(1)
    sac_lists = sorted(sac_list_dir.glob("*.txt"))

    # ---------- 2) 生成 meta window 列表 -----------------------------
    sac_list_example = sac_lists[0]
    with open(sac_list_example, "r") as f:
        sac_example = f.readline().strip()
    stream_example = read(sac_example, headonly=True)

    tr0 = stream_example[0]
    start_epoch = tr0.stats.starttime.datetime       # UTC datetime
    dt           = float(tr0.stats.delta)
    npts         = int(tr0.stats.npts)

    # ★★★ 改动：不再传 start_epoch ★★★
    meta_windows = build_windows(
        dt          = dt,
        npts        = npts,
        winlen      = window["win_len"],
        hop_ratio   = window["hop_ratio"],
        skip_win_idx= window["skip_step"],
    )

    # ----------- 仅在 VERBOSE 模式下预览前 5 个窗口 -------------------
    if VERBOSE:
        preview = attach_times(
            meta_windows[:5],            # 前 5 个就够
            start_epoch = start_epoch,
            dt          = dt,
        )
        for w in preview:
            cfg_logger.info(
                "  win[%d]: %s — %s  (len %.2fs)",
                w.index,
                w.start_time.isoformat(timespec="seconds"),
                w.end_time.isoformat(timespec="seconds"),
                (w.end_idx - w.start_idx) * dt,
            )

    # -----------3) 生成 slows -----------------------------------------
    s_range = slow["s_range"]  # 最大 |s|，单位 s/km
    s_res = slow["s_res"]  # 步长

    import numpy as np

    # --- ① 对称扫描（-s_range ~ +s_range） --------------------------
    slows = np.arange(-s_range, s_range + 1e-9, s_res).tolist()

    # --- ② 仅非负（0 ~ +s_range），若算法只用幅值 ---------------------
    # slows = np.arange(0.0,        s_range + 1e-9,  s_res).tolist()

    cfg_logger.info(
        "Slowness grid: %d points from %.2f to %.2f s/km, step %.2f",
        len(slows),
        slows[0],
        slows[-1],
        s_res,
    )

    # ---------------- 4) 并行 Beamforming --------------------------
    try:
        run_parallel(
            sac_lists=sac_lists,
            out_dir=Path(output["dir"]) / "bf_mat",
            windows=meta_windows,  # ← Window 列表
            bands=bands,  # ← [(fmin, fmax), ...]
            slows=slows,  # ← 刚生成的平面波慢度网格
            stack_size=window["stack_size"],  #  最小叠加数量
            gpu_list=device["ids"],  # ← List[int]
            task_per_gpu=device["procs_per_gpu"],  # ← int
            logger=cfg_logger,
            compute_fn=None if TEST_MODE else ias_capon,
            progress=True,  # 进度条只在 VERBOSE 时显示
        )
    except Exception:
        cfg_logger.exception("Beamforming run failed (parallel)")
        sys.exit(1)

    # ---------------- 5) 串行 Beamforming（取一条作 sanity-check） ----
    # run_serial(
    #     sac_lists    = sac_lists[:1],   # 只跑第一条
    #     out_dir      = Path(output["dir"]) / "bf_mat",
    #     windows      = meta_windows,
    #     bands        = bands,
    #     slows        = slows,
    #     stack_size   = window["stack_size"],
    #     gpu_list     = device["ids"],
    #     task_per_gpu = device["procs_per_gpu"],
    #     logger       = cfg_logger,
    #     compute_fn   = None if TEST_MODE else ias_capon,
    #     progress     = True,          # 进度条只在 VERBOSE 时显示
    # )
