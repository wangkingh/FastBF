from __future__ import annotations
import multiprocessing as mp, os, logging
from pathlib import Path
from typing import List, Dict, Any, Sequence, Callable
from tqdm import tqdm
import traceback
from .windows import Window
from .bftask import BfTask


# ----------------------------------------------------------------------
def _even_split(seq: Sequence[Any], n_chunks: int) -> List[list[Any]]:
    """将序列均匀地分割成 n_chunks 份（最后几份长度可能 +1）。"""
    base, extra = divmod(len(seq), n_chunks)
    out, start = [], 0
    for i in range(n_chunks):
        end = start + base + (1 if i < extra else 0)
        out.append(list(seq[start:end]))
        start = end
    return out


# ---------------- collect ---------------------------------------------
def _collect_tasks(
    sac_files: Sequence[Path],
    windows  : List[Window],
    bands    : List[tuple[float, float]],
    slows    : List[float],
    stack_size:int,
    out_dir  : Path,
    compute_fn:Callable|None,
) -> List[BfTask]:
    """
    将 sac‑list 文本路径封装成 BfTask 对象列表。
    这里 **不再** 关心 compute_fn/skip_idx，保持极简职责。
    """
    tasks: List[BfTask] = []
    for txt in sorted(sac_files):  # 若外层已排序，可去掉 sorted
        tasks.append(BfTask(txt, windows, bands, slows,stack_size,out_dir,compute_fn))
    return tasks

# ----------------------------------------------------------------------
def _run_task(task: BfTask, logger: logging.Logger):
    try:
        task.run(logger=logger)
        return task.sac_txt.name, True, ""
    except Exception:
        return task.sac_txt.name, False, traceback.format_exc()


def _worker(chunk: List[BfTask], gpu_id: int, qlog: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger = logging.getLogger(f"worker-{gpu_id}")
    logger.addHandler(logging.StreamHandler())  # 让日志可见
    for task in chunk:
        qlog.put(_run_task(task, logger))
    qlog.put(("__DONE__", True, ""))


# ---------------- public API ------------------------------------------
def run_parallel(
    sac_lists  : List[Path],
    out_dir    : Path,
    windows : List[Window],
    bands: List[tuple[float,float]],
    slows: List[float],
    stack_size:int,
    gpu_list: List[int],
    task_per_gpu: int,
    logger     : logging.Logger,
    compute_fn : Callable | None = None,   # 保留以兼容旧接口
    progress   : bool = True,
) -> None:
    """并行 Capon 计算 """
    
    if not gpu_list or task_per_gpu <= 0:
        logger.error("gpu_list 为空或 task_per_gpu<=0"); return   # 防除零
        
    # ---------- 收集任务 -------------------------------------------------
    tasks = _collect_tasks(sac_lists, windows, bands, slows,stack_size,out_dir,compute_fn)
    if not tasks:
        logger.error("No sac-list found")
        return

    # ---------- GPU 列表 -------------------------------------------------
    procs_per_gpu = task_per_gpu
    total_proc = len(gpu_list) * procs_per_gpu
    chunks = [c for c in _even_split(tasks, total_proc) if c]

    qlog, procs = mp.Queue(), []
    for idx, chunk in enumerate(chunks):
        gpu_id = gpu_list[idx // procs_per_gpu]
        p = mp.Process(target=_worker, args=(chunk, gpu_id, qlog))
        p.start()
        procs.append(p)

    # ---------- 进度循环 -------------------------------------------------
    finished = 0
    pbar = tqdm(total=len(tasks), desc="Beamforming", ncols=80, disable=not progress)

    while finished < len(procs):
        name, ok, msg = qlog.get()
        if name == "__DONE__":
            finished += 1
            continue
        pbar.update(1)
        if not ok:
            logger.error("Task %s failed:\n%s", name, msg)
    pbar.close()

    # ---------- 等待所有进程退出 -----------------------------------------
    for p in procs:
        p.join()


# ----------------------------------------------------------------------
# public API – serial
# ----------------------------------------------------------------------

def run_serial(
    sac_lists: List[Path],
    out_dir: Path,
    windows: List[Window],
    bands: List[tuple[float, float]],
    slows: List[float],
    stack_size:int,
    gpu_list: List[int],
    task_per_gpu: int,
    logger: logging.Logger,
    compute_fn: Callable | None = None,
    progress: bool = True,
) -> None:
    """串行 Capon 计算，与 `run_parallel 。"""

    # -------- 收集任务 --------------------------------------------------
    tasks = _collect_tasks(sac_lists, windows, bands, slows,stack_size, out_dir, compute_fn)
    if not tasks:
        logger.error("No sac-list found")
        return

    # -------- 设定运行设备 --------------------------------------------
    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])
        logger.info("Serial run on GPU %s", gpu_list[0])
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info("Serial run on CPU")

    # -------- 串行执行 --------------------------------------------------
    for task in tasks:
        task.run(logger=logger)