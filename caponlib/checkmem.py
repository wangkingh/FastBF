from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .progress import progress


@dataclass(frozen=True)
class GpuMemoryPlan:
    gpu_id: int
    arena_bytes: int
    global_steering_bytes: int
    worker_bytes: int
    global_steering_mode: str
    free_bytes: int
    total_bytes: int
    safety_ratio: float
    recommended_workers: int
    configured_workers: int
    cov_window_batch: int
    max_nr: int
    max_nwin: int
    max_nsamp: int
    max_selected_bins: int
    max_chunk_bins: int


@dataclass(frozen=True)
class CheckMemResult:
    workspace_dir: Path
    manifest_path: Path
    native_binary: Path
    sac_list_count: int
    gpu_plans: tuple[GpuMemoryPlan, ...]


def checkmem(config_path: str | Path, *, safety_ratio: float = 0.80) -> CheckMemResult:
    if not 0.0 < safety_ratio <= 1.0:
        raise RuntimeError("safety_ratio must be in (0, 1]")

    from .inventory import ensure_prepared_workspace

    progress(f"checking GPU memory for {Path(config_path).expanduser()}")
    prepared = ensure_prepared_workspace(config_path)
    progress(f"using prepared workspace {prepared.workspace_dir}")
    sac_lists = sorted(prepared.inventory.sac_list_dir.glob("*.tsv"))
    if not sac_lists:
        raise RuntimeError(f"No SAC list TSV files were generated under {prepared.inventory.sac_list_dir}")
    progress(f"found {len(sac_lists):,} SAC list(s) under {prepared.inventory.sac_list_dir}")

    native_dir = prepared.workspace_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    manifest = native_dir / "checkmem_saclists.txt"
    _write_manifest(manifest, sac_lists)
    progress(f"wrote checkmem manifest to {manifest}")

    backend = str(prepared.beamforming.get("backend", "reference"))
    native = _native_binary(backend)
    progress(f"using native binary {native}")
    band = prepared.bands[0]
    window = prepared.window
    slow = prepared.slow
    plans: list[GpuMemoryPlan] = []
    gpu_ids = [int(gpu_id) for gpu_id in prepared.device["ids"]]
    progress(
        "native plan settings: "
        f"backend={backend}, "
        f"method={prepared.beamforming['method']}, "
        f"mode={prepared.beamforming['mode']}, "
        f"band={band[0]}/{band[1]}, "
        f"window={window['win_len']}/{window['shift_len']}, "
        f"gpu_ids={','.join(str(gpu_id) for gpu_id in gpu_ids)}"
    )
    for gpu_id in gpu_ids:
        plan = _run_native_plan(
            native=native,
            manifest=manifest,
            band=band,
            win_len=window["win_len"],
            shift_len=window["shift_len"],
            taper=str(window.get("taper", "tukey")),
            taper_alpha=float(window.get("taper_alpha", 0.05)),
            s_range=slow["s_range"],
            s_res=slow["s_res"],
            mode=prepared.beamforming["mode"],
            method=prepared.beamforming["method"],
            group_windows=int(prepared.beamforming.get("group_windows", 0)),
            gpu_id=gpu_id,
            configured_workers=int(prepared.device["procs_per_gpu"]),
            safety_ratio=safety_ratio,
        )
        plans.append(plan)
        progress(
            f"GPU {gpu_id} plan complete: "
            f"worker={format_bytes(plan.worker_bytes)}, "
            f"free={format_bytes(plan.free_bytes)}, "
            f"recommended_workers={plan.recommended_workers}"
        )

    return CheckMemResult(
        workspace_dir=prepared.workspace_dir,
        manifest_path=manifest,
        native_binary=native,
        sac_list_count=len(sac_lists),
        gpu_plans=tuple(plans),
    )


def _write_manifest(path: Path, sac_lists: Sequence[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sac_list in sac_lists:
            handle.write(str(sac_list.resolve()))
            handle.write("\n")


def _native_binary(backend: str = "reference") -> Path:
    from .system import resolve_executable

    if backend == "power":
        env_bin = "FASTBF_NATIVE_POWER_BIN"
        default_names = ("fastbf_native_power",)
    else:
        env_bin = "FASTBF_NATIVE_BIN"
        default_names = ("fastbf_native",)

    candidate = Path(
        resolve_executable(
            value=os.environ.get(env_bin),
            default_names=default_names,
            root=os.environ.get("FASTBF_NATIVE_ROOT", "AUTO"),
        )
    ).expanduser()
    if not candidate.is_file():
        raise RuntimeError(
            f"native binary not found: {candidate}; run `make install` or set {env_bin}"
        )
    return candidate.resolve()


def _run_native_plan(
    *,
    native: Path,
    manifest: Path,
    band: tuple[float, float],
    win_len: float,
    shift_len: float,
    taper: str,
    taper_alpha: float,
    s_range: float,
    s_res: float,
    mode: str,
    method: str,
    group_windows: int,
    gpu_id: int,
    configured_workers: int,
    safety_ratio: float,
) -> GpuMemoryPlan:
    cmd = [
        str(native),
        "--plan-only",
        "--saclist-list",
        str(manifest),
        "-B",
        f"{band[0]}/{band[1]}",
        "-W",
        f"{win_len}/{shift_len}",
        "--taper",
        taper,
        "--tukey-alpha",
        f"{taper_alpha:g}",
        "-S",
        f"{s_range}/{s_res}",
        "--mode",
        mode,
        "--method",
        method,
        "-G",
        str(gpu_id),
    ]
    if mode == "grouped":
        cmd.extend(["--group-windows", str(group_windows)])
    env = os.environ.copy()
    progress(
        f"GPU {gpu_id}: running native --plan-only on {manifest.name}; "
        "this reads SAC list metadata but does not run beamforming"
    )
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=native.parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(
            "native checkmem failed for GPU "
            f"{gpu_id}:\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    progress(f"GPU {gpu_id}: native plan finished in {elapsed:.1f}s")
    fields = _parse_plan_line(proc.stdout)
    arena = int(fields["arena_bytes"])
    global_steering = int(fields.get("global_steering_bytes", "0"))
    worker_bytes = arena + global_steering
    free = int(fields["free_bytes"])
    total = int(fields["total_bytes"])
    recommended = 0 if worker_bytes <= 0 else max(1, math.floor(free * safety_ratio / worker_bytes))
    return GpuMemoryPlan(
        gpu_id=gpu_id,
        arena_bytes=arena,
        global_steering_bytes=global_steering,
        worker_bytes=worker_bytes,
        global_steering_mode=fields.get("global_steering_mode", "none"),
        free_bytes=free,
        total_bytes=total,
        safety_ratio=safety_ratio,
        recommended_workers=recommended,
        configured_workers=configured_workers,
        cov_window_batch=int(fields["cov_window_batch"]),
        max_nr=int(fields["max_nr"]),
        max_nwin=int(fields["max_nwin"]),
        max_nsamp=int(fields["max_nsamp"]),
        max_selected_bins=int(fields["max_selected_bins"]),
        max_chunk_bins=int(fields["max_chunk_bins"]),
    )


def _parse_plan_line(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if not line.startswith("FASTBF_PLAN\t"):
            continue
        fields: dict[str, str] = {}
        for item in line.split("\t")[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            fields[key] = value
        required = {
            "arena_bytes",
            "free_bytes",
            "total_bytes",
            "cov_window_batch",
            "max_nr",
            "max_nwin",
            "max_nsamp",
            "max_selected_bins",
            "max_chunk_bins",
        }
        missing = required - fields.keys()
        if missing:
            raise RuntimeError(f"native plan output missing fields: {sorted(missing)}")
        return fields
    raise RuntimeError("native plan output did not contain FASTBF_PLAN line")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB")
    unit = 0
    while value >= 1024.0 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(value)} {units[unit]}"
    return f"{value:.2f} {units[unit]}"
