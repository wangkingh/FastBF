from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .inventory import ensure_prepared_workspace


@dataclass(frozen=True)
class NativeWorkerResult:
    gpu_id: int
    worker_index: int
    manifest_path: Path
    log_path: Path
    sac_list_count: int
    returncode: int


@dataclass(frozen=True)
class NativeRunResult:
    config_path: Path
    workspace_dir: Path
    sac_list_dir: Path
    native_dir: Path
    output_dir: Path
    geometry_path: Path
    sac_list_count: int
    worker_count: int
    backend: str
    method: str
    mode: str
    dry_run: bool
    commands: tuple[tuple[str, ...], ...]
    workers: tuple[NativeWorkerResult, ...]


def run_native(
    config_path: str | Path,
    *,
    rebuild: bool = False,
    dry_run: bool = False,
    serial: bool = False,
    limit: int = 0,
    clean_output: bool = False,
    profile: bool = False,
    steering_cache: str = "none",
    freq_chunk_bins: int = 0,
    native_binary: str | Path | None = None,
) -> NativeRunResult:
    prepared = ensure_prepared_workspace(config_path, rebuild=rebuild)
    sac_lists = sorted(prepared.inventory.sac_list_dir.glob("*.tsv"))
    if limit > 0:
        sac_lists = sac_lists[:limit]
    if not sac_lists:
        raise RuntimeError(f"No SAC list TSV files found under {prepared.inventory.sac_list_dir}")

    native_dir = prepared.workspace_dir / "native"
    log_dir = prepared.workspace_dir / "log"
    output_dir = prepared.workspace_dir / "bf"
    native_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = write_native_geometry(prepared.inventory.nsl_catalog, native_dir / "geometry.tsv")
    band = prepared.bands[0]
    window = prepared.window
    slow = prepared.slow
    beamforming = prepared.beamforming
    backend = str(beamforming.get("backend", "reference"))
    native = Path(native_binary).expanduser().resolve() if native_binary else _native_binary(backend)

    gpu_ids = [int(item) for item in prepared.device["ids"]]
    if serial:
        gpu_ids = gpu_ids[:1]
    assignments = _assign_saclists(
        sac_lists,
        gpu_ids=gpu_ids,
        procs_per_gpu=1 if serial else int(prepared.device["procs_per_gpu"]),
    )
    commands: list[tuple[str, ...]] = []
    worker_specs: list[tuple[int, int, Path, Path, list[Path], tuple[str, ...]]] = []
    for worker_index, (gpu_id, worker_saclists) in enumerate(assignments):
        if not worker_saclists:
            continue
        manifest = native_dir / f"saclists_gpu{gpu_id}_worker{worker_index:02d}.txt"
        _write_manifest(manifest, worker_saclists)
        log_path = log_dir / f"native_gpu{gpu_id}_worker{worker_index:02d}.log"
        cmd = _native_command(
            native=native,
            manifest=manifest,
            geometry=geometry_path,
            output_dir=output_dir,
            band=band,
            win_len=float(window["win_len"]),
            shift_len=float(window["shift_len"]),
            taper=str(window.get("taper", "tukey")),
            taper_alpha=float(window.get("taper_alpha", 0.05)),
            s_range=float(slow["s_range"]),
            s_res=float(slow["s_res"]),
            method=str(beamforming["method"]),
            mode=str(beamforming["mode"]),
            group_windows=int(beamforming.get("group_windows", 0)),
            gpu_id=gpu_id,
            profile=profile,
            steering_cache=steering_cache,
            freq_chunk_bins=freq_chunk_bins,
        )
        commands.append(cmd)
        worker_specs.append((gpu_id, worker_index, manifest, log_path, worker_saclists, cmd))

    if dry_run:
        return NativeRunResult(
            config_path=prepared.config_path,
            workspace_dir=prepared.workspace_dir,
            sac_list_dir=prepared.inventory.sac_list_dir,
            native_dir=native_dir,
            output_dir=output_dir,
            geometry_path=geometry_path,
            sac_list_count=len(sac_lists),
            worker_count=len(worker_specs),
            backend=backend,
            method=str(beamforming["method"]),
            mode=str(beamforming["mode"]),
            dry_run=True,
            commands=tuple(commands),
            workers=tuple(
                NativeWorkerResult(gpu, idx, manifest, log, len(items), 0)
                for gpu, idx, manifest, log, items, _cmd in worker_specs
            ),
        )

    workers = _run_workers(worker_specs)
    failed = [item for item in workers if item.returncode != 0]
    if failed:
        details = "\n".join(
            f"GPU {item.gpu_id} worker {item.worker_index} failed with code "
            f"{item.returncode}; log={item.log_path}\n{_tail(item.log_path)}"
            for item in failed
        )
        raise RuntimeError(details)

    return NativeRunResult(
        config_path=prepared.config_path,
        workspace_dir=prepared.workspace_dir,
        sac_list_dir=prepared.inventory.sac_list_dir,
        native_dir=native_dir,
        output_dir=output_dir,
        geometry_path=geometry_path,
        sac_list_count=len(sac_lists),
        worker_count=len(worker_specs),
        backend=backend,
        method=str(beamforming["method"]),
        mode=str(beamforming["mode"]),
        dry_run=False,
        commands=tuple(commands),
        workers=tuple(workers),
    )


def write_native_geometry(nsl_catalog: str | Path, output_path: str | Path) -> Path:
    catalog_path = Path(nsl_catalog).expanduser().resolve()
    rows = _read_catalog_rows(catalog_path)
    if not rows:
        raise RuntimeError(f"{catalog_path} contains no NSL rows")
    ref_lat = float(rows[0]["lat"])
    ref_lon = float(rows[0]["lon"])
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["nsl_id", "rx", "ry"])
        for row in rows:
            nsl_id = row["nsl_id"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            rx, ry = _relative_xy_degrees(ref_lat, ref_lon, lat, lon)
            writer.writerow([nsl_id, f"{rx:.9g}", f"{ry:.9g}"])
    return output


def _relative_xy_degrees(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> tuple[float, float]:
    return _relative_xy_degrees_spherical(lat0, lon0, lat1, lon1)


def _relative_xy_degrees_spherical(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> tuple[float, float]:
    try:
        return _relative_xy_degrees_vincenty(lat0, lon0, lat1, lon1)
    except Exception:
        pass
    phi1 = math.radians(lat0)
    phi2 = math.radians(lat1)
    dlam = math.radians(lon1 - lon0)
    sin_dlam = math.sin(dlam)
    cos_dlam = math.cos(dlam)
    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    sin_phi2 = math.sin(phi2)
    cos_phi2 = math.cos(phi2)
    y = sin_dlam * cos_phi2
    x = cos_phi1 * sin_phi2 - sin_phi1 * cos_phi2 * cos_dlam
    bearing = math.atan2(y, x)
    central = math.atan2(
        math.hypot(y, x),
        sin_phi1 * sin_phi2 + cos_phi1 * cos_phi2 * cos_dlam,
    )
    dist_deg = math.degrees(central)
    return dist_deg * math.sin(bearing), dist_deg * math.cos(bearing)


def _relative_xy_degrees_vincenty(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> tuple[float, float]:
    if lat0 == lat1 and lon0 == lon1:
        return 0.0, 0.0

    a = 6378137.0
    f = 1.0 / 298.257223563
    b = (1.0 - f) * a
    phi1 = math.radians(lat0)
    phi2 = math.radians(lat1)
    L = math.radians(lon1 - lon0)
    U1 = math.atan((1.0 - f) * math.tan(phi1))
    U2 = math.atan((1.0 - f) * math.tan(phi2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    lamb = L

    sin_sigma = cos_sigma = sigma = sin_alpha = cos_sq_alpha = 0.0
    cos2_sigma_m = 0.0
    for _ in range(100):
        sin_lamb = math.sin(lamb)
        cos_lamb = math.cos(lamb)
        sin_sigma = math.sqrt(
            (cosU2 * sin_lamb) ** 2
            + (cosU1 * sinU2 - sinU1 * cosU2 * cos_lamb) ** 2
        )
        if sin_sigma == 0.0:
            return 0.0, 0.0
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lamb
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lamb / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
        if cos_sq_alpha == 0.0:
            cos2_sigma_m = 0.0
        else:
            cos2_sigma_m = cos_sigma - 2.0 * sinU1 * sinU2 / cos_sq_alpha
        C = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
        prev = lamb
        lamb = L + (1.0 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (
                cos2_sigma_m
                + C * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
            )
        )
        if abs(lamb - prev) < 1e-12:
            break
    else:
        raise RuntimeError("Vincenty inverse did not converge")

    u_sq = cos_sq_alpha * (a * a - b * b) / (b * b)
    A = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    B = u_sq / 1024.0 * (
        256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq))
    )
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos2_sigma_m
            + B
            / 4.0
            * (
                cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                - B
                / 6.0
                * cos2_sigma_m
                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)
            )
        )
    )
    dist_m = b * A * (sigma - delta_sigma)
    alpha1 = math.atan2(
        cosU2 * math.sin(lamb),
        cosU1 * sinU2 - sinU1 * cosU2 * math.cos(lamb),
    )
    dist_deg = dist_m / 1000.0 * (180.0 / math.pi / 6371.0)
    return dist_deg * math.sin(alpha1), dist_deg * math.cos(alpha1)


def _read_catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"nsl_id", "lat", "lon"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{path} missing required columns: {sorted(required)}")
        rows = [row for row in reader if row.get("nsl_id", "").strip()]
    return sorted(rows, key=lambda row: row["nsl_id"].strip())


def _assign_saclists(
    sac_lists: Sequence[Path],
    *,
    gpu_ids: Sequence[int],
    procs_per_gpu: int,
) -> list[tuple[int, list[Path]]]:
    if not gpu_ids:
        raise RuntimeError("No GPU ids configured")
    if procs_per_gpu <= 0:
        raise RuntimeError("procs_per_gpu must be positive")
    workers = [(gpu_id, []) for gpu_id in gpu_ids for _ in range(procs_per_gpu)]
    for idx, sac_list in enumerate(sac_lists):
        workers[idx % len(workers)][1].append(sac_list)
    return workers


def _write_manifest(path: Path, sac_lists: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sac_list in sac_lists:
            handle.write(str(Path(sac_list).expanduser().resolve()))
            handle.write("\n")


def _native_command(
    *,
    native: Path,
    manifest: Path,
    geometry: Path,
    output_dir: Path,
    band: tuple[float, float],
    win_len: float,
    shift_len: float,
    taper: str,
    taper_alpha: float,
    s_range: float,
    s_res: float,
    method: str,
    mode: str,
    group_windows: int,
    gpu_id: int,
    profile: bool,
    steering_cache: str,
    freq_chunk_bins: int,
) -> tuple[str, ...]:
    cmd: list[str] = [
        str(native),
        "--saclist-list",
        str(manifest),
        "--geometry",
        str(geometry),
        "-O",
        str(output_dir),
        "-B",
        f"{band[0]}/{band[1]}",
        "-W",
        f"{win_len:g}/{shift_len:g}",
        "--taper",
        taper,
        "--tukey-alpha",
        f"{taper_alpha:g}",
        "-S",
        f"{s_range:g}/{s_res:g}",
        "--method",
        method,
        "--mode",
        mode,
        "-G",
        str(gpu_id),
        "--steering-cache",
        steering_cache,
    ]
    if mode == "grouped":
        cmd.extend(["--group-windows", str(group_windows)])
    if freq_chunk_bins > 0:
        cmd.extend(["--freq-chunk-bins", str(freq_chunk_bins)])
    if profile:
        cmd.append("--profile")
    return tuple(cmd)


def _run_workers(
    worker_specs: Sequence[tuple[int, int, Path, Path, list[Path], tuple[str, ...]]],
) -> list[NativeWorkerResult]:
    procs: list[tuple[int, int, Path, Path, list[Path], subprocess.Popen[str]]] = []
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    for gpu_id, worker_index, manifest, log_path, saclists, cmd in worker_specs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((gpu_id, worker_index, manifest, log_path, saclists, proc))
        log_handle.close()

    results: list[NativeWorkerResult] = []
    for gpu_id, worker_index, manifest, log_path, saclists, proc in procs:
        returncode = proc.wait()
        results.append(
            NativeWorkerResult(
                gpu_id=gpu_id,
                worker_index=worker_index,
                manifest_path=manifest,
                log_path=log_path,
                sac_list_count=len(saclists),
                returncode=returncode,
            )
        )
    return results


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


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])
