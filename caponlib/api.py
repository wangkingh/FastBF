from __future__ import annotations

import logging
import os
import shutil
from importlib import resources
from pathlib import Path

from .config_parser import parse_and_validate, print_array_summary, logger as cfg_logger


def configure_logging(
    *,
    verbose: bool = False,
    level: str | None = None,
    log_file_path: str | Path | None = None,
) -> None:
    resolved_level = logging.INFO if verbose else logging.WARNING
    if level:
        resolved_level = logging._nameToLevel.get(level.upper(), resolved_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file_path is not None:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved_level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def inventory(config_path: str | Path):
    """Build the reusable SAC inventory for a FastBF config."""
    from .inventory import build_inventory

    return build_inventory(config_path)


def prepare(config_path: str | Path):
    """Prepare the reusable FastBF workspace without running CUDA."""
    from .inventory import prepare_workspace

    return prepare_workspace(config_path)


def doctor(config_path: str | Path | None = None) -> list[tuple[str, str]]:
    """Return a compact environment/config report."""
    rows: list[tuple[str, str]] = []
    for module_name in ("numpy", "pandas"):
        rows.append((module_name, _module_status(module_name)))
    rows.append(("nvcc", _nvcc_status()))
    try:
        from .system import executable_report

        rows.extend(executable_report())
    except Exception as exc:
        rows.append(("fastbf_native", f"FAILED ({exc})"))

    if config_path is not None:
        try:
            arrays, time_filter, quality_filter, window, slow, bands, device, output, beamforming = parse_and_validate(config_path)
            rows.append(("config", f"OK ({Path(config_path).expanduser().resolve()})"))
            rows.append(("seisarrays", str(len(arrays))))
            rows.append(("time_start", str(time_filter["time_start"])))
            rows.append(("time_end", str(time_filter["time_end"])))
            rows.append(("time_list", str(time_filter["time_list"])))
            rows.append(("min_stations", str(quality_filter["min_stations"])))
            rows.append(("output_dir", str(output["dir"])))
            rows.append(("output_format", str(output["format"])))
            rows.append(("backend", str(beamforming["backend"])))
            rows.append(("method", str(beamforming["method"])))
            rows.append(("mode", str(beamforming["mode"])))
            if beamforming["mode"] == "grouped":
                rows.append(("group_windows", str(beamforming["group_windows"])))
            rows.append(("gpu_ids", ",".join(str(item) for item in device["ids"])))
            rows.append(("cpu_worker_num", str(device["cpu_worker_num"])))
            rows.append(("bands", str(len(bands))))
            rows.append(("win_len", str(window["win_len"])))
            rows.append(("shift_len", str(window["shift_len"])))
            rows.append(("slowness", f"+/-{slow['s_range']} step {slow['s_res']}"))
        except Exception as exc:
            rows.append(("config", f"FAILED ({exc})"))
    return rows


def generate_template_config(output_path: str | Path = "fastbf.ini", *, force: bool = False) -> Path:
    target = Path(output_path).expanduser().resolve()
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing config: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with resources.as_file(resources.files("caponlib").joinpath("template.ini")) as template:
        shutil.copy2(template, target)
    return target


def _module_status(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return f"NOT FOUND ({exc})"
    version = getattr(module, "__version__", "")
    return f"OK {version}".strip()


def _nvcc_status() -> str:
    roots: list[Path] = []
    explicit = os.environ.get("FASTBF_CUDA_HOME")
    if explicit:
        roots.append(Path(explicit))

    roots.extend(sorted(Path("/usr/local").glob("cuda-*"), key=_cuda_version_key, reverse=True))
    roots.append(Path("/usr/local/cuda"))

    for name in ("CUDA_HOME",):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))

    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        candidate = root / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)

    path = shutil.which("nvcc")
    if path:
        return path
    return "NOT FOUND"


def _cuda_version_key(path: Path) -> tuple[int, ...]:
    import re

    match = re.search(r"(\d+(?:\.\d+)*)$", path.name)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))
