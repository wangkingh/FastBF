from __future__ import annotations
import sys
import logging
import configparser
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Tuple, List, Any, Union

"""
Robust INI configuration parser/validator for FastBF.

* Compatible with Python 3.7+ (thanks to `from __future__ import annotations`).
* Sections expected: [time_filter], [seisarray.*], [window],
  [slowness_grid], [bands], [beamforming], [device], [output].
* The core pipeline is single-band: run FastBF once per target frequency band.
* The native runner owns the concrete on-disk result format.
* Every mandatory key is validated; errors raise `RuntimeError`, leaving exit
  strategy to caller.
"""

logger = logging.getLogger(__name__)

# ============================================================================
# Helper utilities
# ============================================================================

def abort(msg: str) -> None:
    """Log at ERROR level and raise RuntimeError."""
    logger.error(msg)
    raise RuntimeError(msg)


def config_base_dir(cfg: configparser.ConfigParser) -> Path:
    return getattr(cfg, "_fastbf_base_dir", Path.cwd())


def resolve_config_path(p: Union[str, Path], base_dir: Path | None = None) -> Path:
    """Resolve absolute paths normally and relative paths against config dir."""
    path = Path(p).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def path_exists(p: Union[str, Path], what: str, base_dir: Path | None = None) -> Path:
    """Return *p* as resolved Path if it exists, else abort."""
    p = resolve_config_path(p, base_dir)
    if not p.exists():
        abort(f"{what} not found: {p}")
    return p


def check_timefmt(ts: str, desc: str) -> None:
    try:
        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        abort(f"Invalid date format for {desc}: '{ts}' (expect YYYY-mm-dd HH:MM:SS)")


def parse_list_of_ints(txt: str) -> List[int]:
    try:
        return [int(x) for x in txt.split(',') if x.strip()]
    except ValueError as e:
        abort(f"Expected comma-separated integers, got '{txt}': {e}")

# ============================================================================
# Config file loader
# ============================================================================

def load_config(path: str | Path) -> configparser.ConfigParser:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        abort(f"Configuration file does not exist: {path}")

    cfg = configparser.ConfigParser(inline_comment_prefixes={';', '#'}, default_section=None)
    cfg.read(path, encoding="utf-8")

    if not cfg.sections():
        abort("Configuration file contains no sections")
    cfg._fastbf_base_dir = path.parent  # type: ignore[attr-defined]
    return cfg

# ============================================================================
# Individual section parsers
# ============================================================================

# --- [time_filter] / [seisarray.*] -------------------------------------------
def maybe_path(s: str, label: str, base_dir: Path | None = None) -> str:
    """
    Return a path string that downstream code can recognise, **preserving the
    literal sentinel 'NONE'** when requested.

    * ``""``      -> normalized to 'NONE'
    * ``"NONE"``  -> returned as 'NONE'
    * other text   -> validated and returned as an absolute path
    """
    s_strip = s.strip()

    if not s_strip or s_strip.upper() == "NONE":
        return "NONE"

    return str(path_exists(s_strip, label, base_dir))

def parse_time_filter(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "time_filter"
    required = ("time_start", "time_end")
    for key in required:
        if not cfg.has_option(sec, key):
            abort(f"[{sec}] missing required key: {key}")

    time_start = cfg.get(sec, "time_start").strip()
    time_end = cfg.get(sec, "time_end").strip()
    time_list = cfg.get(sec, "time_list", fallback="").strip()

    check_timefmt(time_start, f"{sec}:time_start")
    check_timefmt(time_end, f"{sec}:time_end")

    return {
        "time_start": time_start,
        "time_end": time_end,
        "time_list": maybe_path(time_list, "time_list file", config_base_dir(cfg)),
    }


def parse_seisarrays(
    cfg: configparser.ConfigParser,
    time_filter: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sections = _seisarray_sections(cfg)
    if not sections:
        abort("At least one [seisarray.*] section is required")

    base_dir = config_base_dir(cfg)
    arrays: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for sec in sections:
        required = ("sac_dir", "pattern", "component_list")
        for key in required:
            if not cfg.has_option(sec, key):
                abort(f"[{sec}] missing required key: {key}")

        array_id = sec.split(".", 1)[1]
        name = cfg.get(sec, "name", fallback=f"seisarray{array_id}").strip()
        if not name:
            abort(f"[{sec}] name cannot be empty")
        if name in seen_names:
            abort(f"Duplicate seisarray name: {name}")
        seen_names.add(name)

        component_list = cfg.get(sec, "component_list").strip()
        if not component_list:
            abort(f"[{sec}] component_list cannot be empty")

        sac_path = path_exists(cfg.get(sec, "sac_dir"), f"{sec}:sac_dir", base_dir)
        station_list = cfg.get(sec, "station_list", fallback="")
        station_list_path = maybe_path(station_list, f"{sec}:station_list file", base_dir)

        arrays.append(
            {
                "id": array_id,
                "name": name,
                "sac_dir": str(sac_path),
                "pattern": cfg.get(sec, "pattern"),
                "component_list": component_list,
                "station_list": station_list_path,
                "sta_list": station_list_path,
                "time_start": time_filter["time_start"],
                "time_end": time_filter["time_end"],
                "time_list": time_filter["time_list"],
            }
        )
    return arrays


def _seisarray_sections(cfg: configparser.ConfigParser) -> List[str]:
    sections = [sec for sec in cfg.sections() if sec.startswith("seisarray.")]

    def sort_key(section: str) -> tuple[int, str]:
        suffix = section.split(".", 1)[1]
        if suffix.isdigit():
            return (int(suffix), suffix)
        return (10**9, suffix)

    return sorted(sections, key=sort_key)

# --- [window] ---------------------------------------------------------------

def parse_window(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "window"
    must_int  = ("win_len", "shift_len")
    
    for key in must_int:
        if not cfg.has_option(sec, key):
            abort(f"[{sec}] missing required key: {key}")
    if cfg.has_option(sec, "min_stations"):
        abort("[window] min_stations moved to [quality_filter]")
    if not cfg.has_option(sec, "skip_step"):
        abort(f"[{sec}] missing required key: skip_step")

    geti = cfg.getint
    win_len    = geti(sec, "win_len")
    shift_len  = geti(sec, "shift_len")
    taper = cfg.get(sec, "taper", fallback="tukey").strip().lower()
    taper_alpha = cfg.getfloat(sec, "taper_alpha", fallback=0.05)

    # 解析 skip_step，禁止留空，只能是 NONE 或 0-based 列表
    skip_raw = cfg.get(sec, "skip_step").strip()
    if not skip_raw:
        abort("[window] skip_step cannot be empty; use 'NONE' or a list")

    if skip_raw.upper() == "NONE":
        skip_step: List[int] = []
    else:
        skip_step = parse_list_of_ints(skip_raw)
        if not skip_step:
            abort("[window] skip_step list cannot be empty")
        if any(x < 0 for x in skip_step):
            abort("[window] skip_step elements must be non-negative integers")
        
    if win_len <= 0:
        abort("[window] win_len must be positive")
    if shift_len <= 0:
        abort("[window] shift_len must be positive")
    if taper in {"hanning"}:
        taper = "hann"
    elif taper in {"rect", "rectangular", "boxcar"}:
        taper = "none"
    if taper not in {"hann", "tukey", "none"}:
        abort("[window] taper must be 'hann', 'tukey', or 'none'")
    if taper == "tukey" and not (0.0 < taper_alpha <= 1.0):
        abort("[window] taper_alpha must be in (0, 1] when taper=tukey")
    data = locals().copy()
    for k in (
        "cfg", "sec", "must_int", "geti", "skip_raw","key"
    ):
        data.pop(k, None)
    return data


# --- [quality_filter] --------------------------------------------------------

def parse_quality_filter(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "quality_filter"
    min_stations = cfg.getint(sec, "min_stations", fallback=-1)
    if min_stations <= 0:
        abort("[quality_filter] min_stations must be positive")
    return {"min_stations": min_stations}

# --- [slowness_grid] ---------------------------------------------------------

def parse_slowness(cfg: configparser.ConfigParser) -> Dict[str, float]:
    sec = "slowness_grid"
    for key in ("s_range", "s_res"):
        if not cfg.has_option(sec, key):
            abort(f"[{sec}] missing required key: {key}")
    s_range = cfg.getfloat(sec, "s_range")
    s_res = cfg.getfloat(sec, "s_res")
        
    if s_res <= 0:
        abort("[slowness_grid] s_res must be > 0")
    return {"s_range": s_range, "s_res": s_res}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_num_re = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*/\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")

def _parse_single_band(pair: str) -> Tuple[float, float]:
    """Parse one "f1/f2" pair and validate it.

    Rules:
    - f1, f2 > 0
    - f1 < f2
    - No inter‑band constraints handled here.
    """
    m = _num_re.match(pair)
    if not m:
        abort(f"[bandpass] invalid band specification '{pair}' (expected 'f1/f2')")

    f1, f2 = float(m.group(1)), float(m.group(2))
    if f1 <= 0 or f2 <= 0:
        abort(f"[bandpass] frequencies must be positive in '{pair}'")
    if f1 >= f2:
        abort(f"[bandpass] lower bound must be smaller than upper bound in '{pair}'")
    return f1, f2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_bandpass(cfg: configparser.ConfigParser) -> List[Tuple[float, float]]:
    """Return one target (fmin, fmax) tuple from the [bands] section.

    Accepted INI formats:
        [bands]
        bands = 0.1/1.0

    Multi-band sweeps should be handled by launching FastBF once per band.
    """

    sec = "bands"
    if not cfg.has_section(sec):
        abort(f"[{sec}] section missing")

    bands_str = cfg.get(sec, "bands", fallback="").strip()

    raw_pairs = [p.strip() for p in bands_str.split(',') if p.strip()]
    if len(raw_pairs) != 1:
        abort("[bands] exactly one target band is supported; run FastBF once per band")
    bands = [_parse_single_band(p) for p in raw_pairs]

    return bands

# --- [device] ----------------------------------------------------------------
def parse_device(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "device"
    ids_raw = cfg.get(sec, "ids", fallback="")
    procs_per_gpu = cfg.getint(sec, "procs_per_gpu", fallback=-1)
    cpu_worker_num = cfg.getint(sec, "cpu_worker_num", fallback=-1)
    ids = parse_list_of_ints(ids_raw)
    if not ids:
        abort("[device] ids must list at least one GPU id")
    if len(set(ids)) != len(ids):
        abort("[device] ids contains duplicates")
    if procs_per_gpu <= 0 or cpu_worker_num <= 0:
        abort("[device] procs_per_gpu and cpu_worker_num must be positive")
    return {
        "ids": ids,
        "procs_per_gpu": procs_per_gpu,
        "cpu_worker_num": cpu_worker_num,
    }

# --- [output] ----------------------------------------------------------------

def parse_output(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "output"
    out_dir = cfg.get(sec, "dir", fallback="").strip()
    if not out_dir:
        abort("[output] dir is required")
    if cfg.has_option(sec, "format"):
        abort("[output] format is controlled by the native runner; remove the format key")
    out_path = resolve_config_path(out_dir, config_base_dir(cfg))
    out_path.mkdir(parents=True, exist_ok=True)
    return {"dir": out_path, "format": "native-controlled"}


# --- [beamforming] -----------------------------------------------------------

def parse_beamforming(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "beamforming"
    backend = cfg.get(sec, "backend", fallback="reference").strip().lower()
    method = cfg.get(sec, "method", fallback="capon").strip().lower()
    mode = cfg.get(sec, "mode", fallback="stack").strip().lower()
    group_windows = cfg.getint(sec, "group_windows", fallback=0)

    if backend not in {"reference", "power"}:
        abort("[beamforming] backend must be 'reference' or 'power'")
    if backend == "reference":
        if method not in {"capon", "bartlett", "both"}:
            abort("[beamforming] method must be 'capon', 'bartlett', or 'both' when backend=reference")
    else:
        if method in {"channel_deleted", "channel-deleted"}:
            method = "deleted"
        if method not in {"raw", "deleted", "both"}:
            abort("[beamforming] method must be 'raw', 'deleted', or 'both' when backend=power")
    if mode not in {"stack", "grouped", "window"}:
        abort("[beamforming] mode must be 'stack', 'grouped', or 'window'")
    if mode == "window":
        group_windows = 1
    elif mode == "grouped":
        if group_windows <= 0:
            abort("[beamforming] mode=grouped requires positive group_windows")
    elif group_windows < 0:
        abort("[beamforming] group_windows must be non-negative")

    return {
        "backend": backend,
        "method": method,
        "mode": mode,
        "group_windows": group_windows,
    }


# ============================================================================
# Master parser
# ============================================================================

def parse_and_validate(path: str | Path) -> Tuple[Dict, ...]:
    cfg = load_config(path)
    time_filter = parse_time_filter(cfg)
    arrays = parse_seisarrays(cfg, time_filter)
    return (
        arrays,
        time_filter,
        parse_quality_filter(cfg),
        parse_window(cfg),
        parse_slowness(cfg),
        parse_bandpass(cfg),
        parse_device(cfg),
        parse_output(cfg),
        parse_beamforming(cfg),
    )

# ============================================================================
# Utility: pretty print array summary
# ============================================================================

def print_array_summary(arrays: List[Dict[str, Any]], time_filter: Dict[str, Any]) -> None:
    logger.info("-" * 60)
    logger.info(" ARRAY SUMMARY")
    logger.info("%-15s: %s -> %s", "Time", time_filter["time_start"], time_filter["time_end"])
    logger.info("%-15s: %s", "Time list", time_filter["time_list"])
    for arr in arrays:
        logger.info("[%s] %s", arr["id"], arr["name"])
        logger.info("  %-13s: %s", "SAC dir", arr["sac_dir"])
        logger.info("  %-13s: %s", "Station list", arr["station_list"])
        logger.info("  %-13s: %s", "Components", arr["component_list"])
        logger.info("  Pattern: %s", arr["pattern"])
    logger.info("-" * 60)

# ============================================================================
# CLI driver for ad-hoc testing
# ============================================================================

def _configure_root_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python config_parser.py <config.ini>")
        sys.exit(1)

    ini_file = sys.argv[1]
    try:
        _configure_root_logger()

        # full validation
        (
            arrays,
            time_filter,
            quality_filter,
            window,
            slowness,
            bands,
            device,
            output,
            beamforming,
        ) = parse_and_validate(ini_file)

        logger.info("Configuration parsed OK.")
        print_array_summary(arrays, time_filter)
        logger.debug("Sections sizes: %s", [len(x) for x in (
            arrays, time_filter, quality_filter, window, slowness, bands, device, output,
            beamforming)])
    except RuntimeError:
        # already logged at ERROR level inside parser helpers
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        sys.exit(1)
