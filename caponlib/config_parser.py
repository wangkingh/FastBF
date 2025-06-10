from __future__ import annotations
import sys
import logging
import configparser
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional, Any, Union

"""
Robust INI configuration parser/validator for FastBF.

* Compatible with Python 3.7+ (thanks to `from __future__ import annotations`).
* Sections expected: [general], [array], [window], [slowness_grid], [bandpass],
  [device], [output], [logging].
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


def path_exists(p: Union[str, Path], what: str) -> Path:
    """Return *p* as resolved Path if it exists, else abort."""
    p = Path(p).expanduser().resolve()
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
    return cfg

# ============================================================================
# Individual section parsers
# ============================================================================

# --- [array] ----------------------------------------------------------------
def maybe_path(s: str, label: str) -> str:
    """
    Return a path string that downstream code can recognise, **preserving the
    literal sentinel 'NONE'** when requested.

    * ``""``      → 依旧表示“配置里留空”
    * ``"NONE"``  → 原样返回 'NONE'（或改成统一常量），让调用侧自行判断
    * 其它内容    → 校验后返回合法绝对路径
    """
    s_strip = s.strip()

    # 1. 显式写了 NONE：直接返回大写常量（或项目统一的 _NONE_MARKER）
    if s_strip.upper() == "NONE":
        return "NONE"

    # 2. 否则按正常文件路径处理
    return str(path_exists(s_strip, label))

def parse_array(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    """Parse [array] section and allow time_list / station_list = NONE / empty."""
    sec = "array"
    required = ("sac_dir", "pattern", "time_start", "time_end","component_list")
    for key in required:
        if not cfg.has_option(sec, key):
            abort(f"[{sec}] missing required key: {key}")

    sac_dir      = cfg.get(sec, "sac_dir")
    pattern      = cfg.get(sec, "pattern")
    time_start   = cfg.get(sec, "time_start")
    time_end     = cfg.get(sec, "time_end")
    time_list    = cfg.get(sec, "time_list",    fallback="")
    component_list = cfg.get(sec, "component_list").strip()
    station_list = cfg.get(sec, "station_list", fallback="")

    name = cfg.get(sec, "name", fallback=cfg.get("general", "project", fallback="array"))

    sac_path = path_exists(sac_dir, "SAC directory")
    check_timefmt(time_start, f"{sec}:time_start")
    check_timefmt(time_end,   f"{sec}:time_end")

    return {
        "name": name,
        "sac_dir": str(sac_path),
        "pattern": pattern,
        "time_start": time_start,
        "time_end": time_end,
        "time_list": maybe_path(time_list, "time_list file"),
        "component_list":component_list,
        "station_list": maybe_path(station_list, "station_list file"),
    }

# --- [window] ---------------------------------------------------------------

def parse_window(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "window"
    must_int  = ("win_len", "stack_size", "min_stations")
    must_float= ("hop_ratio",)
    
    for key in (*must_int, *must_float):
        if not cfg.has_option(sec, key):
            abort(f"[{sec}] missing required key: {key}")

    geti = cfg.getint; getf = cfg.getfloat; getb = cfg.getboolean
    win_len    = geti(sec, "win_len")
    hop_ratio    = getf(sec, "hop_ratio")
    stack_size   = geti(sec, "stack_size")
    min_stations = geti(sec, "min_stations")

    # 解析 skip_step，禁止留空，只能是 NONE 或列表
    skip_raw = cfg.get(sec, "skip_step").strip()
    if not skip_raw:
        abort("[window] skip_step cannot be empty; use 'NONE' or a list")

    if skip_raw.upper() == "NONE":
        skip_step: Union[str, List[int]] = []     # sentinel，交由上层判断
    else:
        skip_step = parse_list_of_ints(skip_raw)
        if not skip_step:
            abort("[window] skip_step list cannot be empty")
        if any(x <= 0 for x in skip_step):
            abort("[window] skip_step elements must be positive integers")
        
    if win_len <= 0:
        abort("[window] win_len must be positive")
    if not (0.0 < hop_ratio <= 1.0):
        abort("[window] hop_ratio must be in (0,1]")
    if stack_size < 1:
        abort("[window] stack_size must be ≥ 1")
    if min_stations <= 0:
        abort("[window] min_stations must be positive")

    data = locals().copy()
    for k in (
        "cfg", "sec", "must_int", "must_float", "must_bool",
        "opt_list_int", "geti", "getf", "getb", "skip_raw","key"
    ):
        data.pop(k, None)
    return data

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
    """Return a list of (fmin, fmax) tuples from the [bandpass] section.

    Accepted INI formats:
        [bands]
        bands = 0.1/1.0, 0.8/3.0, 2.5/5.0  # preferred

    Validation applied **only within each band** (fmin > 0, fmax > 0, fmin < fmax).
    Adjacent or overlapping bands are allowed; their ordering is preserved as given.
    """

    sec = "bands"
    if not cfg.has_section(sec):
        abort(f"[{sec}] section missing")

    bands_str = cfg.get(sec, "bands", fallback="").strip()

    raw_pairs = [p.strip() for p in bands_str.split(',') if p.strip()]
    bands = [_parse_single_band(p) for p in raw_pairs]

    return bands

# --- [device] ----------------------------------------------------------------
def parse_device(cfg: configparser.ConfigParser) -> Dict[str, Any]:
    sec = "device"
    ids_raw = cfg.get(sec, "ids", fallback="")
    procs_per_gpu = cfg.getint(sec, "procs_per_gpu", fallback=-1)
    cpu_num = cfg.getint(sec, "cpu_num", fallback=-1)
    ids = parse_list_of_ints(ids_raw)
    if not ids:
        abort("[device] ids must list at least one GPU id")
    if len(set(ids)) != len(ids):
        abort("[device] ids contains duplicates")
    if procs_per_gpu <= 0 or cpu_num <= 0:
        abort("[device] procs_per_gpu and cpu_num must be positive")
    return {"ids": ids, "procs_per_gpu": procs_per_gpu, "cpu_num": cpu_num}

# --- [output] ----------------------------------------------------------------

def parse_output(cfg: configparser.ConfigParser) -> Dict[str, Path]:
    sec = "output"
    out_dir = cfg.get(sec, "dir", fallback="").strip()
    if not out_dir:
        abort("[output] dir is required")
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return {"dir": out_path}

# --- [logging] ---------------------------------------------------------------

def parse_logging(cfg: configparser.ConfigParser) -> Dict[str, str]:
    sec = "logging"
    level = cfg.get(sec, "level", fallback="INFO").upper()
    if level not in logging._nameToLevel:
        abort(f"[logging] invalid log level '{level}'")
    log_file = cfg.get(sec, "file", fallback="")
    return {"level": level, "file": log_file}

# ============================================================================
# Master parser
# ============================================================================

def parse_and_validate(path: str | Path) -> Tuple[Dict, ...]:
    cfg = load_config(path)
    parsers = (
        parse_array, parse_window, parse_slowness,
        parse_bandpass, parse_device, parse_output, parse_logging,
    )
    return tuple(p(cfg) for p in parsers)  # type: ignore[return-value]

# ============================================================================
# Utility: pretty print array summary
# ============================================================================

def print_array_summary(arr: Dict[str, Any]) -> None:
    logger.info("-" * 60)
    logger.info(" ARRAY SUMMARY")
    logger.info("%-15s: %s", "Name", arr["name"])
    logger.info("%-15s: %s", "SAC dir", arr["sac_dir"])
    logger.info("%-15s: %s → %s", "Time", arr["time_start"], arr["time_end"])
    if arr["station_list"]:
        logger.info("%-15s: %s", "Station list", arr["station_list"])
    if arr["time_list"]:
        logger.info("%-15s: %s", "Time list", arr["time_list"])
    logger.info("Pattern: %s", arr["pattern"])
    logger.info("-" * 60)

# ============================================================================
# CLI driver for ad-hoc testing
# ============================================================================

def _configure_root_logger(level: str, logfile: str) -> None:
    logging.basicConfig(
        level=logging._nameToLevel[level],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=(logfile or None),
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python config_parser.py <config.ini>")
        sys.exit(1)

    ini_file = sys.argv[1]
    try:
        # first pass – configure root logger as early as possible
        tmp_cfg = load_config(ini_file)
        log_sec = tmp_cfg["logging"] if "logging" in tmp_cfg else {}
        _configure_root_logger(log_sec.get("level", "INFO"), log_sec.get("file", ""))

        # full validation
        (general, array, window, slowness, bandpass,device, output, logging_cfg) = parse_and_validate(ini_file)

        logger.info("Configuration parsed OK.")
        print_array_summary(array)
        logger.debug("Sections sizes: %s", [len(x) for x in (
            general, array, window, slowness, bandpass, device, output, logging_cfg)])
    except RuntimeError:
        # already logged at ERROR level inside parser helpers
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        sys.exit(1)
