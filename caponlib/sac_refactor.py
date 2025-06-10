"""caponlib/sac_refactor.py
=================

A **self‑contained refactor** of the SAC‑file organisation utilities, keeping
all interactions with `SeisHandler.SeisArray` **unchanged** while修复ing此前
存在的致命 bug、类型不一致、硬编码常量等问题。

Key design decisions
--------------------
*   **Backwards‑compatibility**: The public behaviour stays the same. The
    magic string ``"NONE"`` is still accepted but internally normalised to
    ``None``.
*   **Single responsibility**: Each helper performs exactly one task – parsing
    config, reading lists, building criteria, grouping files, or writing list
    files.
*   **Type safety**: Rich ``typing`` annotations plus early validation to
    surface errors close to their source.
*   **Logging hygiene**: All log messages have matching placeholders and are
    consistently namespaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import logging

from pandas import Timestamp

# Third‑party – left untouched on purpose
from .SeisHandler import SeisArray

__all__ = [
    "ArrayConfig",
    "read_station_list",
    "read_time_list",
    "generate_seis_file_group",
    "organise_sac_files",
    "write_sac_lists",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_NONE_MARKER = "NONE"  # legacy sentinel kept for compatibility


def _normalise_none(value: Optional[str]) -> Optional[str]:
    """Convert legacy ``"NONE"`` strings to actual ``None``."""
    if value is None or value.upper() == _NONE_MARKER:
        return None
    return value

def _as_none_marker(s: Union[str, Path]) -> str | Path:
    # 支持直接传 "NONE"/"none" 进 ArrayConfig(...)
    if isinstance(s, str) and s.strip().upper() == "NONE":
        return _NONE_MARKER
    return s


# ---------------------------------------------------------------------------
# Dataclass for array configuration
# ---------------------------------------------------------------------------
@dataclass
class ArrayConfig:
    sac_dir: Union[str, Path]
    pattern: str
    time_start: Timestamp
    time_end: Timestamp
    component_list: List[str] = field(default_factory=list)
    time_list: Optional[str] = None
    sta_list: Optional[str] = None

    @classmethod
    def from_mapping(cls, cfg):
        sac_dir = _normalise_none(cfg.get("sac_dir")) or _NONE_MARKER
        comps = [c.strip() for c in cfg.get("component_list", "").split(",") if c.strip()]

        def _to_ts(k):
            try:
                return Timestamp(cfg[k])
            except Exception as exc:
                raise ValueError(f"{k} format error: {cfg.get(k)}") from exc

        return cls(
            sac_dir=sac_dir,
            pattern=cfg["pattern"],
            time_start=_to_ts("time_start"),
            time_end=_to_ts("time_end"),
            component_list=comps,
            time_list=_normalise_none(cfg.get("time_list")),
            sta_list=_normalise_none(cfg.get("sta_list")),
        )

    # ---------------- validation ---------------- #
    def validate(self) -> None:
        """Hard fail on invalid configuration."""
        if self.sac_dir != _NONE_MARKER and not Path(self.sac_dir).exists():
            raise FileNotFoundError(f"SAC dir not found: {self.sac_dir}")

        if not (1 <= len(self.component_list) <= 3):
            raise ValueError("component_list must contain 1–3 items")

    # ---------------- helpers ---------------- #
    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
    
    def __post_init__(self) -> None:          # ← 新增
        self.sac_dir = _as_none_marker(self.sac_dir)


# ---------------------------------------------------------------------------
# File‑list readers
# ---------------------------------------------------------------------------

def _read_text_lines(path: str) -> List[str]:
    """Return non‑blank, non‑comment lines from a text file."""
    lines: List[str] = []
    with open(path, "r", encoding="utf‑8") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def read_station_list(path: Optional[str]) -> List[str]:
    """Read station codes from *path* (or return empty list if *None*)."""
    if path is None:
        return []
    return _read_text_lines(path)


def read_time_list(path: Optional[str]) -> List[Timestamp]:
    """Read timestamps from *path*; fail fast on malformed lines."""
    if path is None:
        return []

    ts_list: List[Timestamp] = []
    for idx, line in enumerate(_read_text_lines(path), 1):
        try:
            ts_list.append(Timestamp(line))
        except Exception as exc:
            raise ValueError(
                f"Invalid timestamp '{line}' in {path} line {idx}") from exc
    return ts_list


# ---------------------------------------------------------------------------
# Core – build file group via SeisArray
# ---------------------------------------------------------------------------

def _build_criteria(
    station_list: List[str],
    time_list: List[Timestamp],
    component_list: Sequence[str],
    time_range: Tuple[Timestamp, Timestamp],
) -> Dict[str, Dict[str, object]]:
    """Translate Python lists into `SeisArray.filter` criteria format."""
    criteria: Dict[str, Dict[str, object]] = {}

    if station_list:
        criteria["station"] = {
            "type": "list", "data_type": "str", "value": station_list,
        }

    if time_list:
        criteria["time"] = {
            "type": "list", "data_type": "datetime", "value": time_list,
        }
    else:
        criteria["time"] = {
            "type": "range", "data_type": "datetime", "value": list(time_range),
        }

    criteria["component"] = {
        "type": "list", "data_type": "str", "value": list(component_list),
    }
    return criteria


def generate_seis_file_group(
    cfg: ArrayConfig,
    cpu_count: int = 1,
) -> Dict[Tuple, Dict[str, List[str]]]:
    """Wrapper around :class:`SeisArray` that returns its *files_group*."""
    if cfg.sac_dir == _NONE_MARKER:
        logger.warning("Skipping array because sac_dir is NONE")
        return {}

    cfg.validate()

    station_list = read_station_list(cfg.sta_list)
    time_list = read_time_list(cfg.time_list)
    time_range = (cfg.time_start, cfg.time_end)

    criteria = _build_criteria(station_list, time_list, cfg.component_list, time_range)

    seis_array = SeisArray(str(cfg.sac_dir), cfg.pattern)  # untouched
    seis_array.match()
    seis_array.filter(criteria, threads=cpu_count, verbose=True)

    # Dynamic grouping: one file per timestamp, files sorted by station
    seis_array.group(labels=["time"], sort_labels=["station"], filtered=True)

    return seis_array.files_group  # type: ignore[return‑value]


# ---------------------------------------------------------------------------
# Public helpers – organise & persist lists
# ---------------------------------------------------------------------------

def organise_sac_files(
    cfg: Union[ArrayConfig, Mapping[str, str]],
    cpu_count: int = 1,
) -> Dict[Tuple, Dict[str, List[str]]]:
    """High-level convenience wrapper.

    Accepts either an :class:`ArrayConfig` or a raw mapping (e.g. parsed from
    an INI file) and returns the grouped SAC file dictionary.
    """
    if not isinstance(cfg, ArrayConfig):  # back‑compat shim
        cfg = ArrayConfig.from_mapping(cfg)  # type: ignore[arg‑type]

    return generate_seis_file_group(cfg, cpu_count=cpu_count)


def _label_from_key(key: object) -> str:
    """Turn a dict key into a safe filename stem."""
    if isinstance(key, (datetime, Timestamp)):
        return key.strftime("%Y-%m-%d-%H-%M")
    return str(key)


def write_sac_lists_by_time(
    files_group: Dict[Tuple, Dict[str, List[str]]],
    output_dir: Union[str, Path],
    min_stations:int,
) -> None:
    r"""Write one *\*.txt* file per time-group, each listing SAC paths."""
    if not files_group:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for time_key, info in files_group.items():
        paths: List[str] = info.get("path", [])
        if not paths:
            logger.debug("Empty path list for key %s - skipped", time_key)
            continue
        
        if len(paths)<min_stations:
            continue
        
        filename = out / f"{_label_from_key(time_key)}.txt"
        filename.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")

        logger.debug("Wrote %d SAC paths → %s", len(paths), filename.name)
