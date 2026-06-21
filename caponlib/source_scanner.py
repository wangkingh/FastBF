from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .arrays import SeisArray
from .progress import progress

logger = logging.getLogger(__name__)

NONE_MARKER = "NONE"


def organise_sac_files(
    array: Mapping[str, object],
    cpu_count: int = 1,
) -> dict[datetime, dict[str, list]]:
    """Match/filter one FastBF seisarray source and group rows by timestamp."""
    sac_dir = _none_or_text(array.get("sac_dir"))
    if sac_dir is None:
        logger.warning("Skipping array because sac_dir is NONE")
        return {}

    component_list = _component_list(array.get("component_list"))
    criteria = _build_criteria(
        station_list=read_station_list(_none_or_text(array.get("station_list") or array.get("sta_list"))),
        time_list=read_time_list(_none_or_text(array.get("time_list"))),
        component_list=component_list,
        time_start=_parse_time(array["time_start"]),
        time_end=_parse_time(array["time_end"]),
    )
    station_filter = criteria.get("station")
    station_count = len(station_filter["value"]) if station_filter else 0
    time_filter = criteria.get("time", {})
    progress(
        "source scan criteria: "
        f"components={','.join(component_list)}, "
        f"stations={station_count if station_count else 'ALL'}, "
        f"time={time_filter.get('type', 'NONE')}"
    )

    seis_array = SeisArray(sac_dir, str(array["pattern"]))
    seis_array.match(threads=cpu_count)
    seis_array.filter(criteria, threads=cpu_count)
    seis_array.group(labels=["time"], sort_labels=["station", "component"], filtered=True)
    group_count = len(seis_array.files_group or {})
    progress(f"grouped filtered files into {group_count:,} timestamp group(s)")
    return seis_array.files_group or {}


def read_station_list(path: str | None) -> list[str]:
    if path is None:
        return []
    return [_station_id_from_line(line) for line in _read_text_lines(path)]


def read_time_list(path: str | None) -> list[datetime]:
    if path is None:
        return []
    values: list[datetime] = []
    for idx, line in enumerate(_read_text_lines(path), start=1):
        try:
            values.append(_parse_time(line))
        except Exception as exc:
            raise ValueError(f"Invalid timestamp '{line}' in {path} line {idx}") from exc
    return values


def _build_criteria(
    *,
    station_list: list[str],
    time_list: list[datetime],
    component_list: list[str],
    time_start: datetime,
    time_end: datetime,
) -> dict[str, dict[str, object]]:
    criteria: dict[str, dict[str, object]] = {}
    if station_list:
        criteria["station"] = {"type": "list", "data_type": "str", "value": station_list}
    if time_list:
        criteria["time"] = {"type": "list", "data_type": "datetime", "value": time_list}
    else:
        criteria["time"] = {
            "type": "range",
            "data_type": "datetime",
            "value": [time_start, time_end],
        }
    criteria["component"] = {
        "type": "list",
        "data_type": "str",
        "value": component_list,
    }
    return criteria


def _component_list(value: object) -> list[str]:
    components = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not components:
        raise ValueError("component_list cannot be empty")
    return components


def _none_or_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.upper() == NONE_MARKER:
        return None
    return text


def _read_text_lines(path: str) -> list[str]:
    lines: list[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _station_id_from_line(line: str) -> str:
    parts = line.split()
    if len(parts) >= 3 and _is_float(parts[0]) and _is_float(parts[1]):
        return parts[2]
    return parts[0] if parts else ""


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_time(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text)
