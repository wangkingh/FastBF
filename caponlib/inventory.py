from __future__ import annotations

import csv
import math
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config_parser import parse_and_validate
from .progress import progress


TIMESTAMP_TSV_FIELDS = [
    "timestamp",
    "nsl_id",
    "network",
    "station",
    "location",
    "component",
    "sac_path",
    "array",
]
NSL_CATALOG_FIELDS = [
    "nsl_id",
    "network",
    "station",
    "location",
    "lat",
    "lon",
    "ele",
    "components",
    "representative_sac_path",
]

_SAC_HEADER_BYTES = 632
_SAC_STLA_OFFSET = 31 * 4
_SAC_STLO_OFFSET = 32 * 4
_SAC_STEL_OFFSET = 33 * 4
_SAC_NVHDR_OFFSET = 70 * 4 + 6 * 4
_SAC_UNDEFINED = -12345.0
_EMPTY_LOCATION_VALUES = {"", "--", "none", "null", "nan"}


@dataclass(frozen=True)
class InventoryResult:
    config_path: Path
    workspace_dir: Path
    sac_list_dir: Path
    nsl_catalog: Path
    config_snapshot: Path
    sac_list_count: int
    timestamp_count: int
    sac_row_count: int
    station_count: int
    component_count: int


@dataclass(frozen=True)
class PreparedWorkspace:
    config_path: Path
    workspace_dir: Path
    inventory: InventoryResult
    arrays: list[dict[str, Any]]
    time_filter: dict[str, Any]
    quality_filter: dict[str, Any]
    window: dict[str, Any]
    slow: dict[str, Any]
    bands: list[tuple[float, float]]
    device: dict[str, Any]
    output: dict[str, Any]
    beamforming: dict[str, Any]


@dataclass(frozen=True)
class _StationGeo:
    lat: float | None
    lon: float | None
    ele: float | None
    parse_ok: bool
    source_path: str = ""


@dataclass
class _NslNode:
    nsl_id: str
    network: str
    station: str
    location: str
    representative_sac_path: str
    components: set[str]
    lat: float | None = None
    lon: float | None = None
    ele: float | None = None


def build_inventory(config_path: str | Path, *, clean: bool = True) -> InventoryResult:
    """Scan SAC files and write one timestamp TSV per runnable SAC group."""
    parsed = _load_sections(config_path)
    return _build_inventory_from_sections(*parsed, clean=clean)


def prepare_workspace(config_path: str | Path, *, clean: bool = True) -> PreparedWorkspace:
    """Prepare the lightweight FastBF workspace without running CUDA."""
    parsed = _load_sections(config_path)
    (
        ini_file,
        arrays,
        time_filter,
        quality_filter,
        window,
        slow,
        bands,
        device,
        output,
        beamforming,
    ) = parsed
    inventory = _build_inventory_from_sections(*parsed, clean=clean)
    workspace = inventory.workspace_dir
    for dirname in ("bf", "bf_mat", "log", "progress", "native"):
        (workspace / dirname).mkdir(parents=True, exist_ok=True)

    return PreparedWorkspace(
        config_path=ini_file,
        workspace_dir=workspace,
        inventory=inventory,
        arrays=arrays,
        time_filter=time_filter,
        quality_filter=quality_filter,
        window=window,
        slow=slow,
        bands=bands,
        device=device,
        output=output,
        beamforming=beamforming,
    )


def ensure_prepared_workspace(
    config_path: str | Path,
    *,
    rebuild: bool = False,
) -> PreparedWorkspace:
    """Return an existing prepared workspace, rebuilding only when needed."""
    if rebuild:
        return prepare_workspace(config_path, clean=True)
    try:
        return load_prepared_workspace(config_path)
    except RuntimeError:
        return prepare_workspace(config_path, clean=True)


def load_prepared_workspace(config_path: str | Path) -> PreparedWorkspace:
    """Load the lightweight workspace that ``prepare`` already wrote."""
    parsed = _load_sections(config_path)
    (
        ini_file,
        arrays,
        time_filter,
        quality_filter,
        window,
        slow,
        bands,
        device,
        output,
        beamforming,
    ) = parsed
    workspace = Path(output["dir"]).expanduser().resolve()
    sac_list_dir = workspace / "saclist"
    nsl_catalog = workspace / "nsl_catalog.tsv"
    if not sac_list_dir.is_dir():
        raise RuntimeError(f"Prepared saclist directory not found: {sac_list_dir}")
    sac_lists = sorted(sac_list_dir.glob("*.tsv"))
    if not sac_lists:
        raise RuntimeError(f"No prepared SAC list TSV files found under {sac_list_dir}")
    if not nsl_catalog.is_file():
        raise RuntimeError(f"Prepared NSL catalog not found: {nsl_catalog}")

    summary = _summarize_existing_saclists(sac_lists)
    inventory = InventoryResult(
        config_path=ini_file,
        workspace_dir=workspace,
        sac_list_dir=sac_list_dir,
        nsl_catalog=nsl_catalog,
        config_snapshot=workspace / "config.snapshot.ini",
        sac_list_count=len(sac_lists),
        timestamp_count=len(sac_lists),
        sac_row_count=summary["sac_rows"],
        station_count=summary["stations"],
        component_count=summary["components"],
    )
    for dirname in ("bf_mat", "bf", "log", "progress", "native"):
        (workspace / dirname).mkdir(parents=True, exist_ok=True)

    return PreparedWorkspace(
        config_path=ini_file,
        workspace_dir=workspace,
        inventory=inventory,
        arrays=arrays,
        time_filter=time_filter,
        quality_filter=quality_filter,
        window=window,
        slow=slow,
        bands=bands,
        device=device,
        output=output,
        beamforming=beamforming,
    )


def _load_sections(config_path: str | Path):
    ini_file = Path(config_path).expanduser().resolve()
    try:
        arrays, time_filter, quality_filter, window, slow, bands, device, output, beamforming = parse_and_validate(ini_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config {ini_file}: {exc}") from exc
    return ini_file, arrays, time_filter, quality_filter, window, slow, bands, device, output, beamforming


def _build_inventory_from_sections(
    ini_file: Path,
    arrays: list[dict[str, Any]],
    time_filter: dict[str, Any],
    quality_filter: dict[str, Any],
    window: dict[str, Any],
    slow: dict[str, Any],
    bands: list[tuple[float, float]],
    device: dict[str, Any],
    output: dict[str, Any],
    beamforming: dict[str, Any],
    *,
    clean: bool,
) -> InventoryResult:
    del time_filter, window, slow, bands, beamforming
    workspace = Path(output["dir"]).expanduser().resolve()
    sac_list_dir = workspace / "saclist"
    progress(f"preparing workspace {workspace}")

    if clean:
        progress("cleaning previous inventory outputs")
        _clean_inventory_outputs(workspace, sac_list_dir)

    workspace.mkdir(parents=True, exist_ok=True)
    sac_list_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _write_config_snapshot(ini_file, workspace)
    progress(f"wrote config snapshot to {snapshot}")

    from .source_scanner import organise_sac_files

    rows_by_timestamp: dict[str, list[dict[str, str]]] = {}
    for array_index, array in enumerate(arrays, start=1):
        array_name = _array_label(array)
        progress(
            f"array {array_index}/{len(arrays)} {array_name}: "
            f"scanning {array.get('sac_dir', 'NONE')}"
        )
        files_group = organise_sac_files(array, cpu_count=device["cpu_worker_num"])
        array_rows = 0
        for time_key, info in files_group.items():
            timestamp = _label_from_key(time_key)
            rows = _rows_from_group_info(
                info,
                timestamp=timestamp,
                array_name=array_name,
            )
            array_rows += len(rows)
            rows_by_timestamp.setdefault(timestamp, []).extend(
                rows
            )
        progress(
            f"array {array_index}/{len(arrays)} {array_name}: "
            f"{len(files_group):,} timestamp group(s), {array_rows:,} row(s)"
        )

    grouped_rows = sum(len(rows) for rows in rows_by_timestamp.values())
    progress(f"building station catalog from {grouped_rows:,} grouped row(s)")
    nsl_nodes = _build_nsl_nodes(rows_by_timestamp)
    _attach_nsl_ids(rows_by_timestamp, nsl_nodes)
    nsl_catalog = _write_nsl_catalog(workspace / "nsl_catalog.tsv", nsl_nodes)
    progress(f"wrote station catalog: {len(nsl_nodes):,} station(s) -> {nsl_catalog}")

    min_stations = int(quality_filter["min_stations"])
    progress(
        f"writing timestamp SAC lists to {sac_list_dir} "
        f"(min_stations={min_stations})"
    )
    summary = _write_timestamp_tsvs(rows_by_timestamp, sac_list_dir, min_stations)
    progress(
        "wrote "
        f"{summary['sac_lists']:,} SAC list(s), "
        f"{summary['sac_rows']:,} SAC row(s)"
    )
    if summary["sac_lists"] == 0:
        raise RuntimeError(f"No SAC list TSV files were generated under {sac_list_dir}")

    return InventoryResult(
        config_path=ini_file,
        workspace_dir=workspace,
        sac_list_dir=sac_list_dir,
        nsl_catalog=nsl_catalog,
        config_snapshot=snapshot,
        sac_list_count=summary["sac_lists"],
        timestamp_count=summary["timestamps"],
        sac_row_count=summary["sac_rows"],
        station_count=summary["stations"],
        component_count=summary["components"],
    )


def _build_nsl_nodes(
    rows_by_timestamp: Mapping[str, list[dict[str, str]]],
) -> list[_NslNode]:
    buckets: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in _iter_rows(rows_by_timestamp):
        key = _nsl_key(row)
        bucket = buckets.setdefault(
            key,
            {
                "components": set(),
                "paths": [],
            },
        )
        component = row.get("component", "")
        if component:
            components = bucket["components"]
            assert isinstance(components, set)
            components.add(component)

        paths = bucket["paths"]
        assert isinstance(paths, list)
        paths.append(row["sac_path"])

    if len(buckets) > 9999:
        raise RuntimeError(f"FastBF NSL_ID currently supports up to 9999 stations; got {len(buckets)}")

    progress(
        f"station catalog candidates: {len(buckets):,} unique station(s); "
        "reading one representative SAC header per station"
    )
    nodes: list[_NslNode] = []
    for index, key in enumerate(sorted(buckets), start=1):
        network, station, location = key
        bucket = buckets[key]
        paths = bucket["paths"]
        components = bucket["components"]
        assert isinstance(paths, list)
        assert isinstance(components, set)
        path_values = sorted(set(str(path) for path in paths))
        representative = path_values[0]
        geo = _first_station_geo(path_values)
        lat = lon = ele = None
        if isinstance(geo, _StationGeo):
            lat, lon, ele = geo.lat, geo.lon, geo.ele
        nodes.append(
            _NslNode(
                nsl_id=f"{index:04d}",
                network=network,
                station=station,
                location=location,
                representative_sac_path=representative,
                components=set(str(component) for component in components),
                lat=lat,
                lon=lon,
                ele=ele,
            )
        )
    return nodes


def _first_station_geo(paths: Sequence[str]) -> _StationGeo | None:
    for path in paths:
        geo = _read_station_geo_from_sac_header(path)
        if geo.parse_ok and _has_station_geo(geo):
            return geo
    return None


def _attach_nsl_ids(
    rows_by_timestamp: Mapping[str, list[dict[str, str]]],
    nsl_nodes: Sequence[_NslNode],
) -> None:
    lookup = {
        (node.network, node.station, node.location): node.nsl_id
        for node in nsl_nodes
    }
    for row in _iter_rows(rows_by_timestamp):
        row["nsl_id"] = lookup[_nsl_key(row)]


def _write_nsl_catalog(path: Path, nsl_nodes: Sequence[_NslNode]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NSL_CATALOG_FIELDS, delimiter="\t")
        writer.writeheader()
        for node in nsl_nodes:
            writer.writerow(
                {
                    "nsl_id": node.nsl_id,
                    "network": node.network,
                    "station": node.station,
                    "location": node.location,
                    "lat": _float_cell(node.lat),
                    "lon": _float_cell(node.lon),
                    "ele": _float_cell(node.ele),
                    "components": ",".join(sorted(node.components, key=_component_sort_key)),
                    "representative_sac_path": node.representative_sac_path,
                }
            )
    return path


def _write_timestamp_tsvs(
    rows_by_timestamp: Mapping[str, list[dict[str, str]]],
    sac_list_dir: Path,
    min_stations: int,
) -> dict[str, int]:
    sac_rows = 0
    written = 0
    stations: set[str] = set()
    components: set[str] = set()

    for timestamp in sorted(rows_by_timestamp):
        rows = sorted(rows_by_timestamp[timestamp], key=_row_sort_key)
        if not rows or not _passes_min_station_count(rows, min_stations):
            continue
        _validate_timestamp_rows(timestamp, rows)

        path = sac_list_dir / f"{timestamp}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMESTAMP_TSV_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        written += 1
        sac_rows += len(rows)
        stations.update(row["nsl_id"] for row in rows if row.get("nsl_id"))
        components.update(row["component"] for row in rows if row.get("component"))

    return {
        "sac_lists": written,
        "timestamps": written,
        "sac_rows": sac_rows,
        "stations": len(stations),
        "components": len(components),
    }


def _summarize_existing_saclists(sac_lists: Sequence[Path]) -> dict[str, int]:
    sac_rows = 0
    stations: set[str] = set()
    components: set[str] = set()
    for path in sac_lists:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "nsl_id" not in reader.fieldnames:
                raise RuntimeError(f"{path} missing required 'nsl_id' column")
            for row in reader:
                sac_rows += 1
                nsl_id = row.get("nsl_id", "").strip()
                component = row.get("component", "").strip()
                if nsl_id:
                    stations.add(nsl_id)
                if component:
                    components.add(component)
    return {
        "sac_rows": sac_rows,
        "stations": len(stations),
        "components": len(components),
    }


def _rows_from_group_info(
    info: Mapping[str, object],
    *,
    timestamp: str,
    array_name: str,
) -> list[dict[str, str]]:
    paths = [str(path) for path in _as_list(info.get("path"))]
    components = _aligned_text_list(info.get("component"), len(paths))
    stations = _aligned_text_list(info.get("station"), len(paths))
    networks = _aligned_text_list(info.get("network"), len(paths))
    locids = _aligned_text_list(info.get("locid", info.get("location")), len(paths))

    return [
        {
            "timestamp": timestamp,
            "array": array_name,
            "station": stations[idx].strip(),
            "component": components[idx].strip(),
            "network": networks[idx].strip(),
            "location": _normalize_location(locids[idx]),
            "sac_path": paths[idx],
        }
        for idx in range(len(paths))
        if paths[idx]
    ]


def _passes_min_station_count(
    rows: Sequence[Mapping[str, str]],
    min_stations: int,
) -> bool:
    return len({row.get("nsl_id", "") for row in rows if row.get("nsl_id")}) >= min_stations


def _validate_timestamp_rows(timestamp: str, rows: Sequence[Mapping[str, str]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.get("nsl_id", ""), row.get("component", ""))
        if not key[0] or not key[1]:
            continue
        previous = seen.get(key)
        if previous is not None:
            raise RuntimeError(
                "Duplicate SAC trace for "
                f"timestamp={timestamp}, nsl_id={key[0]}, component={key[1]}: "
                f"{previous} and {row.get('sac_path', '')}"
            )
        seen[key] = row.get("sac_path", "")


def _clean_inventory_outputs(workspace: Path, sac_list_dir: Path) -> None:
    if sac_list_dir.exists():
        shutil.rmtree(sac_list_dir, ignore_errors=True)
    for stale_dir in ("manifest", "sacmanifest"):
        path = workspace / stale_dir
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for stale_file in (
        "nsl_catalog.tsv",
        "inventory.meta.json",
        "prepare.meta.json",
        "run.meta.json",
    ):
        path = workspace / stale_file
        if path.exists():
            path.unlink()


def _write_config_snapshot(ini_file: Path, workspace: Path) -> Path:
    snapshot = workspace / "config.snapshot.ini"
    try:
        if ini_file.exists() and ini_file.resolve() != snapshot.resolve():
            shutil.copy2(ini_file, snapshot)
    except OSError:
        pass
    return snapshot


def _array_label(array: Mapping[str, Any]) -> str:
    return str(array.get("name") or array.get("id") or "array")


def _label_from_key(key: object) -> str:
    """Turn a time key into the saclist filename stem."""
    if isinstance(key, (list, tuple)) and len(key) == 1:
        key = key[0]
    if isinstance(key, datetime):
        return key.strftime("%Y-%m-%d-%H-%M")
    strftime = getattr(key, "strftime", None)
    if callable(strftime):
        return str(strftime("%Y-%m-%d-%H-%M"))
    return str(key)


def _row_sort_key(row: Mapping[str, str]) -> tuple[str, tuple[int, str], str, str, str, str]:
    return (
        row.get("nsl_id", ""),
        _component_sort_key(row.get("component", "")),
        row.get("network", ""),
        row.get("station", ""),
        row.get("location", ""),
        row.get("sac_path", ""),
    )


def _nsl_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    station = row.get("station", "").strip()
    if not station:
        raise RuntimeError(f"Cannot build NSL_ID without station name: {row.get('sac_path', '')}")
    return (
        row.get("network", "").strip(),
        station,
        _normalize_location(row.get("location", row.get("locid", ""))),
    )


def _normalize_location(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in _EMPTY_LOCATION_VALUES:
        return "00"
    return text


def _iter_rows(
    rows_by_timestamp: Mapping[str, list[dict[str, str]]],
):
    for rows in rows_by_timestamp.values():
        yield from rows


def _read_station_geo_from_sac_header(path: str) -> _StationGeo:
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_SAC_HEADER_BYTES)
    except OSError:
        return _StationGeo(None, None, None, False, source_path=path)

    if len(raw) < _SAC_HEADER_BYTES:
        return _StationGeo(None, None, None, False, source_path=path)

    endian = _detect_sac_endian(raw)
    if endian is None:
        return _StationGeo(None, None, None, False, source_path=path)

    lat = _parse_optional_sac_float(_unpack_sac_float(raw, endian, _SAC_STLA_OFFSET), "lat")
    lon = _parse_optional_sac_float(_unpack_sac_float(raw, endian, _SAC_STLO_OFFSET), "lon")
    ele = _parse_optional_sac_float(_unpack_sac_float(raw, endian, _SAC_STEL_OFFSET), "ele")
    return _StationGeo(lat, lon, ele, True, source_path=path)


def _detect_sac_endian(raw: bytes) -> str | None:
    for endian in ("<", ">"):
        try:
            nvhdr = struct.unpack_from(f"{endian}i", raw, _SAC_NVHDR_OFFSET)[0]
        except struct.error:
            continue
        if 0 < nvhdr <= 20:
            return endian
    return None


def _unpack_sac_float(raw: bytes, endian: str, offset: int) -> float:
    return float(struct.unpack_from(f"{endian}f", raw, offset)[0])


def _parse_optional_sac_float(value: float, kind: str) -> float | None:
    if _is_sac_undefined(value) or not math.isfinite(value) or abs(value) > 1.0e30:
        return None
    if kind == "lat" and not -90.0 <= value <= 90.0:
        return None
    if kind == "lon" and not -360.0 <= value <= 360.0:
        return None
    return float(value)


def _is_sac_undefined(value: float) -> bool:
    return abs(value - _SAC_UNDEFINED) <= 1.0e-4


def _has_station_geo(geo: _StationGeo) -> bool:
    return geo.lat is not None or geo.lon is not None or geo.ele is not None


def _same_station_geo(left: _StationGeo, right: _StationGeo) -> bool:
    checks = (
        ("lat", 1.0e-5),
        ("lon", 1.0e-5),
        ("ele", 1.0e-2),
    )
    for attr, tolerance in checks:
        left_value = getattr(left, attr)
        right_value = getattr(right, attr)
        if left_value is None or right_value is None:
            continue
        if abs(left_value - right_value) > tolerance:
            return False
    return True


def _geo_label(geo: _StationGeo) -> str:
    return f"lat={_float_cell(geo.lat)}, lon={_float_cell(geo.lon)}, ele={_float_cell(geo.ele)}"


def _float_cell(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _aligned_text_list(value: object, length: int) -> list[str]:
    values = [str(item) for item in _as_list(value)]
    if not values:
        return [""] * length
    if len(values) >= length:
        return values[:length]
    return values + [""] * (length - len(values))


def _component_sort_key(component: str) -> tuple[int, str]:
    suffix = component[-1:].upper()
    order = {"E": 0, "N": 1, "Z": 2, "R": 3, "T": 4}
    return (order.get(suffix, 99), component)
