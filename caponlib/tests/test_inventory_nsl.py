from __future__ import annotations

import csv
import importlib
import tempfile
import unittest
from pathlib import Path


inventory = importlib.import_module("caponlib.inventory")


class InventoryNslTests(unittest.TestCase):
    def test_empty_location_is_normalized_to_00(self) -> None:
        rows = {
            "2026-01-01-00-00": [
                _row("AAA", ""),
                _row("BBB", None),
            ]
        }
        nodes = inventory._build_nsl_nodes(rows)
        inventory._attach_nsl_ids(rows, nodes)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            saclist_dir = tmp_path / "saclist"
            saclist_dir.mkdir()
            inventory._write_nsl_catalog(tmp_path / "nsl_catalog.tsv", nodes)
            inventory._write_timestamp_tsvs(rows, saclist_dir, min_stations=2)

            catalog_rows = _read_tsv(tmp_path / "nsl_catalog.tsv")
            timestamp_rows = _read_tsv(saclist_dir / "2026-01-01-00-00.tsv")

        self.assertEqual(["0001", "0002"], [row["nsl_id"] for row in catalog_rows])
        self.assertEqual(["AAA", "BBB"], [row["station"] for row in catalog_rows])
        self.assertEqual({"00"}, {row["location"] for row in catalog_rows})
        self.assertEqual({"00"}, {row["location"] for row in timestamp_rows})
        self.assertEqual(
            inventory.TIMESTAMP_TSV_FIELDS,
            list(timestamp_rows[0]),
        )

    def test_duplicate_nsl_component_in_one_timestamp_is_rejected(self) -> None:
        rows = {
            "2026-01-01-00-00": [
                _row("AAA", "", array="a", sac_path="/tmp/a.sac"),
                _row("AAA", "", array="b", sac_path="/tmp/b.sac"),
            ]
        }
        nodes = inventory._build_nsl_nodes(rows)
        inventory._attach_nsl_ids(rows, nodes)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Duplicate SAC trace"):
                inventory._write_timestamp_tsvs(rows, Path(tmp), min_stations=1)


def _row(
    station: str,
    location: object,
    *,
    array: str = "array",
    sac_path: str | None = None,
) -> dict[str, str]:
    return {
        "timestamp": "2026-01-01-00-00",
        "array": array,
        "network": "X2",
        "station": station,
        "location": inventory._normalize_location(location),
        "component": "BHZ",
        "sac_path": sac_path or f"/tmp/{station}.sac",
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


if __name__ == "__main__":
    unittest.main()
