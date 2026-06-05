from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caponlib.patterns import FieldRegistry, check_pattern, match_path, parse_timestamp
from caponlib.source_scanner import organise_sac_files, read_station_list


class SourceScannerTests(unittest.TestCase):
    def test_pattern_matches_root_relative_path_and_aliases_locid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sac = root / "X2.AAA.01.BHZ.2015.001.sac"
            sac.write_text("", encoding="utf-8")
            compiled = check_pattern(
                root,
                "{home}/{network}.{station}.{locid}.{component}.{YYYY}.{JJJ}.{suffix}",
                FieldRegistry(),
            )

            fields = match_path(compiled, sac)

        assert fields is not None
        self.assertEqual("X2", fields["network"])
        self.assertEqual("AAA", fields["station"])
        self.assertEqual("01", fields["location"])
        self.assertEqual("01", fields["locid"])
        self.assertEqual("BHZ", fields["component"])
        self.assertEqual("sac", fields["suffix"])
        self.assertEqual("2015-01-01T00:00:00", parse_timestamp(fields).isoformat())

    def test_station_list_accepts_plain_or_lon_lat_station_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stations.txt"
            path.write_text("AAA\n100.0 40.0 BBB\n# comment\n\n", encoding="utf-8")

            self.assertEqual(["AAA", "BBB"], read_station_list(str(path)))

    def test_organise_sac_files_groups_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "X2.AAA.01.BHZ.2015.001.sac").write_text("", encoding="utf-8")
            (root / "X2.BBB.01.BHZ.2015.001.sac").write_text("", encoding="utf-8")
            array = {
                "sac_dir": str(root),
                "pattern": "{home}/{network}.{station}.{locid}.{component}.{YYYY}.{JJJ}.{suffix}",
                "station_list": "NONE",
                "time_list": "NONE",
                "time_start": "2015-01-01 00:00:00",
                "time_end": "2015-01-01 23:59:59",
                "component_list": "BHZ",
            }

            grouped = organise_sac_files(array, cpu_count=1)

        self.assertEqual(1, len(grouped))
        info = next(iter(grouped.values()))
        self.assertEqual(["AAA", "BBB"], info["station"])
        self.assertEqual(["01", "01"], info["location"])
        self.assertEqual(["BHZ", "BHZ"], info["component"])


if __name__ == "__main__":
    unittest.main()
