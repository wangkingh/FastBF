from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caponlib.config_parser import parse_and_validate


class ConfigParserTests(unittest.TestCase):
    def test_single_band_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2")

            parsed = parse_and_validate(cfg)

            self.assertEqual([(0.1, 0.2)], parsed[5])

    def test_multiple_bands_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2,0.2/0.4")

            with self.assertRaisesRegex(RuntimeError, "exactly one target band"):
                parse_and_validate(cfg)


def _write_config(root: Path, *, bands: str) -> Path:
    sac_dir = root / "sac"
    sac_dir.mkdir()
    station_list = root / "stations.txt"
    station_list.write_text("AAA\n", encoding="utf-8")
    cfg = root / "fastbf.ini"
    cfg.write_text(
        f"""
[seisarray.1]
name = test_array
sac_dir = {sac_dir}
pattern = {{home}}/{{network}}.{{station}}.{{locid}}.{{component}}.{{YYYY}}.{{JJJ}}.{{suffix}}
station_list = {station_list}
component_list = BHZ

[time_filter]
time_start = 2015-01-01 00:00:00
time_end = 2015-01-01 23:59:59
time_list = NONE

[quality_filter]
min_stations = 1

[window]
win_len = 1800
shift_len = 1800
skip_step = NONE

[slowness_grid]
s_range = 12.0
s_res = 0.2

[bands]
bands = {bands}

[beamforming]
method = capon
mode = stack

[device]
cpu_worker_num = 1
ids = 0
procs_per_gpu = 1

[output]
dir = {root / "workspace"}
""".strip(),
        encoding="utf-8",
    )
    return cfg


if __name__ == "__main__":
    unittest.main()
