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

    def test_power_backend_accepts_power_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2", backend="power", method="deleted")

            parsed = parse_and_validate(cfg)

            self.assertEqual("power", parsed[8]["backend"])
            self.assertEqual("deleted", parsed[8]["method"])

    def test_power_backend_rejects_reference_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2", backend="power", method="capon")

            with self.assertRaisesRegex(RuntimeError, "raw', 'deleted', or 'both"):
                parse_and_validate(cfg)

    def test_reference_backend_rejects_power_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2", backend="reference", method="raw")

            with self.assertRaisesRegex(RuntimeError, "capon', 'bartlett', or 'both"):
                parse_and_validate(cfg)

    def test_window_taper_defaults_to_short_tukey(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2")

            parsed = parse_and_validate(cfg)

            self.assertEqual("tukey", parsed[3]["taper"])
            self.assertEqual(0.05, parsed[3]["taper_alpha"])

    def test_window_taper_accepts_tukey_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2", taper="tukey", taper_alpha=0.05)

            parsed = parse_and_validate(cfg)

            self.assertEqual("tukey", parsed[3]["taper"])
            self.assertEqual(0.05, parsed[3]["taper_alpha"])

    def test_window_taper_rejects_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bands="0.1/0.2", taper="mystery")

            with self.assertRaisesRegex(RuntimeError, "taper must"):
                parse_and_validate(cfg)


def _write_config(
    root: Path,
    *,
    bands: str,
    backend: str = "reference",
    method: str = "capon",
    taper: str | None = None,
    taper_alpha: float | None = None,
) -> Path:
    sac_dir = root / "sac"
    sac_dir.mkdir()
    station_list = root / "stations.txt"
    station_list.write_text("AAA\n", encoding="utf-8")
    cfg = root / "fastbf.ini"
    taper_lines = ""
    if taper is not None:
        taper_lines += f"\ntaper = {taper}"
    if taper_alpha is not None:
        taper_lines += f"\ntaper_alpha = {taper_alpha:g}"
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
{taper_lines}

[slowness_grid]
s_range = 12.0
s_res = 0.2

[bands]
bands = {bands}

[beamforming]
backend = {backend}
method = {method}
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
