#!/usr/bin/env python3
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

NETWORK = "SY"
LOCATION = "00"
COMPONENT = "BHZ"
DATE_STEM = "2020.001"


def main() -> int:
    generate_synthetic_sac(DATA_DIR)
    print(f"synthetic SAC data: {DATA_DIR}")
    return 0


def generate_synthetic_sac(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = data_dir / f"{NETWORK}.STA01.{LOCATION}.{COMPONENT}.{DATE_STEM}.sac"
    if expected.exists():
        print(f"[example] using existing synthetic SAC files under {data_dir}", flush=True)
        return

    print(f"[example] generating synthetic SAC files under {data_dir}", flush=True)
    rng = np.random.default_rng(2606)
    dt = 0.2
    npts = 9000
    t = np.arange(npts, dtype=np.float64) * dt
    lat0, lon0 = 35.0, -117.0
    sx, sy = 1.6, -0.8
    freqs = (0.18, 0.24, 0.31)

    offsets = [
        (-0.18, -0.12),
        (-0.12, 0.03),
        (-0.08, 0.16),
        (-0.04, -0.05),
        (0.00, 0.11),
        (0.04, -0.17),
        (0.08, 0.00),
        (0.12, 0.14),
        (0.16, -0.08),
        (-0.16, 0.10),
        (-0.10, -0.18),
        (0.02, 0.20),
        (0.10, -0.22),
        (0.18, 0.04),
        (-0.22, -0.02),
        (0.22, 0.18),
    ]

    for idx, (rx_deg, ry_deg) in enumerate(offsets, start=1):
        station = f"STA{idx:02d}"
        lat = lat0 + ry_deg
        lon = lon0 + rx_deg
        tau = rx_deg * sx + ry_deg * sy
        data = np.zeros_like(t)
        for amp, freq in zip((1.0, 0.75, 0.55), freqs):
            data += amp * np.sin(2.0 * np.pi * freq * (t - tau))
        data += 0.08 * rng.standard_normal(npts)
        path = data_dir / f"{NETWORK}.{station}.{LOCATION}.{COMPONENT}.{DATE_STEM}.sac"
        write_sac(path, data.astype("<f4"), dt=dt, lat=lat, lon=lon, ele=1200.0 + idx)


def write_sac(path: Path, data: np.ndarray, *, dt: float, lat: float, lon: float, ele: float) -> None:
    undefined_f = -12345.0
    undefined_i = -12345
    floats = [undefined_f] * 70
    ints = [undefined_i] * 40
    chars = bytearray(b" " * 192)

    floats[0] = float(dt)
    floats[5] = 0.0
    floats[6] = float(dt * (len(data) - 1))
    floats[31] = float(lat)
    floats[32] = float(lon)
    floats[33] = float(ele)
    ints[6] = 6
    ints[9] = int(len(data))
    ints[15] = 1
    ints[35] = 1

    header = struct.pack("<70f40i192s", *floats, *ints, bytes(chars))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.asarray(data, dtype="<f4").tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
