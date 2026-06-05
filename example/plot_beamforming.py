#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
TIMESTAMP_STEM = "2020-01-01-00-00"
TARGET_PLOT = WORKSPACE / "plots" / "capon_bartlett_db.png"


def main() -> int:
    plot_outputs(
        WORKSPACE / "bf" / f"{TIMESTAMP_STEM}_capon.npy",
        WORKSPACE / "bf" / f"{TIMESTAMP_STEM}_bartlett.npy",
        TARGET_PLOT,
    )
    print(f"plot: {TARGET_PLOT}")
    return 0


def plot_outputs(capon_path: Path, bartlett_path: Path, output: Path) -> None:
    if not capon_path.exists() or not bartlett_path.exists():
        raise SystemExit(
            "FastBF output was not created. Expected:\n"
            f"  {capon_path}\n"
            f"  {bartlett_path}"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting; install it or use environment.yml") from exc

    capon = display_db(capon_path)
    bartlett = display_db(bartlett_path)
    extent = [-4.0, 4.0, -4.0, 4.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    for ax, title, data in (
        (axes[0], "Capon dB", capon),
        (axes[1], "Bartlett dB", bartlett),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap="turbo",
        )
        ax.set_title(title)
        ax.set_xlabel("sx")
        ax.set_ylabel("sy")
        fig.colorbar(im, ax=ax, label="dB")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def display_db(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float64).T
    peak = np.nanmax(arr)
    if not np.isfinite(peak) or peak <= 0.0:
        raise SystemExit(f"{path} has no positive finite peak")
    return 10.0 * np.log10(np.maximum(arr / peak, np.finfo(np.float64).tiny))


if __name__ == "__main__":
    raise SystemExit(main())
