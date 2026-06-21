#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


CAPON_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2})_"
    r"(?P<group>g\d{4})_capon\.npy$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write one Capon/Bartlett comparison page per FastBF window."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent / "x2_window_workspace",
    )
    parser.add_argument("--s-range", type=float, default=12.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    output = args.output or workspace / "plots" / "capon_bartlett_windows.pdf"
    pairs = find_frame_pairs(workspace / "bf")
    write_pdf(pairs, output, s_range=args.s_range)
    print(f"pages: {len(pairs)}")
    print(f"pdf: {output}")
    return 0


def find_frame_pairs(bf_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for capon_path in sorted(bf_dir.glob("*_capon.npy")):
        match = CAPON_RE.match(capon_path.name)
        if match is None:
            continue
        stem = f"{match.group('timestamp')}_{match.group('group')}"
        bartlett_path = bf_dir / f"{stem}_bartlett.npy"
        if not bartlett_path.is_file():
            raise SystemExit(f"Missing Bartlett result for {capon_path.name}")
        pairs.append((match.group("timestamp"), capon_path, bartlett_path))
    if not pairs:
        raise SystemExit(f"No per-window Capon/Bartlett pairs found under {bf_dir}")
    return pairs


def write_pdf(
    pairs: list[tuple[str, Path, Path]],
    output: Path,
    *,
    s_range: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = [-s_range, s_range, -s_range, s_range]
    with PdfPages(output) as pdf:
        for page_number, (timestamp, capon_path, bartlett_path) in enumerate(pairs, 1):
            capon, capon_peak = load_db(capon_path, s_range)
            bartlett, bartlett_peak = load_db(bartlett_path, s_range)
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(11.7, 5.5),
                constrained_layout=True,
            )
            fig.suptitle(
                f"FastBF window {format_timestamp(timestamp)}"
                f"   ({page_number}/{len(pairs)})"
            )
            for ax, title, data, peak in (
                (axes[0], "Capon", capon, capon_peak),
                (axes[1], "Bartlett", bartlett, bartlett_peak),
            ):
                im = ax.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    aspect="equal",
                    cmap="turbo",
                )
                ax.plot(peak[0], peak[1], "wo", markersize=5, markerfacecolor="none")
                ax.set_title(f"{title} dB   peak=({peak[0]:.1f}, {peak[1]:.1f})")
                ax.set_xlabel("sx")
                ax.set_ylabel("sy")
                fig.colorbar(im, ax=ax, label="dB", shrink=0.88)
            pdf.savefig(fig, dpi=160)
            plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "FastBF per-window Capon and Bartlett beamforming"
        info["Subject"] = f"{len(pairs)} windows, slowness range +/-{s_range:g}"


def load_db(path: Path, s_range: float) -> tuple[np.ndarray, tuple[float, float]]:
    arr = np.asarray(np.load(path), dtype=np.float64).T
    peak_index = np.unravel_index(np.nanargmax(arr), arr.shape)
    peak = float(arr[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        raise SystemExit(f"{path} has no positive finite peak")
    axis = np.linspace(-s_range, s_range, arr.shape[0])
    peak_xy = (float(axis[peak_index[1]]), float(axis[peak_index[0]]))
    db = 10.0 * np.log10(np.maximum(arr / peak, np.finfo(np.float64).tiny))
    return db, peak_xy


def format_timestamp(timestamp: str) -> str:
    return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M").strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    raise SystemExit(main())
