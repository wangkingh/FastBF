from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a FastBF profile.tsv file.")
    parser.add_argument("profile", help="path to workspace/log/profile.tsv")
    parser.add_argument("--top", type=int, default=30, help="number of rows to print")
    args = parser.parse_args()

    path = Path(args.profile).expanduser().resolve()
    totals: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            key = (row["layer"], row["stage"])
            totals[key] += float(row["elapsed_ms"])
            counts[key] += 1

    print(f"profile: {path}")
    print(f"{'layer':6s} {'stage':34s} {'total_ms':>12s} {'count':>7s} {'avg_ms':>12s}")
    print("-" * 78)
    for (layer, stage), total in sorted(totals.items(), key=lambda item: item[1], reverse=True)[: args.top]:
        count = counts[(layer, stage)]
        print(f"{layer:6s} {stage:34s} {total:12.3f} {count:7d} {total / count:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
