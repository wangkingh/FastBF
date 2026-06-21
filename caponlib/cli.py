from __future__ import annotations

import argparse
import logging
import shlex
import sys

from .api import doctor, generate_template_config, inventory, prepare
from .checkmem import checkmem, format_bytes
from .native_runner import run_native


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "inventory":
        return _cmd_inventory(args)
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "checkmem":
        return _cmd_checkmem(args)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "guide":
        return _cmd_guide(args)

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastbf", description="FastBF beamforming runner")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="run native beamforming from a config file")
    run_parser.add_argument("config", help="path to config.ini")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="enable INFO logs")
    run_parser.add_argument("--dry-run", action="store_true", help="prepare native command files without running CUDA")
    run_parser.add_argument("--serial", action="store_true", help="use one worker on the first configured GPU")
    run_parser.add_argument("--limit", type=int, default=0, help="limit number of SAC lists")
    run_parser.add_argument("--profile", action="store_true", help="enable native stage timing")
    run_parser.add_argument("--clean-output", action="store_true", help="remove existing native BF outputs first")
    run_parser.add_argument("--force-prepare", action="store_true", help="rebuild inventory before running")
    run_parser.add_argument("--native-bin", help="override native fastbf binary path")
    run_parser.add_argument(
        "--steering-cache",
        default="none",
        choices=("none", "auto", "host", "device"),
        help="native steering cache mode",
    )
    run_parser.add_argument(
        "--freq-chunk-bins",
        type=int,
        default=0,
        help="override native frequency chunk size in bins",
    )

    doctor_parser = sub.add_parser("doctor", help="check Python/CUDA modules and optional config")
    doctor_parser.add_argument("config", nargs="?", help="optional config.ini")

    inventory_parser = sub.add_parser("inventory", help="scan SAC files and write reusable inventory")
    inventory_parser.add_argument("config", help="path to config.ini")

    prepare_parser = sub.add_parser("prepare", help="prepare reusable workspace without running CUDA")
    prepare_parser.add_argument("config", help="path to config.ini")

    checkmem_parser = sub.add_parser("checkmem", help="estimate GPU worker capacity for native beamforming")
    checkmem_parser.add_argument("config", help="path to config.ini")
    checkmem_parser.add_argument(
        "--safety",
        type=float,
        default=0.80,
        help="fraction of currently free GPU memory allowed for FastBF workers",
    )

    init_parser = sub.add_parser("init", help="write a starter config file")
    init_parser.add_argument("-o", "--output", default="fastbf.ini", help="output config path")
    init_parser.add_argument("-f", "--force", action="store_true", help="overwrite output if it exists")

    guide_parser = sub.add_parser("guide", help="print a short usage reminder")
    guide_parser.add_argument(
        "--lang",
        choices=("zh", "en"),
        default="zh",
        help="guide language",
    )
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        result = run_native(
            args.config,
            rebuild=args.force_prepare,
            dry_run=args.dry_run,
            serial=args.serial,
            limit=args.limit,
            clean_output=args.clean_output,
            profile=args.profile,
            steering_cache=args.steering_cache,
            freq_chunk_bins=args.freq_chunk_bins,
            native_binary=args.native_bin,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"Output: {result.output_dir}")
    print(f"Native: {result.native_dir}")
    print(f"Geometry: {result.geometry_path}")
    print(
        "Run: "
        f"{result.sac_list_count} SAC list(s), "
        f"{result.worker_count} worker(s), "
        f"backend={result.backend}, "
        f"method={result.method}, "
        f"mode={result.mode}"
    )
    if result.dry_run or args.verbose:
        for idx, cmd in enumerate(result.commands, start=1):
            print(f"Command {idx}: {shlex.join(cmd)}")
    for worker in result.workers:
        status = "DRY" if result.dry_run else "OK"
        print(
            f"  worker gpu={worker.gpu_id} index={worker.worker_index}: "
            f"{worker.sac_list_count} SAC list(s), log={worker.log_path} [{status}]"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    rows = doctor(args.config)
    failed = False
    for key, value in rows:
        print(f"{key}: {value}")
        if key == "config" and value.startswith("FAILED"):
            failed = True
    return 1 if failed else 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    try:
        result = inventory(args.config)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"Inventory: {result.workspace_dir}")
    print(f"  saclist_dir: {result.sac_list_dir}")
    print(f"  nsl_catalog: {result.nsl_catalog}")
    print(
        "  counts: "
        f"{result.sac_row_count} SAC row(s), "
        f"{result.timestamp_count} timestamp(s), "
        f"{result.station_count} station(s), "
        f"{result.component_count} component(s)"
    )
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    try:
        result = prepare(args.config)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    inv = result.inventory
    print(f"Prepared workspace: {result.workspace_dir}")
    print(f"  saclist_dir: {inv.sac_list_dir}")
    print(f"  nsl_catalog: {inv.nsl_catalog}")
    print(
        "  counts: "
        f"{inv.sac_row_count} SAC row(s), "
        f"{inv.timestamp_count} timestamp(s), "
        f"{inv.sac_list_count} SAC list(s)"
    )
    return 0


def _cmd_checkmem(args: argparse.Namespace) -> int:
    try:
        result = checkmem(args.config, safety_ratio=args.safety)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"Workspace: {result.workspace_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"SAC lists: {result.sac_list_count}")
    print(f"Native: {result.native_binary}")
    print("Note: recommendation applies the safety factor to leave room for CUDA context/handle overhead.")
    for plan in result.gpu_plans:
        status = "OK"
        if plan.configured_workers > plan.recommended_workers:
            status = "TOO HIGH"
        elif plan.configured_workers < plan.recommended_workers:
            status = "ROOM"
        print(
            f"GPU {plan.gpu_id}: "
            f"arena={format_bytes(plan.arena_bytes)}, "
            f"global_steering={format_bytes(plan.global_steering_bytes)}"
            f"({plan.global_steering_mode}), "
            f"worker={format_bytes(plan.worker_bytes)}, "
            f"free={format_bytes(plan.free_bytes)}, "
            f"total={format_bytes(plan.total_bytes)}, "
            f"safety={plan.safety_ratio:.2f}, "
            f"recommended={plan.recommended_workers}, "
            f"configured={plan.configured_workers} [{status}]"
        )
        print(
            "  shape: "
            f"max_nr={plan.max_nr}, "
            f"max_nwin={plan.max_nwin}, "
            f"max_nsamp={plan.max_nsamp}, "
            f"max_selected_bins={plan.max_selected_bins}, "
            f"max_chunk_bins={plan.max_chunk_bins}, "
            f"cov_window_batch={plan.cov_window_batch}"
        )
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        path = generate_template_config(args.output, force=args.force)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Wrote {path}")
    return 0


def _cmd_guide(args: argparse.Namespace) -> int:
    from .guide import guide_text

    print(guide_text(args.lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
