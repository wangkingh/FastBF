#!/usr/bin/env python
"""Linux/WSL FastBF installer.

Typical use:

    python install_fastbf.py

The script prepares the CUDA environment, builds the native backend, stages it
inside the Python package, installs the CLI, and verifies executable discovery.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUPPORTED_SYSTEMS = {"Linux"}


def main(argv: list[str] | None = None) -> int:
    ensure_supported_platform()
    args = parse_args(argv)
    env = os.environ.copy()

    cuda_home = Path(args.cuda_home).expanduser().resolve() if args.cuda_home else find_cuda_home()
    if cuda_home is not None:
        env["FASTBF_CUDA_HOME"] = str(cuda_home)
        env.setdefault("CUDA_HOME", str(cuda_home))
        env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")

    arches = match_supported_cuda_arches(
        normalize_arches(args.arch or env.get("FASTBF_CUDA_ARCH") or detect_cuda_arches(env)),
        env,
    )
    env["FASTBF_CUDA_ARCH"] = arches

    print_header(cuda_home, arches, args)

    if not args.no_deps:
        ensure_build_deps(env)

    if args.compile_only:
        if not args.skip_native:
            build_native_and_stage(args, env)
    else:
        if not args.skip_native:
            build_native_and_stage(args, env)

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if args.editable:
            pip_cmd.append("-e")
        pip_cmd.append(".")
        if args.no_build_isolation or args.editable:
            pip_cmd.append("--no-build-isolation")
        if args.no_deps:
            pip_cmd.append("--no-deps")
        run(pip_cmd, env=env)

    if not args.skip_smoke:
        smoke_test(env)

    print("\nFastBF install/build completed.")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and compile FastBF.")
    parser.add_argument(
        "--arch",
        help="CUDA arch list, e.g. 89, sm_89, or 75,86,89. Default: auto-detect visible GPUs.",
    )
    parser.add_argument("--cuda-home", help="CUDA Toolkit root. Overrides auto-detection.")
    parser.add_argument("--compile-only", action="store_true", help="Only build/stage the native backend.")
    parser.add_argument("--standard", dest="editable", action="store_false", help="Use pip install . instead of editable install.")
    parser.add_argument("--no-build-isolation", action="store_true", help="Pass --no-build-isolation to pip.")
    parser.add_argument("--no-deps", action="store_true", help="Do not install Python dependencies.")
    parser.add_argument("--skip-native", action="store_true", help="Skip native fastbf_native build/staging.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip import/CUDA workspace smoke test.")
    parser.set_defaults(editable=True)
    return parser.parse_args(argv)


def ensure_supported_platform() -> None:
    system = platform.system()
    if system not in SUPPORTED_SYSTEMS or os.name == "nt":
        raise SystemExit("FastBF currently supports Linux/WSL only; Windows support is retired.")


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("\n+ " + command_text(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def command_text(cmd: list[str]) -> str:
    return " ".join(quote_arg(part) for part in cmd)


def quote_arg(text: str) -> str:
    if re.search(r"\s", text):
        return f'"{text}"'
    return text


def ensure_build_deps(env: dict[str, str]) -> None:
    missing = []
    for module_name, requirement in (
        ("wheel", "wheel"),
        ("setuptools", "setuptools>=68"),
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append(requirement)
    if missing:
        run([sys.executable, "-m", "pip", "install", *missing], env=env)


def build_native_and_stage(args: argparse.Namespace, env: dict[str, str]) -> None:
    make = shutil.which("make")
    if not make:
        raise SystemExit("`make` is required to build native backends; rerun with --skip-native to skip it")
    make_cmd = [
        make,
        "-C",
        "native",
        "all",
        f"ARCH={make_arch(env['FASTBF_CUDA_ARCH'])}",
    ]
    run(make_cmd, env=env)
    run([sys.executable, "-m", "caponlib.devtools.stage_binaries"], env=env)


def find_cuda_home() -> Path | None:
    value = os.environ.get("FASTBF_CUDA_HOME")
    if value and (Path(value) / "bin" / "nvcc").exists():
        return Path(value).resolve()

    candidates: list[Path] = []
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), key=version_key, reverse=True))
    candidates.append(Path("/usr/local/cuda"))

    for name in ("CUDA_HOME",):
        value = os.environ.get(name)
        if value and (Path(value) / "bin" / "nvcc").exists():
            candidates.append(Path(value))

    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            return candidate.resolve()

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return None


def version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)*)$", path.name)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))


def detect_cuda_arches(env: dict[str, str]) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return fallback_cuda_arches(env)

    arches: list[str] = []
    for line in out.splitlines():
        text = line.strip()
        if re.match(r"^\d+(\.\d+)?$", text):
            arch = text.replace(".", "")
            if arch not in arches:
                arches.append(arch)
    return ",".join(arches) if arches else fallback_cuda_arches(env)


def fallback_cuda_arches(env: dict[str, str]) -> str:
    """Choose common supported CUDA arches when no GPU is visible."""
    common = ["61", "70", "75", "80", "86", "89", "90"]
    supported = supported_nvcc_arches(env)
    if supported:
        selected = [arch for arch in common if arch in supported]
        if selected:
            return ",".join(selected)
        return supported[-1]
    return "89"


def supported_nvcc_arches(env: dict[str, str]) -> list[str]:
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if not nvcc:
        return []
    try:
        out = subprocess.check_output(
            [nvcc, "--list-gpu-code"],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    arches: list[str] = []
    for match in re.finditer(r"sm_(\d+)", out):
        arch = match.group(1)
        if arch not in arches:
            arches.append(arch)
    return sorted(arches, key=int)


def normalize_arches(text: str) -> str:
    arches: list[str] = []
    for item in re.split(r"[,;\s]+", text.strip()):
        if not item:
            continue
        arch = item.lower().replace("sm_", "").replace("compute_", "").replace(".", "")
        if not arch.isdigit():
            raise SystemExit(f"Invalid CUDA architecture: {item!r}")
        if arch not in arches:
            arches.append(arch)
    if not arches:
        raise SystemExit("No CUDA architecture was selected")
    return ",".join(arches)


def match_supported_cuda_arches(arches: str, env: dict[str, str]) -> str:
    supported = supported_nvcc_arches(env)
    if not supported:
        return arches

    supported_ints = sorted(int(arch) for arch in supported)
    matched: list[str] = []
    for arch in arches.split(","):
        if arch in supported:
            selected = arch
        else:
            value = int(arch)
            lower_or_equal = [item for item in supported_ints if item <= value]
            selected = str(lower_or_equal[-1] if lower_or_equal else supported_ints[-1])
            print(f"[cuda] requested sm_{arch} is not supported by this nvcc; using sm_{selected} with PTX fallback")
        if selected not in matched:
            matched.append(selected)
    return ",".join(matched)


def make_arch(arches: str) -> str:
    """Return the single sm_XX value used by the current native Makefile."""
    selected = max(int(arch) for arch in arches.split(",") if arch)
    return f"sm_{selected}"


def smoke_test(env: dict[str, str]) -> None:
    code = r"""
from caponlib.system import find_packaged_executable
native = find_packaged_executable(("fastbf_native",))
native_power = find_packaged_executable(("fastbf_native_power",))
assert native is not None, "fastbf_native not found"
assert native_power is not None, "fastbf_native_power not found"
print("fastbf_native:", native)
print("fastbf_native_power:", native_power)
print("smoke: OK")
"""
    run([sys.executable, "-c", code], env=env)


def print_header(cuda_home: Path | None, arches: str, args: argparse.Namespace) -> None:
    print("FastBF installer")
    print(f"  platform : {platform.system()} {platform.machine()}")
    print(f"  python   : {sys.executable}")
    print(f"  cuda     : {cuda_home if cuda_home else 'NOT FOUND'}")
    print(f"  arch     : {format_arches(arches)}")
    print(f"  mode     : {'compile-only' if args.compile_only else ('editable' if args.editable else 'standard')}")
    sys.stdout.flush()


def format_arches(arches: str) -> str:
    return ",".join(f"sm_{arch}" for arch in arches.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
