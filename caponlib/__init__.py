"""FastBF Python package."""

from .config_parser import parse_and_validate, print_array_summary

__all__ = [
    "parse_and_validate",
    "print_array_summary",
    "inventory",
    "prepare",
    "organise_sac_files",
    "run_native",
]


def __getattr__(name: str):
    if name in {"inventory", "prepare"}:
        from .api import inventory, prepare

        return {"inventory": inventory, "prepare": prepare}[name]

    if name == "organise_sac_files":
        from .source_scanner import organise_sac_files

        return organise_sac_files

    if name == "run_native":
        from .native_runner import run_native

        return run_native

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
