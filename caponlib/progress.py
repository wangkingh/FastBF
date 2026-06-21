from __future__ import annotations


def progress(message: str) -> None:
    """Print a user-facing progress line immediately."""
    print(f"[fastbf] {message}", flush=True)
