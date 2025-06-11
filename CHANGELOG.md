CHANGELOG – 2025.06.11  “Relative-Window Refactor”
==============================================

Breaking
--------
* **Window hierarchy rewritten**
  * `Window`  – now stores only `index`, `start_idx`, `end_idx`.
  * `WindowAbs(Window)` – extends `Window` with `start_time`, `end_time`.
  * `build_windows()` returns *relative* windows (no UTC).
  * NEW: `attach_times()` converts a list of `Window` → `WindowAbs`
    given `start_epoch` + `dt`.

Internal
--------
* **caponlib.bftask.BfTask**
  * `__post_init__`  – output file name now uses `sac_txt.stem`
    ⇒ guarantees one‐file-per-task, prevents overwrite.
  * `run()`          – calls `attach_times()` to supply absolute-time
    windows to `ias_capon()`; dry-run log unchanged.

* **run.py**
  * Switched to new `build_windows(dt, npts, …)` signature.
  * In VERBOSE mode preview windows via `attach_times()`.

Packaging
---------
* `caponlib/__init__.py` now re-exports `BfTask` from `bftask.py`
  (former `task.py`) to avoid circular import.

