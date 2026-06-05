# FastBF Changelog

## 2026-06 Native-Only Cleanup

- Removed the old Python `cfk`/`ias_capon` beamforming execution path from the
  main source tree.
- Removed the pybind/CUDA extension build remnants (`bindings/`, `cuda/`,
  root `cfk*.so`, and legacy `fastbf_bf`).
- Kept `github_fastbf_zip/` as the archived old implementation for speed and
  regression comparisons.
- Replaced the old `SeisHandler`/`sac_refactor.py` SAC scanner with the
  FastXC-style lightweight scanner modules: `patterns.py`, `files.py`,
  `arrays.py`, and `source_scanner.py`.
- The supported runtime path is now:
  `fastbf prepare` -> `fastbf checkmem` -> `fastbf run`, with Python dispatching
  the staged native `fastbf_native` executable.
