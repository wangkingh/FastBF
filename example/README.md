# FastBF Example

This directory is a self-contained installation smoke test. It generates a tiny
synthetic SAC array, runs the FastBF CLI, and writes a Capon/Bartlett dB
comparison plot.

The generated SAC files are not a scientific reference data set. They only
exist so a new user can verify the install, CLI, native backend, workspace
preparation, and plotting path without local seismic data. Real-data regression
and performance checks belong under `testing/` or in your own project configs.

From the repository root:

```bash
make install
cd example
python make_synthetic_sac.py
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini --safety 0.8
fastbf run config.ini --serial --clean-output
python plot_beamforming.py
```

Expected output:

```text
example/workspace/plots/capon_bartlett_db.png
```

Generated files under `data/` and `workspace/` are local runtime artifacts and
are ignored by git.
