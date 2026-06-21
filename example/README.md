# FastBF Example

This directory is a self-contained installation smoke test. It generates a tiny
synthetic SAC array, runs the FastBF CLI, and writes a Capon/Bartlett dB
comparison plot.

The generated SAC files are not a scientific reference data set. They only
exist so a new user can verify the install, CLI, native backend, workspace
preparation, and plotting path without local seismic data. Real-data regression
and performance checks belong under `testing/` or in your own project configs.

The example configs set `backend = reference` explicitly and use the default
short Tukey window taper (`taper = tukey`, `taper_alpha = 0.05`). Switch to
`backend = power` with `method = raw`, `deleted`, or `both` when checking the
power-preserving Bartlett products.

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

## Uploaded X2 Fixture

`X2_2015_01/` contains the uploaded real-data fixture for 2015 day 001.
Use the separate configuration so the synthetic installation smoke test above
remains reproducible:

```bash
fastbf doctor config.x2.ini
fastbf prepare config.x2.ini
fastbf checkmem config.x2.ini --safety 0.8
fastbf run config.x2.ini --dry-run --serial --limit 1
fastbf run config.x2.ini
```

The fixture currently contains:

- 620 unique BHZ SAC files for `2015.001`.
- 619 files with location `01` and one file with location `02`.
- 217 unique station IDs in `stations_center.txt`.
- 208 listed stations with matching SAC files.
- 9 listed stations without a file in this fixture.
- 412 SAC stations intentionally excluded by the center-station list.

Coordinates for all 208 matching stations agree with their SAC headers. The
station list is therefore a spatial subset, not a complete inventory of the
uploaded SAC directory.
