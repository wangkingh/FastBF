# FastBF

[中文主文档](README.md) | [Usage Guardrails](docs/USAGE_GUARDRAILS.md) | [Config Examples](configs/README.md) | [Project Layout](docs/PROJECT_LAYOUT.md)

The v2606 refactor, documentation cleanup, and release packaging for this
project were assisted by OpenAI Codex / GPT.

FastBF is a CUDA-native beamforming pipeline for SAC array data. Python handles
configuration parsing, SAC inventory, NSL IDs, workspace preparation, GPU memory
planning, and worker dispatch. The CUDA/C++ backend handles SAC reading,
windowing, FFT, covariance accumulation, Capon/Bartlett scanning, and result
output.

The legacy GitHub implementation is preserved on the `legacy/pre-v2606` branch
and the `legacy-pre-v2606` tag. The current `main` branch is the v2606
CUDA-native rewrite.

The supported target is Linux or WSL with NVIDIA CUDA. Native Windows support
is currently retired and will be revisited after the Linux/WSL pipeline is
stable.

## Get The Code

Clone the repository or unpack a release archive, then enter the project root:

```bash
git clone https://github.com/wangkingh/fastbf.git FastBF
cd FastBF

# Or unpack a release archive.
tar -xf FastBF*.tar.gz
cd FastBF*
```

The project root should contain `README.md`, `Makefile`, `caponlib/`,
`native/`, `configs/`, and `example/`.

## Requirements

- Python 3.10 or newer.
- Python dependencies: `numpy`, `pandas`, `matplotlib`.
- NVIDIA CUDA Toolkit and a CUDA-capable GPU.
- GNU Make plus a C/CUDA compiler toolchain.
- Linux or WSL.

To create a local conda environment:

```bash
conda env create -f environment.yml
conda activate fastbf
```

If Anaconda/Miniconda is installed in WSL but `conda` is not available in the
current shell, initialize or source conda first:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
# Or source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

CUDA architecture is auto-detected. If detection fails, override it explicitly:

```bash
python install_fastbf.py --arch 89
```

## Install

From the project root:

```bash
make install
fastbf doctor
```

`make install` calls `install_fastbf.py`, auto-detects the CUDA architecture,
builds both native backends, stages `fastbf_native` and `fastbf_native_power`
into the Python package, and installs the editable `fastbf` command.

If the same conda environment had an older FastBF checkout installed before,
run `make install` again from this checkout so the `fastbf` command and native
binaries point to the current version.

If you only want to build the native binaries without installing the Python
package:

```bash
make build
```

Native binaries are written to `native/fastbf_native`,
`native/fastbf_native_power`, `bin/fastbf_native`, and
`bin/fastbf_native_power`; package-staged binaries are written under
`caponlib/bin/<platform>/`.

If you come back later and forget the full workflow, run:

```bash
fastbf guide --lang en
# Or from the source root.
make guide
```

For a slightly longer checklist, see
[docs/USAGE_GUARDRAILS.md](docs/USAGE_GUARDRAILS.md).

## Command Styles

FastBF can be run in either style:

```bash
# Source-tree style; useful before installing the console command.
python -m caponlib.cli doctor config.ini
python -m caponlib.cli prepare config.ini
python -m caponlib.cli checkmem config.ini
python -m caponlib.cli run config.ini

# Packaged CLI style; available after `make install` or `pip install -e .`.
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini
fastbf run config.ini
```

Both styles call the same Python entry point. The packaged `fastbf` command is
shorter; `python -m caponlib.cli ...` is convenient when debugging directly from
a source checkout.

## Workflow

FastBF is organized around a lightweight reusable workspace:

```bash
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini
fastbf run config.ini
```

`prepare` scans SAC files, assigns NSL IDs, and writes per-timestamp SAC lists
plus a geometry catalog. `checkmem` calls the native backend in plan-only mode
to estimate the CUDA arena, steering memory, and recommended worker count.
`run` consumes prepared `saclist/*.tsv` files, splits work across configured
GPUs/workers, and launches `fastbf_native`.

To inspect the native commands without running CUDA:

```bash
fastbf run config.ini --dry-run --serial --limit 1
```

## Quick Start

Create a starter configuration:

```bash
fastbf init -o config.ini
```

Edit paths, time filters, GPU IDs, and the output directory, then run:

```bash
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini --safety 0.8
fastbf run config.ini
```

`--safety 0.8` means the recommendation uses only 80% of currently free GPU
memory, leaving room for CUDA context/handle overhead and normal system
variation.

## Bundled Example

The repository includes a self-contained synthetic SAC smoke test. Its purpose
is not to provide a scientific example data set. It lets a new user verify the
Python CLI, native CUDA backend, workspace preparation, and plotting path
without any local seismic data.

After installation, run it in the standard CLI style:

```bash
cd example
python make_synthetic_sac.py
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini --safety 0.8
fastbf run config.ini --serial --clean-output
python plot_beamforming.py
```

Here `make_synthetic_sac.py` only generates a small reproducible set of
synthetic SAC files, and `plot_beamforming.py` only plots the `fastbf run`
outputs. The actual FastBF computation is driven by the `fastbf` CLI:

```text
doctor -> prepare -> checkmem -> run -> plot
```

Expected output:

```text
example/workspace/plots/capon_bartlett_db.png
```

This example does not depend on local F-drive data, and it should not be used
to judge real-array scientific performance. It is a first smoke test after
installation: if it runs, the entry points and dependencies are basically in
place.

## Real Data And Local Fixtures

Real data are not shipped with the repository. To run your own data, generate a
starter ini and fill in your SAC root, filename pattern, station list, time
range, and output directory:

```bash
fastbf init -o real_data.ini
```

If you just installed FastBF, run the bundled `example/` first. To validate real
data, prepare the matching SAC data and station list before running commands
like:

```bash
fastbf doctor real_data.ini
fastbf prepare real_data.ini
fastbf checkmem real_data.ini --safety 0.8
fastbf run real_data.ini --dry-run --serial --limit 1
fastbf run real_data.ini
```

If a `testing/` directory exists on the author's machine, it is an ignored local
sandbox for real-data regression, references, and performance experiments. It
is not published to the GitHub repository.

## Key Configuration Fields

Most users mainly edit:

```ini
[seisarray.1]
name = x2
sac_dir = /path/to/sac/root
pattern = {home}/{*}/{*}.{station}.{*}.{component}.{YYYY}.{JJJ}.{suffix}
station_list = NONE
component_list = BHZ

[time_filter]
time_start = 2015-01-01 00:00:00
time_end = 2015-01-01 01:00:00
time_list = NONE

[quality_filter]
min_stations = 10

[window]
win_len = 1800
shift_len = 1800
skip_step = NONE
taper = tukey
taper_alpha = 0.05

[slowness_grid]
s_range = 12.0
s_res = 0.2

[bands]
bands = 0.1/0.2

[beamforming]
backend = reference
method = capon
mode = stack
group_windows = 4

[device]
cpu_worker_num = 10
ids = 0
procs_per_gpu = 1

[output]
dir = ./fastbf_output
```

Configuration notes:

- Multiple `[seisarray.*]` sections are allowed. FastBF does not use FastXC's
  group logic; each array is scanned independently and folded into one
  workspace.
- `pattern` describes paths below `sac_dir`. It must start with `{home}` and
  include `{station}`, `{component}` or `{channel}`, plus date fields such as
  `{YYYY}` + `{JJJ}` or `{YYYY}` + `{MM}` + `{DD}`.
- Common wildcards: `{*}` matches any free-form segment; `{?}` matches a short
  non-path segment. Repeated field names must resolve to the same value.
- `station_list = NONE` disables station filtering. Station-list files may use
  one station code per line or the test-data style `lon lat station`.
- `time_list = NONE` uses `time_start/time_end`; otherwise the `time_list` file
  contains one `YYYY-MM-DD HH:MM:SS` timestamp per line.
- `[quality_filter].min_stations` is the minimum valid station count per
  timestamp/list.
- `[window].taper` accepts `tukey`, `hann`, or `none`. The default short Tukey
  taper (`taper_alpha = 0.05`) avoids hard window edges while staying close to
  older rectangular-window runs.
- `[bands].bands` currently accepts one target band. Run FastBF once per band
  for frequency sweeps.
- `[beamforming].backend` selects the native output family: `reference` writes
  Capon and channel-equalized Bartlett outputs, while `power` writes
  power-preserving raw / channel-deleted Bartlett outputs.
- `[output] format` is intentionally absent; the native runner owns the concrete
  output format.

## Beamforming Options

`[beamforming].backend` first selects the output family:

- `reference`: use `fastbf_native` for Capon and channel-equalized Bartlett.
- `power`: use `fastbf_native_power` for power-preserving Bartlett variants.

`[beamforming].method` depends on the selected backend:

- `reference`: `capon`, `bartlett`, or `both`.
- `power`: `raw`, `deleted`, or `both`; `deleted` also accepts the
  `channel_deleted` / `channel-deleted` aliases.

`[beamforming].mode` controls output cadence:

- `stack`: one beamforming result per SAC timestamp/list.
- `window`: one result per base window.
- `grouped`: accumulate covariance for every `group_windows` base windows and
  write one result.

Example:

```ini
[window]
win_len = 1800
shift_len = 1800
taper = tukey
taper_alpha = 0.05

[beamforming]
backend = reference
method = both
mode = grouped
group_windows = 4
```

This uses 30-minute base windows and writes Capon/Bartlett output roughly every
2 hours. With `backend = power`, the corresponding outputs are
`_bartlett_raw.npy` and/or `_bartlett_channel_deleted.npy`.

## Workspace Outputs

`fastbf prepare` and `fastbf run` maintain a lightweight workspace under
`[output].dir`:

```text
workspace/
  config.snapshot.ini
  nsl_catalog.tsv
  saclist/
    2015-01-01-00-00.tsv
  native/
    geometry.tsv
    saclists_gpu0_worker00.txt
  bf/
    2015-01-01-00-00.npy
  log/
    fastbf.log
    native_gpu0_worker00.log
  progress/
```

`nsl_catalog.tsv` assigns deterministic 4-digit `nsl_id` values from
`network + station + location`. Empty SAC location IDs are normalized to `00`.
Timestamp TSV columns are `timestamp`, `nsl_id`, `network`, `station`,
`location`, `component`, `sac_path`, and `array`.

The native backend currently writes temporary `.npy` outputs; these are expected
to be replaced by a FastBF project binary format.

## Native Backend

Python prepares work and dispatches processes; heavy compute is handled by the
native binaries. `backend = reference` uses `fastbf_native`, and
`backend = power` uses `fastbf_native_power`. A representative native command
looks like:

```bash
fastbf_native \
  --saclist-list workspace/native/saclists_gpu0_worker00.txt \
  --geometry workspace/native/geometry.tsv \
  -O workspace/bf \
  -B 0.1/0.2 \
  -W 1800/1800 \
  --taper tukey \
  --tukey-alpha 0.05 \
  -S 12/0.2 \
  --method capon \
  --mode stack \
  -G 0
```

Useful development flags:

- `--profile`: print native stage timings.
- `--freq-chunk-bins N`: override frequency chunk size.
- `--steering-cache none|auto|host|device`: experimental steering cache mode.
- `--plan-only`: estimate memory without running beamforming.

## Project Layout

```text
caponlib/          Python CLI, config parsing, inventory, checkmem, dispatch
native/            CUDA/C++ native backend
example/           self-contained synthetic SAC installation smoke test
configs/           reusable configuration examples
docs/              project layout and development notes
tools/             plotting and post-processing helpers
bin/               locally built native binaries
```

FastBF borrows FastXC's lightweight outer inventory/prepare discipline, but it
does not create FastXC middle-stage structures such as `path_plan/`,
`spack_by_timestamp/`, `xcache/`, `sourcepack/`, or `stack/`.

The current main pipeline has removed the old Python `SeisHandler`,
`sac_refactor.py`, and `cfk`/`ias_capon` execution paths. The old
implementation is kept in the `legacy/pre-v2606` branch and the
`legacy-pre-v2606` tag for comparisons.
