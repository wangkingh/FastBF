# FastBF 防忘护栏

这份文档不是完整手册，而是给未来的自己留的最短路线。半年后如果忘了从哪里
开始，先运行：

```bash
fastbf guide
```

## 三种场景

### 1. 我只是想确认程序还能不能跑

用 `example/`。它会生成 synthetic SAC，只用于安装烟测，不用于科学判断。

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

目标文件：

```text
example/workspace/plots/capon_bartlett_db.png
```

### 2. 我想跑自己的真实数据

先生成模板，再改路径和参数：

```bash
fastbf init -o my.ini
```

最常改的是：

- `[seisarray.*]`：`sac_dir`、`pattern`、`station_list`、`component_list`
- `[time_filter]`：`time_start`、`time_end`、`time_list`
- `[quality_filter]`：`min_stations`
- `[window]`：`win_len`、`shift_len`、`group_windows`
- `[bands]`：当前一次只跑一个目标频带
- `[beamforming]`：`method = capon|bartlett|both`，`mode = stack|window`
- `[device]`：`ids`、`procs_per_gpu`
- `[output]`：`dir`

每次真跑前按这个顺序走：

```bash
fastbf doctor my.ini
fastbf prepare my.ini
fastbf checkmem my.ini --safety 0.8
fastbf run my.ini --dry-run --serial --limit 1
fastbf run my.ini
```

### 3. 我想跑真实数据或本地回归

真实数据不随仓库发布。用自己的 SAC 路径和 station list 生成或修改 ini：

```bash
fastbf init -o real_data.ini
fastbf doctor real_data.ini
fastbf prepare real_data.ini
fastbf checkmem real_data.ini --safety 0.8
fastbf run real_data.ini --dry-run --serial --limit 1
fastbf run real_data.ini
```

作者机器上如果存在 `testing/`，它只是 ignored 的本地沙盒，用于真实数据回归、
参考结果和性能实验，不随 GitHub 仓库发布。

## 跑完应该看哪里

FastBF 的 workspace 在 `[output].dir` 下：

```text
[output].dir/
  config.snapshot.ini      本次运行使用的配置快照
  nsl_catalog.tsv          NSL ID 和台站几何 catalog
  saclist/*.tsv            每个 timestamp 一个 SAC list
  native/                  native runner 的 geometry/manifest
  bf/                      beamforming 输出矩阵
  log/                     worker 日志
  progress/                进度文件
```

`bf/` 里的结果文件名来自 SAC list 的 timestamp，例如：

```text
2020-01-01-00-00_capon.npy
2020-01-01-00-00_bartlett.npy
```

## 最容易忘的边界

- `example/` 是安装烟测，不是科学示例数据。
- `testing/` 若存在，是作者 ignored 本地沙盒，不随仓库发布。
- 当前一次只跑一个频带；需要多个频带就多次运行。
- 当前主流程目标是 Linux/WSL + NVIDIA CUDA，原生 Windows 暂时退休。
- 真跑大数据前先 `checkmem --safety 0.8`，再用 `--dry-run --serial --limit 1`
  看 native 命令是否合理。
- `--clean-output` 会清理已有 beamforming 输出，只在确认要重跑该 workspace
  时使用。

## Short English Reminder

This file is mainly a personal Chinese checklist. For an English command-line
reminder, run:

```bash
fastbf guide --lang en
```

The intended split is:

- `example/`: synthetic SAC installation smoke test only.
- `testing/`, if present: ignored author-local regression sandbox, not shipped.
- user projects: copy or generate a config with `fastbf init -o my.ini`, then run
  `doctor -> prepare -> checkmem -> dry-run -> run`.
