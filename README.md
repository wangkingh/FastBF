# FastBF

[English](README.en.md) | [防忘护栏](docs/USAGE_GUARDRAILS.md) | [配置示例](configs/README.md) | [项目布局](docs/PROJECT_LAYOUT.md) | [更新日志](CHANGELOG.md)

本项目的 v2606 重构、文档整理和发布打包过程使用了 OpenAI Codex / GPT 辅助。

FastBF 是一套面向 SAC 台阵数据的 CUDA 原生 beamforming 流程。Python 层负责
配置解析、SAC inventory、NSL ID、workspace 准备、GPU 显存预估和任务调度；
CUDA/C++ 后端负责 SAC 读取、窗口化、FFT、协方差累计、Capon/Bartlett 扫描
和结果写出。

旧版 GitHub 实现已保留在 `legacy/pre-v2606` 分支和 `legacy-pre-v2606` 标签；
当前 `main` 是 v2606 CUDA-native 重构版。

当前支持目标是 Linux 或 WSL + NVIDIA CUDA。原生 Windows 支持已经暂时退休，
等 Linux/WSL 主流程稳定后再重新加入。

## 获取代码

从 GitHub clone 或解压发布包后，进入项目根目录：

```bash
git clone https://github.com/wangkingh/fastbf.git FastBF
cd FastBF

# 或者解压发布包
tar -xf FastBF*.tar.gz
cd FastBF*
```

项目根目录应能看到 `README.md`、`Makefile`、`caponlib/`、`native/`、
`configs/` 和 `example/`。

## 环境要求

- Python 3.10 或更新版本。
- Python 依赖：`numpy`、`pandas`、`matplotlib`。
- NVIDIA CUDA Toolkit 和可用 GPU。
- GNU Make 以及 C/CUDA 编译工具链。
- Linux 或 WSL。

如果需要创建一个本地 conda 环境：

```bash
conda env create -f environment.yml
conda activate fastbf
```

CUDA 架构默认自动检测。如果自动检测失败，可以手动指定：

```bash
python install_fastbf.py --arch 89
```

## 安装

推荐在项目根目录执行：

```bash
make install
fastbf doctor
```

`make install` 会调用 `install_fastbf.py`，自动检测 CUDA 架构，编译 native
后端、把 `fastbf_native` 部署进 Python 包，并安装可编辑的 `fastbf` 命令。

如果只想编译原生二进制、不安装 Python 包：

```bash
make build
```

原生二进制会写入 `native/fastbf_native` 和 `bin/fastbf_native`，用于 Python
包分发的 staged binary 会写入 `caponlib/bin/<platform>/`。

如果隔一段时间忘了完整流程，可以直接运行：

```bash
fastbf guide
# 或者在源码根目录
make guide
```

更详细的防忘清单见 [docs/USAGE_GUARDRAILS.md](docs/USAGE_GUARDRAILS.md)。

## 两种运行方式

FastBF 支持两种等价的命令风格：

```bash
# 源码树风格：适合还没有安装 fastbf 命令时使用。
python -m caponlib.cli doctor config.ini
python -m caponlib.cli prepare config.ini
python -m caponlib.cli checkmem config.ini
python -m caponlib.cli run config.ini

# 包化 CLI 风格：执行 make install 或 pip install -e . 后可用。
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini
fastbf run config.ini
```

两种方式调用的是同一个 Python 入口。`fastbf ...` 更短；
`python -m caponlib.cli ...` 更适合直接在源码目录中调试。

## 工作流

FastBF 当前围绕轻量 workspace 运行：

```bash
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini
fastbf run config.ini
```

`prepare` 会扫描 SAC 文件、分配 NSL ID、写出每个 timestamp 对应的 SAC list
和台站几何 catalog。`checkmem` 会调用 native 后端的 plan-only 模式，估算每
个 worker 的 CUDA arena、steering 内存和推荐并发数。`run` 会读取准备好的
`saclist/*.tsv`，按 GPU/worker 切分任务并调用 `fastbf_native`。

如果只想查看 native 命令而不真正运行 CUDA：

```bash
fastbf run config.ini --dry-run --serial --limit 1
```

## 快速开始

生成一份 starter 配置：

```bash
fastbf init -o config.ini
```

修改路径、时间、GPU 和输出目录后执行：

```bash
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini --safety 0.8
fastbf run config.ini
```

`--safety 0.8` 表示只使用当前空闲显存的 80% 来推荐 worker 数，给 CUDA
context、handle 和系统波动留下余量。

## 内置示例

仓库包含一个自包含的 synthetic SAC 安装烟测。它的目的不是提供科学示例数据，
而是让新用户在没有任何本地地震数据的情况下，确认 Python CLI、native CUDA
后端、workspace 生成和绘图链路都能工作。

安装完成后，可以按标准 CLI 风格运行：

```bash
cd example
python make_synthetic_sac.py
fastbf doctor config.ini
fastbf prepare config.ini
fastbf checkmem config.ini --safety 0.8
fastbf run config.ini --serial --clean-output
python plot_beamforming.py
```

其中 `make_synthetic_sac.py` 只负责生成一小组可复现的合成 SAC 文件，
`plot_beamforming.py` 只负责把 `fastbf run` 的输出画成图；真正的 FastBF
计算由 `fastbf` CLI 完成：

```text
doctor -> prepare -> checkmem -> run -> plot
```

成功后会得到：

```text
example/workspace/plots/capon_bartlett_db.png
```

这个例子不依赖你的本地 F 盘数据，也不代表真实台阵数据处理效果。它适合安装
后第一时间做 smoke test：能跑通，说明程序入口和依赖基本正常。

## 真实数据和本地夹具

真实数据不随仓库发布。跑自己的数据时，建议从模板生成 ini，然后填写自己的
SAC 根目录、文件名 pattern、station list、时间范围和输出目录：

```bash
fastbf init -o real_data.ini
```

如果你刚安装 FastBF，请优先运行上面的 `example/`。如果你想验证真实数据
结果，请先准备好对应 SAC 数据和 station list，再使用类似下面的命令：

```bash
fastbf doctor real_data.ini
fastbf prepare real_data.ini
fastbf checkmem real_data.ini --safety 0.8
fastbf run real_data.ini --dry-run --serial --limit 1
fastbf run real_data.ini
```

作者机器上如果存在 `testing/`，它只是 ignored 的本地沙盒，用于真实数据回归、
参考结果和性能实验，不随 GitHub 仓库发布。

## 主要配置项

普通用户通常主要修改这些字段：

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

[slowness_grid]
s_range = 12.0
s_res = 0.2

[bands]
bands = 0.1/0.2

[beamforming]
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

配置要点：

- `[seisarray.*]` 可以有多个。FastBF 不使用 FastXC 的 group 逻辑，每个
  array 独立扫描后合并到同一个 workspace。
- `pattern` 描述 `sac_dir` 下 SAC 文件的相对路径，必须从 `{home}` 开始，
  并至少包含 `{station}`、`{component}` 或 `{channel}`，以及日期字段，例如
  `{YYYY}` + `{JJJ}` 或 `{YYYY}` + `{MM}` + `{DD}`。
- 常用通配符：`{*}` 匹配任意字符串，`{?}` 匹配较短的非路径片段；重复出现
  的同名字段必须取相同值。
- `station_list = NONE` 表示不按台站筛选；台站表支持一行一个 station，也支
  持 `lon lat station` 这种测试数据格式。
- `time_list = NONE` 表示使用 `time_start/time_end`；否则 `time_list` 文件
  一行一个 `YYYY-MM-DD HH:MM:SS`。
- `[quality_filter].min_stations` 是 timestamp/list 级别的最小有效台站数。
- `[bands].bands` 当前只支持一个目标频带。多频带扫描建议多次运行 FastBF。
- `[output] format` 不再配置，结果格式由 native runner 控制。

## Beamforming 选项

`[beamforming].method` 控制扫描算法：

- `capon`：只输出 Capon 结果。
- `bartlett`：只输出 Bartlett 结果。
- `both`：共享前面的读取、FFT 和协方差流程，同时输出 Capon 和 Bartlett。

`[beamforming].mode` 控制输出时间粒度：

- `stack`：每个 SAC timestamp/list 输出一个 beamforming 结果。
- `window`：每个基础窗口输出一个结果。
- `grouped`：每 `group_windows` 个基础窗口累计一次协方差并输出一个结果。

例如：

```ini
[window]
win_len = 1800
shift_len = 1800

[beamforming]
method = both
mode = grouped
group_windows = 4
```

表示每个基础窗口为 30 分钟，每 4 个窗口输出一次结果，也就是约每 2 小时输
出一组 Capon/Bartlett。

## Workspace 输出

`fastbf prepare` 和 `fastbf run` 会在 `[output].dir` 下维护轻量 workspace：

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

`nsl_catalog.tsv` 使用 `network + station + location` 分配确定性的 4 位
`nsl_id`。空 SAC location 会归一化为 `00`。timestamp TSV 的列为
`timestamp`、`nsl_id`、`network`、`station`、`location`、`component`、
`sac_path` 和 `array`。

native 当前写出临时 `.npy` 结果；后续计划替换为 FastBF 自定义二进制结构。

## Native 后端

Python 只负责任务准备和进程调度，重计算由 `fastbf_native` 完成。native 命令
的大致形状如下：

```bash
fastbf_native \
  --saclist-list workspace/native/saclists_gpu0_worker00.txt \
  --geometry workspace/native/geometry.tsv \
  -O workspace/bf \
  -B 0.1/0.2 \
  -W 1800/1800 \
  -S 12/0.2 \
  --method capon \
  --mode stack \
  -G 0
```

可选开发参数：

- `--profile`：输出 native 阶段耗时。
- `--freq-chunk-bins N`：覆盖频率 chunk 大小。
- `--steering-cache none|auto|host|device`：实验性 steering cache 模式。
- `--plan-only`：只估算显存，不执行 beamforming。

## 项目目录

```text
caponlib/          Python CLI、配置解析、inventory、checkmem 和调度
native/            CUDA/C++ native backend
example/           自包含 synthetic SAC 安装烟测
configs/           可复用配置示例
docs/              项目布局和开发说明
tools/             绘图和后处理辅助脚本
github_fastbf_zip/ 旧 GitHub 拷贝，仅用于速度和结果对比
bin/               本地构建出的 native 二进制
```

FastBF 的 inventory/prepare 部分借鉴 FastXC 的轻量外骨架，但没有引入 FastXC
复杂的中间结构，例如 `path_plan/`、`spack_by_timestamp/`、`xcache/`、
`sourcepack/` 或 `stack/`。

当前主流程已经移除旧 Python `SeisHandler`、`sac_refactor.py`、
`cfk`/`ias_capon` 执行路径。旧实现只保留在 `github_fastbf_zip/` 中用于对比。
