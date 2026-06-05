# Configs

These INI files are examples. Copy one to a local file before editing paths for
real data:

```text
configs/default.ini
configs/desktop.ini
configs/server.ini
```

Local configs such as `configs/local_x2.ini` are ignored by git.

FastBF now uses `[time_filter]` plus one or more `[seisarray.*]` sections.
`[output] format` is intentionally absent because the native runner owns the
on-disk result format.
`[bands] bands` is a single target frequency band; run FastBF once per band for
multi-band sweeps.

Beamforming output cadence is controlled by `[beamforming] mode`:

```ini
[window]
win_len = 1800
shift_len = 1800

[beamforming]
method = both          ; capon, bartlett, or both
mode = grouped         ; stack, grouped, or window
group_windows = 4      ; 4 * 1800 s = one result every 2 hours
```

Native grouped output writes one file per output frame and method:

```text
2015-01-01-00-00_g0000_capon.npy
2015-01-01-00-00_g0000_bartlett.npy
2015-01-01-02-00_g0001_capon.npy
2015-01-01-02-00_g0001_bartlett.npy
```

The timestamp is the start time of the first base window in that group. The
last group is written even if it contains fewer than `group_windows` windows.
