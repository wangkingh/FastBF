# Configs

These INI files are examples. Copy one to a local file before editing paths for
real data:

```text
configs/default.ini
configs/default_20150101.ini
configs/power_deleted_20150101.ini
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
skip_step = NONE      ; 0-based base-window indices to skip, or NONE
taper = tukey         ; tukey, hann, or none
taper_alpha = 0.05    ; used only when taper = tukey

[beamforming]
backend = reference    ; reference or power
method = both          ; reference: capon, bartlett, both; power: raw, deleted, both
mode = grouped         ; stack, grouped, or window
group_windows = 4      ; 4 * 1800 s = one result every 2 hours
```

The `reference` backend writes Capon and channel-equalized Bartlett outputs:

```text
2015-01-01-00-00_g0000_capon.npy
2015-01-01-00-00_g0000_bartlett.npy
2015-01-01-02-00_g0001_capon.npy
2015-01-01-02-00_g0001_bartlett.npy
```

The `power` backend writes power-preserving Bartlett variants:

```text
2015-01-01-00-00_g0000_bartlett_raw.npy
2015-01-01-00-00_g0000_bartlett_channel_deleted.npy
```

The timestamp is the start time of the first base window in that group. The
last group is written even if it contains fewer than `group_windows` windows.
`skip_step` removes 0-based base-window indices before stack/grouped/window
output is built.

The default short Tukey taper (`taper = tukey`, `taper_alpha = 0.05`) keeps the
old rectangular-window behavior close while avoiding hard window edges. Use
`taper = none` only when comparing against older runs that intentionally used
untapered windows.
