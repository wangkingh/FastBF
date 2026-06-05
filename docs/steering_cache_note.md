# Steering Cache Experiment Note

FastBF's steering vector is mathematically independent of the daily waveform
data.  It depends only on station geometry, frequency bins, the slowness grid,
`dt`, and `nsamp`.  Therefore, a natural optimization idea is to precompute the
full steering table once and reuse it for all daily beamforming jobs.

For the X2 2015 test case, this full table is large:

```text
721 frequency bins * 121 * 121 slowness points * 208 receivers * complex float
~= 16.36 GiB
```

We implemented and tested a global device-side steering cache.  The result was
numerically identical to the on-the-fly chunk strategy, but it did not improve
runtime:

```text
0.1-0.5 Hz, 10 synthetic days

On-the-fly chunk steering: 45.38 s
Global device steering:   46.11 s
```

The reason is the classical HPC tradeoff between recomputation and memory
traffic.  Steering generation is an `O(Nr)` operation per frequency and slowness
point, while Capon beamforming evaluates `a^H R^-1 a`, which is effectively
`O(Nr^2)`.  For `Nr=208`, the pairwise beamforming accumulation dominates the
cost.  Caching steering removes the relatively small `sincos` generation step,
but it introduces a huge persistent table that must still be read by the FK
kernel.

Thus the current chunk-local strategy is preferable: generate steering for the
active frequency chunk, consume it immediately, and let the workspace be reused.
This keeps global memory pressure low and preserves locality.  The experiment is
a useful reminder that in GPU/HPC codes, precomputing intermediate quantities is
not automatically faster; the deciding factor is arithmetic intensity and memory
bandwidth.  Cheap intermediates are often better recomputed near their point of
use, while only expensive and heavily reused intermediates should be cached.

A lighter cache is still useful.  FastBF now precomputes the geometry delay
table

```text
tau(receiver, slowness) = rx * sx + ry * sy
```

for the master receiver list once at native start-up.  This table is small:

```text
208 receivers * 121 * 121 slowness points * float
~= 11.62 MiB
```

Each daily SAC list then maps its `nsl_id` rows into the master receiver table
and gathers only the active receiver rows into the per-list workspace.  This is
different from full steering caching because it does not include the frequency
dimension and does not materialize complex exponentials.  It also keeps the
production path robust when a daily list has missing stations: a day may use any
subset of the master geometry, while a new unknown station correctly requires a
new prepared geometry table.
