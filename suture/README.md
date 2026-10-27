# Using SUTURE

## Deploy at a measurement boundary

Use the NumPy-only API from the repository root:

```python
from suture import repair_boundary

result = repair_boundary(measured_bits, C, b, rng=rng, shots=64)
if result.can_continue:
    initialize_next_segment(result.next_register)
else:
    stop_or_use_application_fallback()
```

`initialize_next_segment` and `stop_or_use_application_fallback` are your
application's operations. A complete runnable example is available through
`python -m examples.boundary_repair`.

| Argument | Meaning |
|---|---|
| `measured_bits` | Integer/bool array `(S, n)` of measured zero/one bitstrings |
| `C` | Dense integer matrix `(m, n)` describing equality constraints |
| `b` | Integer vector `(m,)` of required constraint values |
| `rng` | `numpy.random.Generator` for repair ties and resampling |
| `shots` | Number of rows for the next register; defaults to `S` |
| `max_flips` | Maximum repair weight; default `6` |

For inequalities, include your encoding's binary slack variables in both the
matrix and measured bitstrings. Use small integer coefficients; computation uses
signed 64-bit arithmetic. Create and reuse a seeded generator for repeatable
execution, for example `rng = numpy.random.default_rng(7)`.

The function checks the constraints, retains feasible measurements, and searches
suspect-bit flips in increasing Hamming weight. It samples uniformly among
minimum-weight repairs. The search also enforces a 200,000-combination cap per
weight level. A rejected shot contributes no initialization state.

| Result field | Meaning |
|---|---|
| `accepted` | Feasible and repaired occurrences before resampling |
| `next_register` | Feasible initialization rows, or `None` when the pool is empty |
| `can_continue` | Whether a next register is available |
| `n_feasible_measured` | Input shots already satisfying all constraints |
| `n_repaired` | Input shots successfully repaired |
| `n_rejected` | Input shots discarded after the bounded search |
| `n_measured` | Total input shot count |

Inputs are not modified. Resampling weights each accepted occurrence equally,
so repeated measured bitstrings retain their multiplicity. SUTURE does not
choose a restart seed or previous register when the pool is empty; the caller
controls that policy.

## Choose a deployment policy

1. Run purification and repair on the same workload, shot budget, and device
   settings. Keep the circuit and optimization parameters fixed for comparison.
2. Record chain completion, feasible measured fraction, repaired/rejected counts,
   decoder time, and your application's solution-quality metric.
3. Choose repair when it meets the application's quality and latency requirements
   or sustains execution that purification cannot complete. Configure what to do
   if neither policy provides a next register.

Measure solution quality on raw outputs before repair when evaluating measured
success. Quality of repaired states is a separate hybrid-system metric. A live
chain provides further sampling opportunities; whether those samples improve
solution quality is workload-dependent.

The deployment API implements generic objective-blind SUTURE_C. The experiment
runners also contain block-specialized hard-edge SUTURE_C and graph-aware
SUTURE_G. Graph-aware selection uses adjacency and should be evaluated as a
separate policy when edge conflicts are the objective.

## Experiments

Install `requirements-lock.txt` in a Python 3.12 virtual environment for the
complete experiment suite. This includes dependencies imported by the vendored
Rasengan code. The NumPy-only boundary API requires none of those quantum or
classical solver packages. Upstream model-solving commands may require a suitable
Gurobi license.

Run these commands from the repository root:

```bash
python -m pytest suture/tests -q
python scripts/run_experiment.py --output runs/scale \
  remote/remote_scale_campaign.py --workers 8
python scripts/run_experiment.py --output runs/hard-ladder \
  run_hard_ladder.py --workers 8
python scripts/run_experiment.py --output runs/kv remote/remote_bench10.py
python scripts/run_experiment.py --output runs/families remote/remote_bench10_eval.py
python scripts/run_experiment.py --output runs/gonogo run_gonogo.py
```

Add `--smoke --workers 1` to the scale or hard-edge runner for a two-segment,
one-cell check. Each output directory must be new. The launcher sets
`PYTHONHASHSEED=0`, pins BLAS threads, links supplied bundles and records source,
package and input checksums. Output files are written inside the run directory.
The scale and hard-edge runners write their final CSV after the task pool
finishes. Full KV and cross-family campaigns can take hours to days.

Purification is the boundary policy inherited from Rasengan. `purify_restart`
and `purify_last` are comparison policies introduced in this study: they supply
the original seed or previous register when purification has no survivors.

### Hardware

An Aer-only check for the objective-blind quality path:

```bash
python scripts/run_experiment.py --output runs/device-dry \
  device/run_a4.py --dry --lam 0 --seeds 1 --rounds 2 --shots 64
```

| Experiment | Runner | Bundle and instance |
|---|---|---|
| Soft-edge 36/48q | `device/run_a2.py` | `bench10_bundle.pkl`; `gcp_p12_s0` / `gcp_p16_s0` |
| Soft-edge 72q | `device/run_a2.py` | `scale_bundle.pkl`; `gcp_p24_s0`; pass `--shots 64` |
| Objective-blind quality, 36q | `device/run_a4.py` | `bench10_bundle.pkl`; `gcp_p12_s0` |
| Hard-edge liveness, 48q | `device/run_a3.py` | `hard_device_bundle.pkl`; `hardD_v7` |
| Hard-edge liveness, 78q | `device/run_a3.py` | `hard_ladder_bundle.pkl`; `hardL_v12` |

Pass `--bundle`, `--instance`, `--backend`, `--shots`, and the desired
`--rounds`/`--seeds` through the launcher. Real-device runs use shell variables
`IBM_QUANTUM_TOKEN` and `CRN_IBMQ`. Configure `--cap` and `--used-baseline`
for your account's QPU allowance. Removing `--dry` submits real jobs.

## Data and figures

`results/README.md` defines the result fields and campaign files.
`python paper_figs/make_all.py` regenerates figures from supplied data.
Use the included pickle/dill bundles as trusted inputs.

## License

New SUTURE source is Apache-2.0. The vendored Rasengan source retains MIT.
