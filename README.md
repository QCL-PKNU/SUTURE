# SUTURE: Syndrome-Guided Repair for Segmented Feasibility-Preserving VQAs on Noisy Hardware

Repair infeasible measurement bitstrings before the next segment of a quantum
optimization workload. SUTURE uses existing integer constraints `C @ x = b`
to identify suspect bits, search for feasible corrections, and prepare the next
initialization register.

See [deployment instructions](suture/README.md) for input shapes, counters,
resampling, and choosing a policy using workload's measurements.

## Reproduce the experiments

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r suture/requirements-lock.txt
python -m pytest suture/tests -q
python scripts/run_experiment.py --output runs/scale-smoke \
  remote/remote_scale_campaign.py --smoke --workers 1
python scripts/run_experiment.py --output runs/hard-smoke \
  run_hard_ladder.py --smoke --workers 1
python paper_figs/make_all.py
```

These commands use local simulation and supplied data; they submit no QPU jobs.
[Experiment instructions](suture/README.md#experiments) describe full runs
and hardware execution. [Figure inputs](paper_figs/README.md) identify the data
used by each manuscript plot. `SUTURE_C` denotes constraint-only repair;
`SUTURE_G` adds graph-aware selection. Experiment code retains `SYND` arm names.


## License

New SUTURE source is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Vendored Rasengan retains its [MIT license](rasengan/LICENSE).
