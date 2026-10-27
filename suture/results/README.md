# Experiment data

Use the CSVs here as reference inputs; place new run outputs under `runs/` using
`scripts/run_experiment.py`. Figure inputs are mapped in `paper_figs/README.md`.

| Files | Experiment |
|---|---|
| `gonogo_gcp.csv` | 24q soft-edge policy comparison |
| `scale_gcp.csv` | Soft-edge graph-aware scale study |
| `bench10_kv*.csv`, `families*.csv` | KV and cross-family studies |
| `hard_ladder_measured_v2.csv` | Hard-edge measurement and completion outcomes |
| `hard_ladder_measured_v2_paired.csv` | Paired tests across the hard-edge noise grid |
| `hard_ladder_measured_v2_manifest.json` | Package, source and input checksums |
| `a2_*_p24s64.csv` | 72q hardware, 64 shots |
| `a2_*_p24.csv` | 72q hardware, 32-shot sensitivity study |
| `a4_*_b36.csv` | Objective-blind 36q hardware quality |
| `a3_*_hard48.csv`, `a3_*_hard78.csv` | Hard-edge hardware completion |

For the hard-edge ladder, `proper_seen` is true when a circuit-fed run contains
at least one raw feasible measurement. `best_cost` is the minimum objective cost
among those measurements; it is undefined when none is measured. `survived`
records chain completion. `decoder_proper_seen` records whether the accepted
pool ever contains a proper coloring, including repaired states. A0 uses this
last field as its decoder-on-random-input diagnostic.

Paired comparisons use the same twelve seeds. The hard-edge analysis applies
a two-sided paired binomial test and Holm correction over seven noise levels
for each instance/control pair:

```bash
python suture/experiments/analyze_hard_ladder.py \
  --input suture/results/hard_ladder_measured_v2.csv \
  --output runs/hard-ladder-paired.csv
```

Arm names `synd`, `rec_synd`, `synd_c`, and `synd_g` follow the corresponding
runner. SUTURE_C reads constraints only; SUTURE_G also reads graph adjacency.
`purify_restart` and `purify_last` are study-specific extensions of purification.
