# Manuscript figures

```bash
python paper_figs/make_all.py
```

This refreshes the input tables, generates all current figures, checks plot and
table processes, and leaves the PDFs in each figure's `out/` directory. The
author-only `--stage` flag additionally copies the PDFs into
`paper/manuscript/figures`, which is a LaTeX working directory that is not part
of this release. Pipeline/circuit illustrations are supplied directly in the
manuscript figure directory.

| Output | Input |
|---|---|
| E1 | `gonogo_gcp.csv`; 24q, 64 shots |
| E5 | `encoding_flips.csv`, `dw_chain.csv` |
| E6 | `paper_data/{profiler_prediction,cliff_positions,window_discordance}.csv` |
| E7 | `a1_rows.csv`; `a2_chain/final` at 36q, `_p16` at 48q, `_p24s64` at 72q; `latency_breakdown.csv` |
| E8 | `hard_ladder_measured_v2.csv` |
| E9 | `a4_chain/final_b36.csv`; `a3_final_hard48/hard78.csv` |
| Q3 | `scale_gcp.csv`, `bench10_kv_paired_all.csv`, `paper_data/lownoise_paired.csv` |

E8 displays measured success for circuit-fed arms and labels A0's decoder-only
success separately. E6 and Q3 use curated statistical tables derived by
`suture/experiments/curate_paper_data.py`. See `suture/results/README.md`
for endpoint definitions and the paired-analysis command.

Generated PNGs and plot output folders are excluded from Git. The LaTeX source
is `paper/manuscript/main_update.tex`.
