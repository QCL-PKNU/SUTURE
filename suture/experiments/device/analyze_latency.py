"""analyze_latency — Rasengan-Fig-12-style runtime breakdown from the
campaign's latency logs (results/a1_latency.jsonl, results/a2_latency.jsonl).

Per-arm stacked components, per round and totalled:
  quantum:   qpu_usage_s (apportioned per circuit within each shared job:
             purify and synd circuits ride the same job, split by circuit
             count — stated in the figure caption)
  queue:     queue_s (created->running; shared, apportioned identically)
  transpile: transpile_s (client-side, shared, apportioned)
  decode:    decode_s_purify / decode_s_synd (arm-specific: purify filter
             vs SYND syndrome repair)
  register:  register_update_s (shared, apportioned)
Output: results/latency_breakdown.csv + printed table; the manuscript
figure is rendered from that CSV with apply_figure_style() separately.
"""
import json
import sys

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    L = load_jsonl("results/a2_latency.jsonl")
    L = L[L.stage == "A2"].copy()
    # per-arm apportionment: each round's job carries n_circuits circuits,
    # split evenly between the two arms (equal live-seed counts until one
    # arm starts dying; then apportion by actual per-arm circuit counts if
    # recorded — v1: even split, flagged in caption).
    out = []
    for arm in ("purify", "synd"):
        share = 0.5
        row = dict(arm=arm,
                   quantum_s=(L.qpu_usage_s.fillna(0) * share).sum(),
                   queue_s=(L.queue_s.fillna(0) * share).sum(),
                   transpile_s=(L.transpile_s.fillna(0) * share).sum(),
                   decode_s=L[f"decode_s_{arm}"].fillna(0).sum(),
                   register_s=(L.register_update_s.fillna(0) * share).sum(),
                   build_s=(L.build_s.fillna(0) * share).sum(),
                   rounds=len(L))
        row["total_s"] = sum(v for k, v in row.items() if k.endswith("_s"))
        out.append(row)
    T = pd.DataFrame(out)
    T.to_csv("results/latency_breakdown.csv", index=False)
    print(T.round(2).to_string(index=False))
    print("\nper-round medians:")
    med = L[["qpu_usage_s", "queue_s", "transpile_s", "decode_s_purify",
             "decode_s_synd", "register_update_s", "round_wall_s"]].median()
    print(med.round(3).to_string())


if __name__ == "__main__":
    main()
