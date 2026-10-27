"""Paired success tests for hard-edge ladder outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, (len(values)-rank)*values[i])
        result[i] = min(maximum, 1.)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('hard_ladder.csv'))
    parser.add_argument('--output', type=Path, default=Path('hard_ladder_paired.csv'))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    schema = set(data.endpoint_schema) if 'endpoint_schema' in data else {'hybrid-v1'}
    if schema not in ({'measured-v2'}, {'hybrid-v1'}):
        parser.error('requires one consistent endpoint schema')
    rows = []
    for (instance, n, lam), group in data.groupby(['instance', 'n', 'lam']):
        pivot = group.pivot(index='seed', columns='arm', values='proper_seen')
        for control in ['purify', 'purify_restart', 'purify_last', 'nearest']:
            pair = pivot[['synd', control]].dropna().astype(bool)
            if len(pair) != 12:
                raise ValueError(f'{instance}/{lam}/{control}: expected 12 paired seeds')
            wins = int((pair.synd & ~pair[control]).sum())
            losses = int((~pair.synd & pair[control]).sum())
            p = binomtest(wins, wins+losses).pvalue if wins+losses else 1.
            rows.append(dict(instance=instance,n=n,lam=lam,control=control,
                             endpoint_schema=next(iter(schema)),pairs=len(pair),wins=wins,losses=losses,ties=len(pair)-wins-losses,p=p))
    out = pd.DataFrame(rows)
    out['p_holm_grid'] = np.nan
    for _, group in out.groupby(['instance','control']):
        out.loc[group.index, 'p_holm_grid'] = holm(group.p.to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out[(out.n>=78)&(out.p_holm_grid<.05)].to_string(index=False))
    print(f'Wrote {len(out)} paired comparisons to {args.output}')


if __name__ == '__main__':
    main()
