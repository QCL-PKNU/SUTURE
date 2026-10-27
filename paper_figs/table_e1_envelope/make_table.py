"""Generate the supplemental envelope-observation table from its actual schema."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

def escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

env = pd.read_csv(HERE / 'data/t1_envelope.csv')
required = {'item', 'value', 'detail'}
if not required <= set(env):
    raise ValueError(f'Expected columns {required}, found {set(env)}')
rows = [f"{escape(r['item'])} & {r['value']:g} & {escape(r['detail'])} " + r'\\' for _, r in env.iterrows()]
body = '\n'.join([r'\begin{tabular}{lcl}', r'\toprule',
                  r'observation & measured value & detail \\', r'\midrule',
                  *rows, r'\bottomrule', r'\end{tabular}', ''])
(HERE / 'out').mkdir(exist_ok=True)
(HERE / 'out/table_e1.tex').write_text(body)
print('Supplemental envelope table generated.')
