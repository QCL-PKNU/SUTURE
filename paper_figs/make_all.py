"""Regenerate current-manuscript figures; optionally stage PDFs into LaTeX."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIGURES = {
    'fig_e1_gonogo': 'fig_e1',
    'fig_e5_encoding': 'fig_e5',
    'fig_e6_generalization': 'fig_e6',
    'fig_e7_device': 'fig_e7',
    'fig_e8_blind_ladder': 'fig_e8',
    'fig_e9_blind_device': 'fig_e9',
    'fig_q3_regime': 'fig_q3',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='store_true', help='Copy generated PDFs into manuscript/figures')
    args = parser.parse_args()
    staging = HERE.parent / 'paper/manuscript/figures'
    if args.stage and not staging.is_dir():
        parser.error(f'--stage needs the manuscript directory {staging}, which is not part of '
                     'this release; omit --stage to leave PDFs in each figure\'s out/ directory')
    subprocess.run([sys.executable, str(HERE / 'prep_all_data.py')], check=True)
    for folder, name in FIGURES.items():
        result = subprocess.run([sys.executable, str(HERE / folder / 'plot.py')],
                                capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        if 'TEXT OVERLAPS' in result.stdout:
            print(f'{name}: text-overlap check failed', file=sys.stderr)
            return 2
        if args.stage:
            shutil.copy2(HERE / folder / 'out' / f'{name}.pdf', staging / f'{name}.pdf')
    subprocess.run([sys.executable, str(HERE / 'table_e1_envelope/make_table.py')], check=True)
    print('All current manuscript figures built.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
