"""Run an artifact experiment in an isolated output directory with pinned hashing."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'suture'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for results, state, and provenance; never archived results')
    parser.add_argument('experiment', help='Path relative to suture/experiments')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = (ARTIFACT / 'experiments' / args.experiment).resolve()
    if not script.is_relative_to(ARTIFACT / 'experiments') or not script.is_file():
        parser.error('experiment must be an existing script under suture/experiments')
    output = args.output.resolve()
    if output.is_relative_to(ARTIFACT / 'results') or output == ROOT:
        parser.error('choose a separate run directory')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('results', 'device'):
        (output / name).mkdir()
    (output / 'packs').symlink_to(ARTIFACT / 'packs', target_is_directory=True)
    bundles = {}
    for source in sorted((ARTIFACT / 'packs').glob('*')):
        if source.is_file():
            (output / source.name).symlink_to(source)
            bundles[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    command = [sys.executable, str(script), *args.arguments]
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    manifest = dict(command=[args.experiment, *args.arguments],
                    python=sys.version, started=datetime.now(timezone.utc).isoformat(),
                    source_revision=revision.stdout.strip() or None,
                    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ('suture/lib', 'suture/experiments', 'rasengan/rasengan')
                                   for p in sorted((ROOT / folder).rglob('*.py'))},
                    environment={k:env[k] for k in ('PYTHONHASHSEED', 'OMP_NUM_THREADS',
                                                   'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                    bundle_sha256=bundles,
                    endpoint_schema='hybrid-v1' if script.name == 'run_hard_ladder.py' else 'campaign-specific')
    path = output / 'run_manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Running in {output}', flush=True)
    result = subprocess.run(command, cwd=output, env=env)
    manifest.update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
