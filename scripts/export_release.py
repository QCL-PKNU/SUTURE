"""Export the current non-ignored source snapshot without Git history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'SHA256SUMS.json'   # regenerated below; never carried over from the tree


def git(*arguments):
    """Run a git command in ROOT, failing with an actionable message."""
    try:
        return subprocess.run(['git', '-C', str(ROOT), *arguments],
                              capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit('export_release.py needs git on PATH to enumerate non-ignored files.')
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f'git {" ".join(arguments)} failed in {ROOT}: '
            f'{error.stderr.decode().strip() or "no detail"}\n'
            'This script exports from the SUTURE git working tree. An extracted source '
            f'snapshot is not a repository and already ships its {MANIFEST}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/suture-source.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a new archive name')
    raw = git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
    names = sorted({x.decode() for x in raw.split(b'\0') if x})
    tracked_ignored = git('ls-files', '-ci', '--exclude-standard', '-z')
    excluded = {x.decode() for x in tracked_ignored.split(b'\0') if x}
    manifest = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = ROOT / name
            if name in excluded or path.resolve() == output or not path.is_file():
                continue
            if name == MANIFEST:
                continue      # a stale copy would duplicate and contradict the new manifest
            if path.is_symlink():
                raise ValueError(f'Release source must not contain symlinks: {name}')
            data = path.read_bytes()
            archive.writestr('suture/' + name, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        archive.writestr('suture/' + MANIFEST, json.dumps(manifest, indent=2)+'\n')
    print(f'{output}: {len(manifest)} files, SHA-256 manifest included; no Git history')


if __name__ == '__main__':
    main()
