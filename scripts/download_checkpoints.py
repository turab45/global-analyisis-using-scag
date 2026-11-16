"""Download release checkpoints into the layout expected by CINEMA_RESULTS_ROOT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('CINEMA_RESULTS_ROOT', './cinema_results')))
    parser.add_argument('--dataset', choices=['acdc', 'mnms2'])
    parser.add_argument('--model')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--list', action='store_true', help='List selected files without downloading.')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'checkpoints/manifest.json').read_text())
    selected = [r for r in manifest['checkpoints'] if all(value is None or r[key] == value for key, value in [('dataset', args.dataset), ('model', args.model), ('seed', args.seed)])]
    if not selected:
        parser.error('No checkpoints match the requested filters.')
    root = args.root.expanduser().resolve()
    for record in selected:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or Path(record['asset']).name != record['asset']:
            raise ValueError('Unsafe manifest path.')
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError('Checkpoint path escapes the requested root.')
        if args.list:
            print(f"{record['bytes'] / 1024**2:.1f} MiB  {destination}")
            continue
        if destination.exists():
            if sha256(destination) == record['sha256']:
                print('Verified, skipping:', destination)
                continue
            raise FileExistsError(f'Refusing to overwrite a different checkpoint: {destination}')
        if shutil.which('gh') is None:
            raise RuntimeError('Install GitHub CLI and run gh auth login before downloading.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=destination.parent) as temporary:
            subprocess.run(['gh', 'release', 'download', manifest['release'], '--repo', manifest['repository'], '--pattern', record['asset'], '--dir', temporary], check=True)
            downloaded = Path(temporary) / record['asset']
            if downloaded.stat().st_size != record['bytes'] or sha256(downloaded) != record['sha256']:
                raise ValueError(f'Checkpoint checksum/size mismatch: {record["asset"]}')
            # Hard-linking publishes the verified file atomically without overwriting an existing path.
            os.link(downloaded, destination)
        print('Installed:', destination)


if __name__ == '__main__':
    main()
