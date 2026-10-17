"""Download only the files needed by a ConfigWM task from Hugging Face."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_layout import canonical_task_name, manifest_files, selected_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Hugging Face dataset repository ID')
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--env', choices=['pusht', 'tworoom', 'ogbcube'], required=True)
    parser.add_argument('--task', type=canonical_task_name, choices=['configurable_encoder', 'configwm', 'control'], required=True)
    parser.add_argument('--variant', choices=[f'd{i}' for i in range(9)], default='d0')
    parser.add_argument('--list-only', action='store_true', help='Fetch small manifests and print the required file list')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    def fetch_json(name):
        path = hf_hub_download(args.repo, name, repo_type='dataset', local_dir=args.local_dir)
        return json.loads(Path(path).read_text())

    index = fetch_json('dataset.json')
    pending = selected_manifests(index, args.env, args.task, args.variant)
    visited = set(); files = {'dataset.json', 'README.md'}
    while pending:
        name = pending.pop()
        if name in visited: continue
        visited.add(name); files.add(name)
        manifest = fetch_json(name)
        files.update(manifest_files(manifest))
        pending.extend(manifest.get('dependencies', []))
    if args.list_only:
        print('\n'.join(sorted(files)))
        return
    snapshot_download(args.repo, repo_type='dataset', local_dir=args.local_dir,
                      allow_patterns=sorted(files), max_workers=4)
    missing = sorted(name for name in files - {'README.md'}
                     if not (args.local_dir / name).is_file())
    if missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError(
            f'Download incomplete: {len(missing)} required files are missing.\n'
            f'{preview}\nRetry after the repository contains all required files.'
        )
    print(f'Data ready: {args.local_dir.resolve()}')


if __name__ == '__main__': main()
