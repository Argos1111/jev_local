#!/usr/bin/env python3
"""Fetch the pinned official Sarashina checkpoint with local HF authentication; never run its code."""
import argparse
import json
from pathlib import Path

if __package__:
    from .setup_runtime import ROOT, download, model_specs, sha256
else:
    from setup_runtime import ROOT, download, model_specs, sha256

REFERENCE_DIR = ROOT/'.cache/sarashina-official'


def reference_lock():
    return json.loads((ROOT/'native/sarashina-reference.json').read_text())


def verify_file(path, spec):
    if (not path.is_file() or path.stat().st_size != spec['size']
            or sha256(path) != spec['sha256']):
        raise ValueError(f'Missing or changed official reference file: {path}')


def verify_reference(directory, names=None):
    """Verify before executing any local checkpoint/processor Python."""
    spec = reference_lock()
    for name in spec['files'] if names is None else names:
        verify_file(Path(directory)/name, spec['files'][name])
    return spec


def fetch_file(spec, name, output):
    destination = output/name
    if not destination.exists() and not destination.is_symlink():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('Use the isolated .venv-hf environment; see docs/SARASHINA.md') from exc
        # token=True reads the locally configured login. Never log or embed it.
        hf_hub_download(repo_id=spec['repository'], revision=spec['revision'], filename=name,
                        local_dir=output, token=True)
    verify_file(destination, spec['files'][name])
    print(f'Verified: {destination.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Also fetch tokenizer, processor/model Python and license files for reference diagnostics')
    parser.add_argument('--model', choices=('sarashina', 'sarashina-q8'),
                        help='Also fetch the pinned mradermacher LLM GGUF without saving a model/runtime selection')
    parser.add_argument('--output', type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    spec = reference_lock()
    print(spec['notice'], flush=True)
    names = spec['files'] if args.full else spec['conversion_files']
    for name in names:
        fetch_file(spec, name, args.output)
    if args.model:
        runtime = json.loads((ROOT/'scripts/runtime.json').read_text())
        for model in model_specs(runtime, args.model):
            download(model['url'], ROOT/'models'/model['filename'], model['sha256'])
    print('Verified official reference files. No checkpoint code executed; no runtime/model selection changed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Cannot fetch official reference: {exc}')
