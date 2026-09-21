#!/usr/bin/env python3
"""Explicitly fetch the checksum-pinned public reference clone; never run its code."""
import argparse
import json
from pathlib import Path

if __package__:
    from .setup_runtime import ROOT, download, model_specs
else:
    from setup_runtime import ROOT, download, model_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-public-clone', action='store_true',
                        help='Acknowledge unverified equivalence to the gated official checkpoint')
    parser.add_argument('--full', action='store_true', help='Also fetch LLM shards and Python files for reference diagnostics')
    parser.add_argument('--model', choices=('sarashina', 'sarashina-q8'),
                        help='Also fetch the pinned LLM GGUF without saving a model/runtime selection')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/sarashina-reference')
    args = parser.parse_args()
    if not args.accept_public_clone:
        parser.error('Review native/sarashina-reference.json and the model terms, then pass --accept-public-clone')
    spec = json.loads((ROOT/'native/sarashina-reference.json').read_text())
    print(spec['notice'], flush=True)
    names = spec['files'] if args.full else ('config.json', 'model-00001-of-00008.safetensors')
    for name in names:
        file = spec['files'][name]
        download(f'https://huggingface.co/{spec["repository"]}/resolve/{spec["revision"]}/{name}',
                 args.output/name, file['sha256'])
        if (args.output/name).stat().st_size != file['size']:
            raise RuntimeError(f'Unexpected size: {name}')
    if args.model:
        runtime = json.loads((ROOT/'scripts/runtime.json').read_text())
        for model in model_specs(runtime, args.model):
            download(model['url'], ROOT/'models'/model['filename'], model['sha256'])
    print('Verified reference files. No checkpoint code executed; no runtime/model selection changed.')


if __name__ == '__main__':
    main()
