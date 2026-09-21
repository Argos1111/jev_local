#!/usr/bin/env python3
"""Prepare the verified F16 Sarashina projector bundle locally; never upload or read tokens.

Only the documented, checksum-pinned projector is packaged. No Hugging Face,
PyTorch, or GPU dependency is needed. Existing output directories are not replaced.
"""
import argparse
import json
from pathlib import Path
import shutil
from string import Template

if __package__:
    from .setup_runtime import ROOT, sha256
else:
    from setup_runtime import ROOT, sha256

PROJECTOR_NAME = 'sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf'
PROJECTOR_SHA256 = '7c170758eabf18eaf60d6c584d27b9955c3fdd595975d21fc4e7b9c93ca1b321'
PROJECTOR_BYTES = 892579936
LICENSE_HASHES = {
    'LICENSE.sarashina2.2-vision-3b': 'c2142adf07ff4607749b33314f969ffca01ce0bd629eee341323bdf5ca825b3f',
    'LICENSE.siglip': '43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1',
}


def prepare(mmproj, output):
    mmproj, output = Path(mmproj), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}; inspect it or choose another --output')
    if mmproj.stat().st_size != PROJECTOR_BYTES or sha256(mmproj) != PROJECTOR_SHA256:
        raise ValueError('Expected the documented F16 Sarashina projector; size/SHA256 does not match')
    source = json.loads((ROOT/'native/sarashina-reference.json').read_text())
    manifest = json.loads(mmproj.with_suffix(mmproj.suffix + '.json').read_text())
    expected = {
        'source_repository': source['repository'],
        'source_revision': source['revision'],
        'checkpoint_file': source['checkpoint_file'],
        'config_sha256': source['files']['config.json']['sha256'],
        'checkpoint_sha256': source['files'][source['checkpoint_file']]['sha256'],
        'preprocessor_sha256': source['files']['preprocessor_config.json']['sha256'],
        'preprocessing': 'sarashina_official_v1',
        'output_sha256': PROJECTOR_SHA256,
        'tensors': 442,
        'projector_type': 'sarashina2vl',
        'outtype': 'f16',
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Projector/conversion manifest mismatch')
    # Publish only known fields: custom local paths or credentials must not leak.
    small_files = {
        PROJECTOR_NAME + '.json': (json.dumps(expected, indent=2) + '\n').encode(),
        'sarashina-reference.json': (ROOT/'native/sarashina-reference.json').read_bytes(),
    }
    for name, digest in LICENSE_HASHES.items():
        path = ROOT/'licenses'/name
        if sha256(path) != digest:
            raise ValueError(f'License differs from the reviewed upstream text: {name}')
        small_files[name] = path.read_bytes()
    substitutions = {'filename': PROJECTOR_NAME, 'size': f'{PROJECTOR_BYTES:,}',
                     'sha256': PROJECTOR_SHA256, 'source_repo': source['repository'],
                     'source_revision': source['revision']}
    for name in ('README.md', 'NOTICE.md'):
        template = (ROOT/'native/sarashina-mmproj'/f'{name}.in').read_text()
        small_files[name] = Template(template).substitute(substitutions).encode()

    # Create exclusively after all inputs have been checked. A failed copy may
    # leave an incomplete directory; SHA256SUMS is written only on success.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(mmproj, output/PROJECTOR_NAME)
    if sha256(output/PROJECTOR_NAME) != PROJECTOR_SHA256:
        raise ValueError('Projector changed while copying; do not upload this incomplete directory')
    for name, data in small_files.items():
        with (output/name).open('xb') as stream:
            stream.write(data)
    hashes = {PROJECTOR_NAME: PROJECTOR_SHA256}
    hashes.update({name: sha256(output/name) for name in small_files})
    with (output/'SHA256SUMS').open('x') as stream:
        stream.write(''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())))
    return sorted([*hashes, 'SHA256SUMS'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmproj', type=Path, default=ROOT/'models'/PROJECTOR_NAME)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/hf-sarashina-mmproj-official')
    args = parser.parse_args()
    files = prepare(args.mmproj, args.output)
    print(f'Prepared locally: {args.output.resolve()}')
    for name in files:
        print(f'  {name}')
    print('No upload performed. Review README.md, NOTICE.md and SHA256SUMS before using hf upload.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f'Cannot prepare projector upload: {exc}')
