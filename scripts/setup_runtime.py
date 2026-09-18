#!/usr/bin/env python3
"""Install the pinned GPU/CPU runtime and model without Python dependencies."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) == digest:
            print(f'Verified: {destination.name}', flush=True)
            return
        raise RuntimeError(f'Checksum mismatch: {destination}. Move it aside and retry.')
    partial = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading: {destination.name}', flush=True)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'jev-local-setup'})
        with urllib.request.urlopen(request, timeout=60) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        if sha256(partial) != digest:
            raise RuntimeError(f'Checksum mismatch for download: {destination.name}')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


class RuntimeUnavailable(RuntimeError):
    """An installed build cannot run or cannot see its requested GPU backend."""


def gpu_vendors():
    vendors = set()
    for path in Path('/sys/class/drm').glob('card*/device/vendor'):
        try:
            vendors.add(path.read_text().strip().lower())
        except OSError:
            pass
    # Also works on WSL, where DRM vendor information may be absent.
    command = shutil.which('nvidia-smi')
    if not command and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        command = '/usr/lib/wsl/lib/nvidia-smi'
    if command:
        try:
            result = subprocess.run([command, '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                vendors.add('0x10de')
        except (OSError, subprocess.TimeoutExpired):
            pass
    return vendors


def candidates(system, machine, backend, vendors):
    system = system.lower()
    machine = {'AMD64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine) if system == 'linux' else machine
    prefix = f'{system}-{machine}'
    supported = {
        'linux-x86_64': ['cuda', 'rocm', 'cpu'],
        'linux-aarch64': ['cpu'],
        'darwin-arm64': ['metal', 'cpu'],
        'darwin-x86_64': ['cpu'],
    }
    if prefix not in supported:
        raise RuntimeError(f'Unsupported platform: {prefix}. Use WSL2 on Windows; see docs/SETUP.md.')
    if backend != 'auto':
        if backend not in supported[prefix]:
            raise RuntimeError(f'{backend} has no pinned build for {prefix}')
        return [f'{prefix}-{backend}']
    choices = []
    if prefix == 'linux-x86_64':
        if '0x10de' in vendors:
            choices.append('cuda')
        if '0x1002' in vendors:
            choices.append('rocm')
    if prefix == 'darwin-arm64':
        choices.append('metal')
    return [f'{prefix}-{choice}' for choice in [*choices, 'cpu']]


def runtime_env(directory):
    env = os.environ.copy()
    if platform.system() == 'Linux':
        paths = sorted({str(p.parent.resolve()) for p in directory.rglob('*.so*')})
        if paths:
            env['LD_LIBRARY_PATH'] = os.pathsep.join(paths + ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    return env


def install_runtime(spec, variant):
    destination = ROOT/'.cache/runtime'/spec['version']/variant
    if not destination.exists():
        archives = []
        for asset in spec['variants'][variant]['assets']:
            archive = ROOT/'.cache/downloads'/asset['url'].rsplit('/', 1)[1]
            download(asset['url'], archive, asset['sha256'])
            archives.append(archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            unpacked = Path(temporary)/'bundle'
            unpacked.mkdir()
            for archive in archives:
                with tarfile.open(archive) as bundle:
                    bundle.extractall(unpacked, filter='data')
            if len(list(unpacked.rglob('llama-server'))) != 1:
                raise RuntimeError('Expected one llama-server in the runtime archive')
            unpacked.rename(destination)
    servers = list(destination.rglob('llama-server'))
    if len(servers) != 1:
        raise RuntimeError(f'Incomplete runtime: {destination}. Move it aside and retry.')
    return servers[0], destination


def validate_runtime(server, directory, backend):
    flag = '--version' if backend == 'cpu' else '--list-devices'
    try:
        result = subprocess.run([str(server), flag], env=runtime_env(directory),
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeUnavailable(str(exc)) from exc
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeUnavailable(f'{server.name} exited {result.returncode}: {output[-1500:]}')
    patterns = {'cuda': r'^\s*CUDA\d+\s*:', 'rocm': r'^\s*(?:ROCm|HIP)\d+\s*:',
                'metal': r'^\s*(?:MTL|Metal)\d+\s*:'}
    if backend != 'cpu' and not re.search(patterns[backend], output, re.MULTILINE):
        raise RuntimeUnavailable(f'No usable {backend} device reported: {output[-1500:]}')
    print(output.strip(), flush=True)


def select_runtime(spec, options, automatic):
    for variant in options:
        backend = variant.rsplit('-', 1)[1]
        print(f'Trying runtime: {variant}', flush=True)
        server, directory = install_runtime(spec, variant)
        try:
            validate_runtime(server, directory, backend)
        except RuntimeUnavailable as exc:
            if not automatic or backend == 'cpu':
                raise
            print(f'{backend} unavailable: {exc}\nTrying the next detected backend (CPU is the final fallback).', flush=True)
            continue
        return {'variant': variant, 'server': str(server.relative_to(ROOT)),
                'directory': str(directory.relative_to(ROOT)), 'gpu_layers': '0' if backend == 'cpu' else 'all'}
    raise RuntimeError('No usable runtime found')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-only', action='store_true', help='Only download the model')
    parser.add_argument('--backend', choices=['auto', 'cpu', 'cuda', 'rocm', 'metal'], default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Show detected runtime candidates without downloading')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Python 3.12 or newer is required')
    lock = json.loads((ROOT/'scripts/runtime.json').read_text())
    options = [] if args.model_only else candidates(platform.system(), platform.machine(), args.backend, gpu_vendors())
    print('Runtime candidates: ' + (', '.join(options) or 'model only'), flush=True)
    if args.dry_run:
        return
    selection = select_runtime(lock['llama'], options, args.backend == 'auto') if options else None
    spec = lock['model']
    download(spec['url'], ROOT/'models'/spec['filename'], spec['sha256'])
    if selection:
        path = ROOT/'.cache/runtime/selected.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(selection, indent=2) + '\n')
        temporary.replace(path)
        print(f'Selected: {selection["variant"]} (GPU_LAYERS={selection["gpu_layers"]})', flush=True)
    print('Setup complete. Run ./run.sh, then python3 systemone_client.py in another terminal.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        sys.exit(f'Setup failed: {exc}\nSee docs/SETUP.md for prerequisites and manual setup.')
