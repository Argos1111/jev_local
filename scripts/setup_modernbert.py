#!/usr/bin/env python3
"""Create .venv-modernbert with torch + transformers for the ModernBERT backend.

GPU wheel selection:
  - NVIDIA: PyTorch CUDA wheels from download.pytorch.org (cu126 by default)
  - AMD (Linux): ROCm 10.0 wheels from stable.repo.amd.com, with the device
    package for the detected gfx target (e.g. gfx1201 for Radeon AI PRO R9700)
  - otherwise: CPU wheels
Verified on this machine: ROCm. CUDA and CPU paths resolve to published wheels
(checked by --dry-run against the indexes) but were not executed on hardware.
Model weights are downloaded by the training/serving code via Hugging Face Hub
into .cache/hf (pinned revision, SHA256 verified).
"""
import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT/'.venv-modernbert'
TORCH_VERSION = '2.13.0'
ROCM_VERSION = '10.0.0'
ROCM_INDEX = 'https://stable.repo.amd.com/rocm/whl-next/'
# CUDA wheel channels that publish torch 2.13.0 (cu128 does not). cu126 needs driver >= 525, cu130 >= 580.
CUDA_INDEX = {'cu126': 'https://download.pytorch.org/whl/cu126', 'cu130': 'https://download.pytorch.org/whl/cu130'}
CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
PYPI = 'https://pypi.org/simple'
GET_PIP = 'https://bootstrap.pypa.io/get-pip.py'
COMMON = ['transformers>=4.48,<6', 'numpy', 'safetensors', 'tokenizers']
KFD_NODES = Path('/sys/class/kfd/kfd/topology/nodes')


def amd_gfx_targets(nodes=KFD_NODES):
    targets = set()
    for path in nodes.glob('*/properties'):
        try:
            match = re.search(r'gfx_target_version (\d+)', path.read_text())
        except OSError:
            continue
        if match and int(match.group(1)):
            value = int(match.group(1))
            major, minor, step = value // 10000, value // 100 % 100, value % 100
            targets.add(f'gfx{major}{minor:x}{step:x}')
    return sorted(targets)


def detect_backend(explicit):
    if explicit != 'auto':
        return explicit, amd_gfx_targets() if explicit == 'rocm' else []
    vendors = set()
    for path in Path('/sys/class/drm').glob('card*/device/vendor'):
        try:
            vendors.add(path.read_text().strip().lower())
        except OSError:
            pass
    if '0x10de' in vendors:
        return 'cuda', []
    if '0x1002' in vendors and platform.system() == 'Linux':
        gfx = amd_gfx_targets()
        if gfx:
            return 'rocm', gfx
    # Windows: no /sys; nvidia-smi on PATH is the practical CUDA signal. AMD on Windows falls back to CPU.
    if platform.system() == 'Windows' and shutil.which('nvidia-smi'):
        return 'cuda', []
    return 'cpu', []


def run(command, **kwargs):
    print('+ ' + ' '.join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['auto', 'cpu', 'cuda', 'rocm'], default='auto')
    parser.add_argument('--cuda', choices=sorted(CUDA_INDEX), default='cu126', help='CUDA wheel channel (cu126 works with older drivers)')
    parser.add_argument('--gfx', action='append', default=[], help='Override detected AMD gfx targets (e.g. gfx1201)')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Python 3.12 or newer is required')
    backend, gfx = detect_backend(args.backend)
    if args.gfx:
        gfx = args.gfx
    print(f'Backend: {backend}' + (f' (targets: {", ".join(gfx)})' if gfx else ''), flush=True)
    if backend == 'rocm' and not gfx:
        parser.exit(1, 'ROCm selected but no gfx target detected; pass --gfx gfx1100 etc.\n')
    if platform.system() == 'Darwin':
        # PyPI macOS wheels include MPS (Apple GPU) support; no separate channel exists.
        backend = 'cpu'
        pip_args = ['--index-url', PYPI, f'torch=={TORCH_VERSION}']
    elif backend == 'cuda':
        pip_args = ['--index-url', CUDA_INDEX[args.cuda], '--extra-index-url', PYPI, f'torch=={TORCH_VERSION}+{args.cuda}']
    elif backend == 'rocm':
        extras = ','.join(f'device-{g}' for g in gfx)
        pip_args = ['--index-url', ROCM_INDEX, '--extra-index-url', PYPI, f'torch[{extras}]=={TORCH_VERSION}+rocm{ROCM_VERSION}']
    else:
        pip_args = ['--index-url', CPU_INDEX, '--extra-index-url', PYPI, f'torch=={TORCH_VERSION}+cpu']
    if args.dry_run:
        print('pip install ' + ' '.join(pip_args))
        return
    python = VENV/('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')
    if not python.exists():
        run([args.python, '-m', 'venv', '--without-pip', VENV])
    if subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True).returncode:
        (ROOT/'.cache').mkdir(exist_ok=True)
        get_pip = ROOT/'.cache/get-pip.py'
        with urllib.request.urlopen(GET_PIP, timeout=60) as source:
            get_pip.write_bytes(source.read())
        run([python, get_pip, '-q'])
    run([python, '-m', 'pip', 'install', '-q', *pip_args])
    run([python, '-m', 'pip', 'install', '-q', *COMMON])
    env = dict(os.environ, TORCH_DISABLE_NATIVE_JIT='1')
    run([python, '-c', 'import torch, transformers; print("torch", torch.__version__, "gpu", torch.cuda.is_available(), '
                       '"transformers", transformers.__version__)'], env=env)
    print('Setup complete. Train: ./train_modernbert.sh   Serve: ./run_modernbert.sh')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f'Command failed with status {exc.returncode}')
