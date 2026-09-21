#!/usr/bin/env python3
"""Launch the isolated Sarashina experiment without changing normal saved selections."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__:
    from .build_sarashina import BACKENDS, BASE, DEFAULT_DEVICES, GPU_PLUGINS, RUNTIME_VARS
    from .setup_runtime import ROOT, model_specs, sha256
else:
    from build_sarashina import BACKENDS, BASE, DEFAULT_DEVICES, GPU_PLUGINS, RUNTIME_VARS
    from setup_runtime import ROOT, model_specs, sha256


def launch_environment(backend, profile, mmproj, text_only, flash_attn, cache_mib, device=None):
    build = BASE/f'build-{backend}'
    manifest_path = build/'jev-build.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Run python3 scripts/build_sarashina.py --backend {backend} first')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('backend') != backend:
        raise RuntimeError('Build manifest/backend mismatch')
    required = {'llama-server', 'libmtmd.so', 'libllama.so', 'libllama-server-impl.so',
                'libggml.so', 'libggml-base.so', 'libggml-cpu.so'}
    if backend in GPU_PLUGINS:
        required.add(GPU_PLUGINS[backend])
    if not required <= manifest.get('binaries', {}).keys():
        raise RuntimeError('Incomplete build manifest; rebuild the experiment')
    for name, digest in manifest['binaries'].items():
        if Path(name).name != name or not (build/'bin'/name).is_file() or sha256(build/'bin'/name) != digest:
            raise RuntimeError(f'Changed or missing build artifact: {name}; rebuild the experiment')
    fa_tests = manifest.get('fa160_tests') or {}
    if backend == 'stock-rocm' and flash_attn == 'on':
        print('WARNING: Stock ROCm has no GPU FA160 kernel; forced FA can fall back to CPU and slow down.', file=sys.stderr)
    if not 0 <= cache_mib <= 4096:
        raise ValueError('Image cache size must be in 0..4096 MiB')
    env = dict(os.environ)
    for key in RUNTIME_VARS:
        if not env.get(key) and manifest.get('runtime_env', {}).get(key):
            env[key] = manifest['runtime_env'][key]
    model = ROOT/'models'/model_specs(json.loads((ROOT/'scripts/runtime.json').read_text()), profile)[0]['filename']
    if env.get('LFM_MODEL') and Path(env['LFM_MODEL']).resolve() != model.resolve():
        raise RuntimeError('Unset LFM_MODEL before using this profile-specific launcher')
    if not text_only:
        record = mmproj.with_suffix(mmproj.suffix + '.json')
        if not mmproj.is_file() or not record.is_file():
            raise RuntimeError(f'Convert the projector first: {mmproj}; see docs/SARASHINA.md')
        info = json.loads(record.read_text())
        if info.get('projector_type') != 'sarashina2vl' or info.get('output_sha256') != sha256(mmproj):
            raise RuntimeError('Projector/manifest mismatch; refusing to use a mismatched artifact')
    env.update(LLAMA_SERVER=str(build/'bin/llama-server'), LFM_MODEL=str(model),
               LFM_MMPROJ='' if text_only else str(mmproj.resolve()),
               GPU_LAYERS='0' if backend == 'cpu' else 'all',
               LLAMA_ARG_FLASH_ATTN=flash_attn, JEV_IMAGE_CACHE_MIB=str(cache_mib),
               SLOT_CACHE_DIR=str(BASE/'slots'), JEV_BACKEND_LOG=str(BASE/'llama-server.log'))
    if backend in DEFAULT_DEVICES:
        env['GPU_DEVICE'] = device or env.get('GPU_DEVICE') or DEFAULT_DEVICES[backend]
        if backend in ('hip', 'cuda') and flash_attn != 'off':
            passed = fa_tests.get('status') == 'passed' or ('status' not in fa_tests and fa_tests.get('passed') == 252)
            if not passed or fa_tests.get('device') != env['GPU_DEVICE']:
                print(f'WARNING: FA160 checks are not recorded as passed for {env["GPU_DEVICE"]} on this build. '
                      'Continuing; optional checks: python3 -m tools.verify_sarashina_fa '
                      f'--backend {backend} --device {env["GPU_DEVICE"]}', file=sys.stderr)
            else:
                print(f'FA160: recorded check on {fa_tests["device"]}; this does not identify the current physical GPU.', file=sys.stderr)
            if any(env.get(key, 'f16') != 'f16' for key in ('LLAMA_ARG_CACHE_TYPE_K', 'LLAMA_ARG_CACHE_TYPE_V')):
                print('WARNING: FA160 numerical checks cover F16 KV only; other KV types use upstream conversion/support rules. Continuing.', file=sys.stderr)
    else:
        env.pop('GPU_DEVICE', None)
    env.setdefault('CTX_SIZE', '8192')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, default='hip')
    parser.add_argument('--device', help='llama.cpp device(s), e.g. CUDA0 or ROCm1; overrides GPU_DEVICE')
    parser.add_argument('--model', choices=('sarashina', 'sarashina-q8'), default='sarashina')
    parser.add_argument('--mmproj', type=Path, default=ROOT/'models/sarashina2.2-vision-3b.mmproj-jev-f16.gguf')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--flash-attn', choices=('auto', 'off', 'on'), default=os.environ.get('LLAMA_ARG_FLASH_ATTN', 'auto'))
    parser.add_argument('--image-cache-mib', type=int, default=int(os.environ.get('JEV_IMAGE_CACHE_MIB', '128')))
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend-port', type=int, default=8097)
    parser.add_argument('--state-cache', choices=('auto', 'off', 'shared'), default='auto')
    args = parser.parse_args()
    env = launch_environment(args.backend, args.model, args.mmproj, args.text_only, args.flash_attn, args.image_cache_mib, args.device)
    command = [sys.executable, str(ROOT/'scripts/run_local.py'), '--model', args.model,
               '--port', str(args.port), '--backend-port', str(args.backend_port), '--state-cache', args.state_cache]
    print(f'Experimental Sarashina: {args.backend}, FA={args.flash_attn}, image cache={args.image_cache_mib} MiB', flush=True)
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f'Cannot start Sarashina experiment: {exc}')
