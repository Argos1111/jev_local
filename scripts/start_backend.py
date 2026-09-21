#!/usr/bin/env python3
"""Launch the selected runtime, allowing explicit environment overrides."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__:
    from .setup_runtime import ROOT, MODEL_PROFILES, runtime_env, model_profile, model_specs
else:
    from setup_runtime import ROOT, MODEL_PROFILES, runtime_env, model_profile, model_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_PROFILES)
    options, extra = parser.parse_known_args()
    profile = model_profile(ROOT, options.model)
    selected = ROOT/'.cache/runtime/selected.json'
    selection = json.loads(selected.read_text()) if selected.exists() else {}
    explicit = os.environ.get('LLAMA_SERVER')
    server = Path(explicit) if explicit else ROOT/selection.get('server', '.cache/runtime/llama/llama-server')
    lock = json.loads((ROOT/'scripts/runtime.json').read_text())
    specs = model_specs(lock, profile)
    model = Path(os.environ.get('LFM_MODEL', str(ROOT/'models'/specs[0]['filename'])))
    # An explicit custom model needs its own projector; never pair it with ours.
    default_projector = str(ROOT/'models'/specs[1]['filename']) if len(specs) > 1 and 'LFM_MODEL' not in os.environ else ''
    projector = os.environ.get('LFM_MMPROJ', default_projector)
    if not server.is_file() or not os.access(server, os.X_OK):
        sys.exit(f'llama-server not found: {server}. Run ./setup.sh or set LLAMA_SERVER.')
    if not model.is_file():
        sys.exit(f'Model not found: {model}. Run ./setup.sh --model {profile} --model-only or set LFM_MODEL.')
    if projector and not Path(projector).is_file():
        sys.exit(f'Projector not found: {projector}. Run ./setup.sh or set LFM_MMPROJ.')
    directory = server.parent if explicit else ROOT/selection.get('directory', '.cache/runtime/llama')
    env = runtime_env(directory)
    layers = os.environ.get('GPU_LAYERS', '0' if explicit else selection.get('gpu_layers', '0'))
    slot_dir = Path(os.environ.get('SLOT_CACHE_DIR', str(ROOT/'.cache/slots'))).resolve()
    slot_dir.mkdir(parents=True, exist_ok=True)
    args = [str(server), '-m', str(model), '--host', '127.0.0.1', '--port', env.get('PORT', '8097'),
            '-ngl', layers, '-c', env.get('CTX_SIZE', '8192'), '--slot-save-path', str(slot_dir),
            '-np', '4', '-t', env.get('THREADS', '8')]
    if env.get('GPU_DEVICE'):
        args.extend(['--device', env['GPU_DEVICE']])
    if projector:
        args.extend(['--mmproj', projector])
        if layers == '0':
            args.append('--no-mmproj-offload')
    args.extend(extra)
    print(f'Model profile: {profile}; model: {model.name}', flush=True)
    print(f'Backend: {selection.get("variant", "manual") if not explicit else "manual"}; GPU layers: {layers}', flush=True)
    os.execve(server, args, env)


if __name__ == '__main__':
    main()
