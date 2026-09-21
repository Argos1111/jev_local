#!/usr/bin/env python3
"""Compare native pixels/embeddings with the hash-checked official checkpoint and AutoProcessor."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('.cache/sarashina-official'))
    p.add_argument('--native', type=Path, default=Path('.cache/sarashina-official-runtime/build-hip/bin/test_sarashina_embedding'))
    p.add_argument('--mmproj', type=Path, default=Path('models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf'))
    p.add_argument('--output', type=Path, default=Path('results/sarashina-official/embeddings'))
    p.add_argument('--allow-reviewed-code', action='store_true',
                   help='Execute the hash-checked local official AutoProcessor Python')
    p.add_argument('--device', default='ROCm0')
    p.add_argument('--torch-device', default='cpu')
    p.add_argument('--reference-half-pixels', action='store_true', help='Diagnostic: match ggml Conv2D F16 im2col input rounding')
    p.add_argument('--min-cosine', type=float, default=0.999)
    p.add_argument('--max-rmse', type=float, default=0.04)
    p.add_argument('--flash-attn', type=int, choices=(-1, 0, 1), default=0)
    p.add_argument('--pixels-only', action='store_true', help='Test resize/normalization boundaries without loading encoder weights')
    p.add_argument('--native-library-dir', help='Additional CUDA/HIP library path; otherwise use the build manifest/environment')
    args = p.parse_args()
    if not args.allow_reviewed_code:
        p.error('Review processing_sarashina2_vision.py locally, then pass --allow-reviewed-code')
    from scripts.fetch_sarashina_reference import verify_reference
    lock = verify_reference(args.reference)
    import numpy as np
    from importlib.metadata import version
    from scripts.build_sarashina import RUNTIME_VARS
    from scripts.setup_runtime import sha256
    from PIL import Image, ImageDraw
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

    torch.set_num_threads(8)
    config = json.loads((args.reference / 'config.json').read_text())
    from scripts.convert_sarashina_mmproj import validate_config
    validate_config(config)
    if not args.pixels_only:
        cfg = Qwen2VLVisionConfig(**config['vision_config'])
        cfg._attn_implementation = 'eager'
        visual = Qwen2VisionTransformerPretrainedModel(cfg).eval()
        norm = torch.nn.LayerNorm(2560).eval()
        with safe_open(args.reference / lock['checkpoint_file'], framework='pt') as f:
            visual.load_state_dict({k.removeprefix('visual.'): f.get_tensor(k).float() for k in f.keys() if k.startswith('visual.')}, strict=True)
            norm.load_state_dict({k.removeprefix('norm.'): f.get_tensor(k).float() for k in f.keys() if k.startswith('norm.')}, strict=True)
        visual.to(args.torch_device)
        norm.to(args.torch_device)
    processor = AutoProcessor.from_pretrained(args.reference, trust_remote_code=True,
                                              local_files_only=True, use_fast=False).image_processor
    args.output.mkdir(parents=True, exist_ok=True)
    images = {name: Image.new('RGB', (224, 224), name) for name in ('red', 'blue')}
    image = Image.new('RGB', (224, 224), 'white')
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((16, 16, 100, 100), fill='red')
    drawing.ellipse((120, 110, 210, 200), fill='blue')
    images['shapes'] = image
    y, x = np.mgrid[:157, :233]
    images['gradient'] = Image.fromarray(np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype('uint8'))
    sample = Path('sample_pics/20260911_image_resized.png')
    if sample.is_file():
        images['gss'] = Image.open(sample).convert('RGB')
    if args.pixels_only:
        for name, size in {'half-even': (238, 182), 'half-odd': (266, 210), 'tiny': (7, 13),
                           'narrow': (14, 224), 'large': (2048, 1536), 'aspect-200': (2000, 10),
                           'official-max-area': (1008, 1008), 'full-document-size': (1190, 665)}.items():
            y, x = np.mgrid[:size[1], :size[0]]
            images[name] = Image.fromarray(np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype('uint8'))
        y, x = np.mgrid[:61, :101]
        checker = (((x//3 + y//3) % 2) * 255).astype('uint8')
        images['checkerboard'] = Image.fromarray(np.stack([checker]*3, axis=-1))
        images['noise'] = Image.fromarray(np.random.default_rng(0).integers(0, 256, (193, 317, 3), dtype='uint8'))
    native_env = dict(os.environ)
    manifest_path = args.native.parent.parent/'jev-build.json'
    if manifest_path.is_file():
        saved_env = json.loads(manifest_path.read_text()).get('runtime_env', {})
        for key in RUNTIME_VARS:
            if not native_env.get(key) and saved_env.get(key):
                native_env[key] = saved_env[key]
    library_dirs = [str(args.native.parent.resolve())]
    if args.native_library_dir:
        library_dirs.append(args.native_library_dir)
    if native_env.get('LD_LIBRARY_PATH'):
        library_dirs.append(native_env['LD_LIBRARY_PATH'])
    native_env['LD_LIBRARY_PATH'] = os.pathsep.join(library_dirs)
    native_env['JEV_IMAGE_CACHE_MIB'] = '0'
    report = {'transformers_dtype': 'float32', 'torch_device': args.torch_device,
              'reference_lock': lock, 'scope': 'Official AutoProcessor and checkpoint; float32 encoder formula',
              'versions': {name: version(name) for name in ('torch', 'transformers', 'pillow', 'numpy')},
              'input_sha256': {str(p): sha256(p) for p in (args.native, args.mmproj)},
              'native_device': args.device, 'native_fa': args.flash_attn, 'pixels_only': args.pixels_only,
              'reference_half_pixels': args.reference_half_pixels,
              'limits': {'min_cosine': args.min_cosine, 'max_rmse': args.max_rmse, 'pixel_max_abs': 1e-6},
              'cases': {}}
    for name, image in images.items():
        stem = args.output/name
        image.save(stem.with_suffix('.png'))
        stem.with_suffix('.rgb').write_bytes(image.tobytes())
        command = [str(args.native), str(args.mmproj), str(stem.with_suffix('.rgb')),
                   str(image.width), str(image.height), str(stem), args.device, str(args.flash_attn)]
        if args.pixels_only:
            command.append('pixels-only')
        with stem.with_suffix('.log').open('w') as log:
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=log, env=native_env)
        shape = json.loads(result.stdout)
        inputs = processor(images=[image], return_tensors='pt').to(args.torch_device)
        if not args.pixels_only:
            with torch.inference_mode():
                model_pixels = inputs['pixel_values'].half().float() if args.reference_half_pixels else inputs['pixel_values']
                projected = visual(model_pixels, inputs['image_grid_thw'])
                reference = norm(projected).float().cpu().numpy()
            np.save(stem.with_suffix('.reference.npy'), reference)
            native = np.fromfile(str(stem)+'.embd.f32', dtype=np.float32).reshape(-1, 2560)
        # Undo the official processor's temporal duplication and patch ordering.
        gh, gw = inputs['image_grid_thw'][0, 1:].cpu().tolist()
        pixels = inputs['pixel_values'].float().cpu().numpy()
        pixels = pixels.reshape(1, gh//2, gw//2, 2, 2, 3, 2, 14, 14)
        pixels = pixels[0, :, :, :, :, :, 0].transpose(0, 2, 5, 1, 3, 6, 4).reshape(gh*14, gw*14, 3)
        native_pixels = np.fromfile(str(stem)+'.pixels.f32', dtype=np.float32).reshape(shape['height'], shape['width'], 3)
        if pixels.shape != native_pixels.shape:
            raise AssertionError((pixels.shape, native_pixels.shape))
        finite = np.isfinite(pixels).all() and np.isfinite(native_pixels).all()
        row = {**shape, 'pixel_max_abs': float(np.max(np.abs(pixels - native_pixels)))}
        row['passed'] = bool(finite and row['pixel_max_abs'] <= 1e-6)
        if not args.pixels_only:
            if native.shape != reference.shape:
                raise AssertionError((native.shape, reference.shape))
            finite = finite and np.isfinite(native).all() and np.isfinite(reference).all()
            difference = native - reference
            row.update(rmse=float(np.sqrt(np.mean(difference**2))), max_abs=float(np.max(np.abs(difference))),
                       cosine=float(np.sum(native*reference) / np.sqrt(np.sum(native**2)*np.sum(reference**2))),
                       native_std=float(native.std()), reference_std=float(reference.std()))
            row['passed'] = bool(row['passed'] and finite and row['rmse'] <= args.max_rmse and row['cosine'] >= args.min_cosine)
        row['finite'] = bool(finite)
        report['cases'][name] = row
        report['passed'] = all(r['passed'] for r in report['cases'].values())
        print(name, row, flush=True)
        (args.output/'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('Embedding comparison exceeded tolerance; see comparison.json')


if __name__ == '__main__':
    main()
