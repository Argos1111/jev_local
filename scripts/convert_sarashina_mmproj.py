#!/usr/bin/env python3
"""Convert Sarashina2.2 Vision's encoder, merger and final LayerNorm to F16 GGUF.

The official single-file checkpoint contains visual.* and norm.*; LLM tensors
are not loaded. This private projector needs the official-preprocessing update
to the patched b11042 runtime. No download or checkpoint Python execution occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .fetch_sarashina_reference import reference_lock
else:
    from fetch_sarashina_reference import reference_lock

PREPROCESSING = 'sarashina_official_v1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_config(config):
    v = config.get('vision_config', {})
    expected = {'embed_dim': 1152, 'hidden_size': 2560, 'num_heads': 16,
                'depth': 27, 'patch_size': 14, 'spatial_merge_size': 2,
                'temporal_patch_size': 2, 'hidden_act': 'gelu_pytorch_tanh', 'in_channels': 3}
    if (config.get('model_type') != 'sarashina2_vision'
            or any(v.get(k) != value for k, value in expected.items())
            or int(v.get('mlp_ratio', 0) * 1152) != 4304
            or config.get('text_config', {}).get('hidden_size') != 2560
            or [config.get(k) for k in ('image_token_index', 'start_image_token_index', 'end_image_token_index')]
               != [14, 102397, 102398]):
        raise ValueError('Expected Sarashina2.2 Vision 3B configuration')


def validate_preprocessor(config):
    expected = {'image_processor_type': 'Sarashina2VisionImageProcessor',
                'do_resize': True, 'do_rescale': True, 'do_normalize': True, 'do_convert_rgb': True,
                'image_mean': [0.5]*3, 'image_std': [0.5]*3, 'rescale_factor': 1/255,
                'patch_size': 14, 'temporal_patch_size': 2, 'merge_size': 2,
                'min_pixels': 3136, 'max_pixels': 1016064}
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError('Expected official Sarashina2.2 Vision 3B preprocessing configuration')
    # The official Python uses F.interpolate(mode="bicubic"), ignoring resample.


def source_metadata(config_path, checkpoint_path, preprocessor_path):
    lock = reference_lock()
    inputs = {'config.json': config_path, lock['checkpoint_file']: checkpoint_path,
              'preprocessor_config.json': preprocessor_path}
    hashes = {}
    for name, path in inputs.items():
        spec = lock['files'][name]
        digest = sha256(path)
        if path.stat().st_size != spec['size'] or digest != spec['sha256']:
            raise ValueError(f'Expected checksum-pinned official reference input: {path}')
        hashes[name] = digest
    return {'source_repository': lock['repository'], 'source_revision': lock['revision'],
            'checkpoint_file': lock['checkpoint_file'], 'config_sha256': hashes['config.json'],
            'checkpoint_sha256': hashes[lock['checkpoint_file']],
            'preprocessor_sha256': hashes['preprocessor_config.json'], 'preprocessing': PREPROCESSING}


def tensor_layout():
    """Checkpoint name -> (GGUF name, PyTorch shape); QKV/Conv3D split separately."""
    layout = {}
    for i in range(27):
        for src, dst, shape in [('attn.proj', 'attn_out', (1152, 1152)),
                                ('mlp.fc1', 'ffn_up', (4304, 1152)),
                                ('mlp.fc2', 'ffn_down', (1152, 4304)),
                                ('norm1', 'ln1', (1152,)), ('norm2', 'ln2', (1152,))]:
            for suffix in ('weight', 'bias'):
                layout[f'visual.blocks.{i}.{src}.{suffix}'] = (
                    f'v.blk.{i}.{dst}.{suffix}', shape if suffix == 'weight' else shape[:1])
        for suffix, shape in [('weight', (3456, 1152)), ('bias', (3456,))]:
            layout[f'visual.blocks.{i}.attn.qkv.{suffix}'] = (None, shape)
    for src, dst, shape in [('visual.merger.ln_q', 'v.post_ln', (1152,)),
                            ('visual.merger.mlp.0', 'mm.0', (4608, 4608)),
                            ('visual.merger.mlp.2', 'mm.2', (2560, 4608)),
                            ('norm', 'mm.post_norm', (2560,))]:
        for suffix in ('weight', 'bias'):
            layout[f'{src}.{suffix}'] = (f'{dst}.{suffix}', shape if suffix == 'weight' else shape[:1])
    layout['visual.patch_embed.proj.weight'] = (None, (1152, 3, 2, 14, 14))
    return layout


def convert(config_path, checkpoint_path, output, outtype='f16', preprocessor_path=None):
    config_path, checkpoint_path, output = map(Path, (config_path, checkpoint_path, output))
    preprocessor_path = Path(preprocessor_path) if preprocessor_path else config_path.with_name('preprocessor_config.json')
    config = json.loads(config_path.read_text())
    preprocessing = json.loads(preprocessor_path.read_text())
    validate_config(config)
    validate_preprocessor(preprocessing)
    if outtype not in ('f16', 'f32'):
        raise ValueError('outtype must be f16 or f32')
    manifest = output.with_suffix(output.suffix + '.json')
    temporary = output.with_suffix(output.suffix + '.partial')
    if any(p.exists() or p.is_symlink() for p in (output, manifest, temporary)):
        raise FileExistsError(f'Refusing to overwrite {output}, its manifest or partial output')
    source = source_metadata(config_path, checkpoint_path, preprocessor_path)
    import gguf
    import numpy as np
    from safetensors import safe_open

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(temporary, 'clip')
    writer.add_type('mmproj')
    writer.add_name('Sarashina2.2 Vision 3B - complete projector')
    writer.add_string('general.description', 'Official checkpoint conversion with post-merger LayerNorm; requires jev-sarashina official preprocessing v1')
    writer.add_string('clip.projector_type', 'sarashina2vl')
    writer.add_bool('clip.has_vision_encoder', True)
    writer.add_vision_projection_dim(2560)
    writer.add_vision_image_size(560)
    writer.add_vision_patch_size(14)
    writer.add_vision_embedding_length(1152)
    writer.add_vision_feed_forward_length(4304)
    writer.add_vision_block_count(27)
    writer.add_vision_head_count(16)
    writer.add_vision_image_mean([0.5, 0.5, 0.5])
    writer.add_vision_image_std([0.5, 0.5, 0.5])
    writer.add_vision_attention_layernorm_eps(1e-6)
    writer.add_vision_use_gelu(True)
    writer.add_vision_spatial_merge_size(2)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if outtype == 'f16' else gguf.LlamaFileType.ALL_F32)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32('jev.sarashina.preprocess_version', 1)
    writer.add_uint32('clip.vision.image_min_pixels', preprocessing['min_pixels'])
    writer.add_uint32('clip.vision.image_max_pixels', preprocessing['max_pixels'])
    for key, value in source.items():
        writer.add_string('jev.source.' + key, value)

    layout = tensor_layout()
    names = set()
    def add(name, tensor):
        array = tensor.float().numpy()
        if name.startswith('mm.post_norm.') and array.shape != (2560,):
            raise ValueError(f'Bad final normalization shape: {array.shape}')
        array = np.ascontiguousarray(array, dtype=np.float16 if outtype == 'f16' and array.ndim > 1 else np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite values in {name}')
        if name in names:
            raise ValueError(f'Duplicate output tensor: {name}')
        names.add(name)
        writer.add_tensor(name, array)

    # Reserve the temporary name only after metadata and input hashes are ready.
    temporary.touch(exist_ok=False)
    try:
        with safe_open(checkpoint_path, framework='pt', device='cpu') as checkpoint:
            relevant = {k for k in checkpoint.keys() if k.startswith(('visual.', 'norm.'))}
            if set(layout) != relevant:
                raise ValueError(f'Incomplete/unexpected vision weights: {sorted(set(layout) ^ relevant)}')
            for src, (dst, shape) in layout.items():
                if tuple(checkpoint.get_slice(src).get_shape()) != shape:
                    raise ValueError(f'Unexpected shape for {src}; expected {shape}')
            for src, (dst, _) in layout.items():
                if dst is not None:
                    add(dst, checkpoint.get_tensor(src))
            for i in range(27):
                for suffix in ('weight', 'bias'):
                    qkv = checkpoint.get_tensor(f'visual.blocks.{i}.attn.qkv.{suffix}')
                    if qkv.shape[0] != 3456:
                        raise ValueError('Unexpected QKV size')
                    for name, tensor in zip(('q', 'k', 'v'), qkv.chunk(3, dim=0)):
                        add(f'v.blk.{i}.attn_{name}.{suffix}', tensor)
            patch = checkpoint.get_tensor('visual.patch_embed.proj.weight')
            if tuple(patch.shape) != (1152, 3, 2, 14, 14):
                raise ValueError('Unexpected Conv3D patch embedding size')
            add('v.patch_embd.weight', patch[:, :, 0])
            add('v.patch_embd.weight.1', patch[:, :, 1])
        if len(names) != 442:
            raise ValueError(f'Expected 442 tensors, got {len(names)}')
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        # link() installs atomically without replacing a concurrently created file.
        import os
        os.link(temporary, output)
    finally:
        writer.close()
        temporary.unlink(missing_ok=True)
    report = {**source, 'output_sha256': sha256(output), 'tensors': len(names),
              'projector_type': 'sarashina2vl', 'outtype': outtype}
    with manifest.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', '--shard', dest='checkpoint', type=Path, required=True,
                        help='Official model.safetensors (--shard is a compatibility alias)')
    parser.add_argument('--preprocessor', type=Path, help='Default: preprocessor_config.json next to --config')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outtype', choices=('f16', 'f32'), default='f16', help='F32 is for numerical diagnostics')
    args = parser.parse_args()
    print(json.dumps(convert(args.config, args.checkpoint, args.output, args.outtype, args.preprocessor), indent=2))


if __name__ == '__main__':
    main()
