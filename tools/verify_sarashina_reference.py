#!/usr/bin/env python3
"""Local checkpoint generation, using native token IDs to isolate vision/runtime errors.

Executes the reviewed local modeling/configuration files via trust_remote_code.
Requires all checkpoint shards, Transformers 4.57.1, torch and accelerate.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('.cache/sarashina-reference'))
    p.add_argument('--url', default='http://127.0.0.1:28197')
    p.add_argument('--output', type=Path, default=Path('results/sarashina-repair/reference-generation.json'))
    p.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--multi-image', action='store_true', help='Also compare two-image order handling')
    p.add_argument('--allow-reviewed-code', action='store_true',
                   help='Execute the hash-checked local Python model implementation')
    args = p.parse_args()
    if not args.allow_reviewed_code:
        p.error('Review the local modeling/configuration Python, then pass --allow-reviewed-code')
    from scripts.setup_runtime import ROOT, sha256
    lock = json.loads((ROOT/'native/sarashina-reference.json').read_text())
    for name, spec in lock['files'].items():
        if not (args.reference/name).is_file() or sha256(args.reference/name) != spec['sha256']:
            p.error(f'Missing or changed reference file: {name}')
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForCausalLM, Qwen2VLImageProcessor

    torch.set_num_threads(8)
    def post(path, payload):
        req = urllib.request.Request(args.url+path, json.dumps(payload, ensure_ascii=False).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    def tokens(text):
        return post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']
    def decode(ids):
        return post('/detokenize', {'tokens': ids})['content']

    model = AutoModelForCausalLM.from_pretrained(args.reference, trust_remote_code=True,
                local_files_only=True, dtype=getattr(torch, args.dtype), device_map={'': 'cuda:0'},
                attn_implementation='sdpa').eval()
    processor = Qwen2VLImageProcessor(image_mean=[.5]*3, image_std=[.5]*3)
    cases = []
    for color in ('red', 'blue', 'green', 'black', 'white'):
        cases.append((f'solid-{color}', Image.new('RGB', (224, 224), color),
                      'この画像全体は何色ですか？色の名前だけを日本語で答えてください。'))
    for color in ('red', 'blue'):
        image = Image.new('RGB', (224, 224), 'white')
        ImageDraw.Draw(image).rectangle((30, 30, 194, 194), fill=color)
        cases.append((f'square-{color}', image, '図形は何色ですか？色の名前だけを日本語で答えてください。'))
    image = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 100, 100), fill='red')
    draw.ellipse((120, 110, 210, 200), fill='blue')
    cases.append(('shapes', image, 'この画像には何が写っていますか？日本語で簡潔に答えてください。'))
    if args.multi_image:
        squares = {}
        for color in ('red', 'blue'):
            square = Image.new('RGB', (224, 224), 'white')
            ImageDraw.Draw(square).rectangle((30, 30, 194, 194), fill=color)
            squares[color] = square
        for first, second in (('red', 'blue'), ('blue', 'red')):
            cases.append((f'pair-{first}-{second}', [squares[first], squares[second]],
                '2枚目の画像の中央にある四角形は何色ですか？色の名前だけを日本語で答えてください。'))
    from importlib.metadata import version
    report = {'dtype': args.dtype, 'source': str(args.reference), 'reference_lock': lock,
              'versions': {name: version(name) for name in ('torch', 'transformers', 'pillow', 'numpy')},
              'scope': 'Public-clone reference with reconstructed Qwen processor, not gated official AutoProcessor',
              'cases': []}
    with urllib.request.urlopen(args.url+'/props') as r:
        props = json.load(r)
    marker = props['media_marker']
    report['native'] = {key: props.get(key) for key in ('build_info', 'model_path', 'modalities', 'media_marker')}
    model_path = Path(props.get('model_path', ''))
    if model_path.is_file():
        report['native']['model_sha256'] = sha256(model_path)
    with torch.inference_mode():
        for name, image, question in cases:
            images = image if isinstance(image, list) else [image]
            vision = processor(images=images, return_tensors='pt').to('cuda:0')
            ids = tokens('<|user|>')
            for grid in vision['image_grid_thw']:
                ids += tokens('<|prefix|>') + [14]*(int(grid.prod().item()) // 4) + tokens('<|suffix|>')
            ids += tokens(question+'</s><|assistant|>')
            inputs = {'input_ids': torch.tensor([ids], device='cuda:0'),
                      'attention_mask': torch.ones((1, len(ids)), device='cuda:0', dtype=torch.long),
                      **vision}
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=80, eos_token_id=2, pad_token_id=3)
            output_ids = generated[0, len(ids):].tolist()
            text = decode(output_ids)
            encoded = []
            for image in images:
                data = io.BytesIO()
                image.save(data, format='PNG')
                encoded.append(base64.b64encode(data.getvalue()).decode())
            native = post('/completion', {'prompt': {
                'prompt_string': '<|user|>'+marker*len(images)+question+'</s><|assistant|>',
                'multimodal_data': encoded},
                'temperature': 0, 'cache_prompt': False, 'n_predict': 80})
            row = {'case': name, 'reference': text, 'native': native['content'],
                   'reference_output_ids': output_ids, 'input_tokens': len(ids),
                   'native_input_tokens': native['tokens_evaluated'],
                   'same_text': text.removesuffix('</s>') == native['content'],
                   'same_token_count': len(ids) == native['tokens_evaluated']}
            report['cases'].append(row)
            print(row, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report['exact_text_matches'] = sum(r['same_text'] for r in report['cases'])
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not all(r['same_token_count'] for r in report['cases']):
        raise SystemExit('Reference/native input token counts differ')


if __name__ == '__main__':
    main()
