#!/usr/bin/env python3
"""Official local checkpoint/AutoProcessor generation compared with the native runtime.

Executes only hash-checked local model/processor Python, with explicit permission.
Native tokenization is checked independently against the official tokenizer.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('.cache/sarashina-official'))
    p.add_argument('--url', default='http://127.0.0.1:28197')
    p.add_argument('--output', type=Path, default=Path('results/sarashina-official/reference-generation.json'))
    p.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--torch-device', default='cuda:0')
    p.add_argument('--image', type=Path, help='Also compare a real document image (not resized by this tool)')
    p.add_argument('--multi-image', action='store_true', help='Also compare two-image order handling')
    p.add_argument('--allow-reviewed-code', action='store_true',
                   help='Execute the hash-checked local official model and processor Python')
    args = p.parse_args()
    if not args.allow_reviewed_code:
        p.error('Review the local model/processor Python, then pass --allow-reviewed-code')
    from scripts.setup_runtime import sha256
    from scripts.fetch_sarashina_reference import verify_reference
    lock = verify_reference(args.reference)
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForCausalLM, AutoProcessor

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
                local_files_only=True, dtype=getattr(torch, args.dtype), device_map={'': args.torch_device},
                attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(args.reference, trust_remote_code=True,
                                              local_files_only=True, use_fast=False)
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
    cases.append(('shapes-resized', image.resize((233, 157)),
                  'この画像には何が写っていますか？日本語で簡潔に答えてください。'))
    if args.image:
        cases.append(('document', Image.open(args.image).convert('RGB'),
                      'この文書のタイトルと検知日を読み取ってください。'))
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
    report = {'dtype': args.dtype, 'torch_device': args.torch_device, 'source': str(args.reference), 'reference_lock': lock,
              'processor_class': type(processor).__name__, 'tokenizer_class': type(processor.tokenizer).__name__,
              'versions': {name: version(name) for name in ('torch', 'transformers', 'pillow', 'numpy', 'sentencepiece', 'protobuf')},
              'scope': 'Official checkpoint/AutoProcessor; official vs native token IDs checked independently',
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
            messages = [{'role': 'user', 'content': [*({'type': 'image'} for _ in images),
                                                   {'type': 'text', 'text': question}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=images, text=[prompt], return_tensors='pt').to(args.torch_device)
            ids = inputs['input_ids'][0].tolist()
            native_ids = tokens('<|user|>')
            for grid in inputs['image_grid_thw']:
                native_ids += tokens('<|prefix|>') + [14]*(int(grid.prod().item()) // 4) + tokens('<|suffix|>')
            native_ids += tokens(question+'</s><|assistant|>')
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=80, eos_token_id=2, pad_token_id=3)
            output_ids = generated[0, len(ids):].tolist()
            text = processor.decode(output_ids, skip_special_tokens=False)
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
                   'same_token_count': len(ids) == native['tokens_evaluated'],
                   'same_input_ids': ids == native_ids, 'same_detokenization': text == decode(output_ids)}
            report['cases'].append(row)
            print(row, flush=True)
            row.update(official_input_ids=ids, native_reconstructed_input_ids=native_ids)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report['exact_text_matches'] = sum(r['same_text'] for r in report['cases'])
            report['input_id_matches'] = sum(r['same_input_ids'] for r in report['cases'])
            report['input_token_count_matches'] = sum(r['same_token_count'] for r in report['cases'])
            report['tokenization_passed'] = all(r['same_token_count'] and r['same_input_ids'] and r['same_detokenization']
                                               for r in report['cases'])
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not all(r['same_token_count'] and r['same_input_ids'] and r['same_detokenization'] for r in report['cases']):
        raise SystemExit('Official/native tokenization differs; see report')


if __name__ == '__main__':
    main()
