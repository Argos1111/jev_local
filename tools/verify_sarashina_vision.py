#!/usr/bin/env python3
"""Image sensitivity, option-order, and cache-isolation checks for the Sarashina experiment.

Uses bordered shapes, not solid colors (which the reference clone also misreads).
Reports the solid-color limitation separately; it is not silently counted as a pass.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, default=Path('results/sarashina-repair/vision-smoke.json'))
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    def image_url(image):
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    images = {}
    for color in ('red', 'blue', 'green'):
        image = Image.new('RGB', (224, 224), 'white')
        ImageDraw.Draw(image).rectangle((30, 30, 194, 194), fill=color)
        images[color] = image_url(image)
    def call(images, reverse=False, question='画像の中央にある四角形は何色ですか？'):
        criteria = {'red': '赤色', 'blue': '青色', 'green': '緑色'}
        if reverse:
            criteria = dict(reversed(list(criteria.items())))
        payload = {'model': 'jev-latest', 'state': '添付画像を見て答えてください。',
                   'questions': {'color': {'type': 'choice', 'instructions': question, 'criteria': criteria}}}
        if images:
            payload['images'] = images
        request = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone', json.dumps(payload).encode(),
            {'Content-Type': 'application/json', 'Authorization': 'Bearer '+os.environ.get('JEV_API_KEY', 'local-dev')})
        with urllib.request.urlopen(request, timeout=180) as response:
            return {'response': json.load(response), 'cache_mode': response.headers.get('X-Jev-Local-State-Cache')}
    report = {'cases': [], 'scope': 'Synthetic shape sensitivity, not a general vision benchmark'}
    # Repeated/interleaved images must not collide in the encoder cache.
    for reverse in (False, True):
        for color in ('red', 'blue', 'red', 'green', 'blue'):
            result = call([images[color]], reverse)
            answer = result['response']['answers']['color']
            report['cases'].append({'color': color, 'reverse': reverse, **result,
                'passed': answer['choice'] == color and abs(sum(answer['probabilities'].values())-1) < 1e-9
                          and result['cache_mode'] == 'off-images'})
    def text_call():
        payload = {'model': 'jev-latest', 'state': '顧客は二重請求の返金を要求しています。'*70,
                   'questions': {'refund': {'type': 'noul', 'instructions': '返金を要求しているか？'},
                                 'route': {'type': 'choice', 'instructions': '担当部署は？',
                                           'criteria': {'billing': '請求・返金', 'technical': '技術的な障害'}}}}
        request = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone', json.dumps(payload).encode(),
            {'Content-Type': 'application/json', 'Authorization': 'Bearer '+os.environ.get('JEV_API_KEY', 'local-dev')})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = {'response': json.load(response), 'cache_mode': response.headers.get('X-Jev-Local-State-Cache')}
        result['passed'] = result['cache_mode'] == 'shared'
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        text = pool.submit(text_call)
        concurrent = list(pool.map(lambda color: call([images[color]]), ('red', 'blue', 'green')))
        report['concurrent_text_shared_cache'] = text.result()
    report['concurrent'] = [dict(color=c, **r, passed=r['response']['answers']['color']['choice'] == c
                                and r['cache_mode'] == 'off-images')
                            for c, r in zip(('red', 'blue', 'green'), concurrent)]
    report['multi_image'] = []
    for colors in (('red', 'blue'), ('blue', 'red')):
        result = call([images[c] for c in colors], question='2枚目の画像の中央にある四角形は何色ですか？')
        report['multi_image'].append({'order': colors, **result,
            'passed': result['response']['answers']['color']['choice'] == colors[1]})
    report['no_image_control'] = call([])
    report['known_solid_color_limitation'] = {}
    for color in ('red', 'blue'):
        result = call([image_url(Image.new('RGB', (224, 224), color))], question='画像全体は何色ですか？')
        report['known_solid_color_limitation'][color] = {**result,
            'passed': result['response']['answers']['color']['choice'] == color}
    report['single_image_passed'] = all(r['passed'] for group in ('cases', 'concurrent') for r in report[group])
    report['multi_image_passed'] = all(r['passed'] for r in report['multi_image'])
    report['solid_color_passed'] = all(r['passed'] for r in report['known_solid_color_limitation'].values())
    report['passed'] = (report['single_image_passed'] and report['multi_image_passed']
                        and report['solid_color_passed'] and report['concurrent_text_shared_cache']['passed'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for group in ('cases', 'concurrent', 'multi_image'):
        print(group, sum(r['passed'] for r in report[group]), '/', len(report[group]))
    print('Concurrent text shared cache:', report['concurrent_text_shared_cache']['passed'])
    print('Solid colors', sum(r['passed'] for r in report['known_solid_color_limitation'].values()), '/ 2')
    print('Full report:', args.output)
    if not report['passed']:
        raise SystemExit('Image sensitivity check failed; see report')


if __name__ == '__main__':
    main()
