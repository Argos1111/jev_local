#!/usr/bin/env python3
"""Live VL smoke test: identical question, red versus blue PNG, plus text cache."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import struct
import urllib.request
import zlib


def solid_png(rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes(rgb)*224)*224)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    def call(payload):
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/systemone', json.dumps(payload).encode(), {
            'Content-Type':'application/json', 'Authorization':'Bearer ' + os.environ.get('JEV_API_KEY', 'local-dev')})
        with urllib.request.urlopen(request, timeout=180) as response:
            return {'response':json.load(response), 'cache_mode':response.headers.get('X-Jev-Local-State-Cache')}

    payloads = {}
    for color, rgb in [('red', (255, 0, 0)), ('blue', (0, 0, 255))]:
        payloads[color] = {'model':'jev-latest', 'state':'Look at the attached image.',
            'images':['data:image/png;base64,' + base64.b64encode(solid_png(rgb)).decode()],
            'questions':{'color':{'type':'choice', 'instructions':'What is the dominant color in the image?',
                                  'criteria':{'red':'Red', 'blue':'Blue', 'green':'Green'}}}}
    payloads['text'] = {'model':'jev-latest', 'state':'The customer requests a refund. ' * 70,
                       'questions':{'refund':{'type':'noul', 'instructions':'Is a refund requested?'},
                                    'route':{'type':'choice', 'instructions':'Which department should handle this?',
                                             'criteria':{'billing':'Payments and refunds', 'technical':'Technical issues'}}}}
    # Exercise slot leasing while text snapshots and images are both in flight.
    with ThreadPoolExecutor(max_workers=3) as pool:
        reports = dict(zip(payloads, pool.map(call, payloads.values())))
    for color in ('red', 'blue'):
        answer = reports[color]['response']['answers']['color']
        assert answer['choice'] == color, reports[color]
        assert abs(sum(answer['probabilities'].values()) - 1) < 1e-9
        assert reports[color]['cache_mode'] == 'off-images'
    assert reports['text']['cache_mode'] == 'shared', reports['text']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    print('PASS: red/blue image inference, normalized probabilities, concurrent text shared cache')


if __name__ == '__main__':
    main()
