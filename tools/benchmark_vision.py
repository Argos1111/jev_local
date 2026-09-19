#!/usr/bin/env python3
"""Measure image request latency against an already running local API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

from image_input import load_image, validate_images

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--input', type=Path, default=ROOT/'examples/vision_gss_4.json')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'results/vision_benchmark.json')
    args = parser.parse_args()
    if args.rounds < 1: parser.error('--rounds must be positive')
    try:
        payload = json.loads(args.input.read_text())
        payload['images'] = payload.get('images', []) + [load_image(args.image)]
        validate_images(payload['images'])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    body = json.dumps(payload, ensure_ascii=False).encode()
    report = {'created_at':datetime.now(timezone.utc).isoformat(),
              'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
              'request_without_images':{k:v for k,v in payload.items() if k != 'images'},
              'url':args.url, 'runs':[],
              'timing_scope':'HTTP request through complete JSON response; excludes model loading and file/base64 processing'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.rounds + 1):
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/systemone', body,
            {'Content-Type':'application/json', 'Authorization':'Bearer ' + os.environ.get('JEV_API_KEY','local-dev')})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
            headers = {k:v for k,v in response.headers.items() if k.lower().startswith('x-jev-')}
        elapsed = (time.perf_counter() - start) * 1000
        row = {'elapsed_ms':elapsed, 'response':result, 'headers':headers}
        if i == 0: report['warmup'] = row
        else: report['runs'].append(row)
        print(f'{"warmup" if i == 0 else f"run {i}"}: {elapsed:.1f} ms', flush=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    values = [r['elapsed_ms'] for r in report['runs']]
    report['summary_ms'] = {'median':statistics.median(values), 'mean':statistics.mean(values),
                            'min':min(values), 'max':max(values)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary_ms'], indent=2))


if __name__ == '__main__': main()
