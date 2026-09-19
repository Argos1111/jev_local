#!/usr/bin/env python3
"""Evaluate the GSS image questions, with a no-image control and saved evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

from image_input import load_image

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'intrusion_method':'vpn_vulnerability', 'detection_date':'june_25',
    'potentially_leaked_count':'246000', 'count_breakdown':'officials_189k_contractors_57k',
    'red_dashed_arrow':'possible_outbound_leak', 'recurrence_prevention':'management_and_connection',
    'general_public_included':False, 'sensitive_identifiers_included':False,
    'account_stopped':True, 'external_investigation_support':True,
    'leak_confirmation_level':1, 'containment_progress':2,
}


def grade(response):
    rows = {}
    for key, expected in EXPECTED.items():
        answer = response['answers'][key]
        if answer['type'] == 'choice':
            predicted = answer['choice']
            probability = answer['probabilities'][expected]
        elif answer['type'] == 'noul':
            predicted = answer['noul'] >= .5
            probability = answer['noul'] if expected else 1 - answer['noul']
        else:
            predicted = int(max(answer['probabilities'], key=answer['probabilities'].get))
            probability = answer['probabilities'][str(expected)]
        rows[key] = {'expected':expected, 'predicted':predicted,
                     'match':predicted == expected, 'expected_probability':probability}
    return {'matched':sum(row['match'] for row in rows.values()), 'total':len(rows), 'questions':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--backend-url', default='http://127.0.0.1:8097')
    parser.add_argument('--image', type=Path, required=True, help='Local GSS image; not bundled with the repository')
    parser.add_argument('--output', type=Path, default=ROOT/'results/vision_gss/evaluation.json')
    parser.add_argument('--reverse-choices', action='store_true', help='Reverse Choice option order for a position-bias control')
    args = parser.parse_args()
    if not args.image.is_file(): parser.error('--image must be an existing local PNG/JPEG')
    source = ROOT/'examples/vision_gss.json'
    payload = json.loads(source.read_text())
    if args.reverse_choices:
        for question in payload['questions'].values():
            if question['type'] == 'choice':
                question['criteria'] = dict(reversed(list(question['criteria'].items())))
    raw_image = args.image.read_bytes()
    report = {'created_at':datetime.now(timezone.utc).isoformat(),
              'image':{'path':str(args.image.resolve()), 'bytes':len(raw_image),
                       'sha256':hashlib.sha256(raw_image).hexdigest(),
                       'dimensions':list(struct.unpack('>II', raw_image[16:24])) if raw_image.startswith(b'\x89PNG\r\n\x1a\n') else None},
              'question_file':str(source), 'question_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'request_without_image':payload, 'reverse_choices':args.reverse_choices,
              'grading':'Choice: selected option; Noul: P(true)>=0.5; Score: most probable level, not rounded expectation.',
              'runs':{}}
    with urllib.request.urlopen(args.backend_url.rstrip('/')+'/props', timeout=10) as response:
        props = json.load(response)
    report['backend'] = {key:props.get(key) for key in ['model_path','modalities','build_info','total_slots','default_generation_settings']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in ('image', 'no_image_control'):
        request_payload = dict(payload)
        if mode == 'image': request_payload['images'] = [load_image(args.image)]
        request = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone', json.dumps(request_payload, ensure_ascii=False).encode(),
            {'Content-Type':'application/json', 'Authorization':'Bearer '+os.environ.get('JEV_API_KEY','local-dev')})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
            headers = {k:v for k,v in response.headers.items() if k.lower().startswith('x-jev-')}
        run = {'elapsed_ms':(time.perf_counter()-start)*1000, 'headers':headers, 'response':result, 'grading':grade(result)}
        report['runs'][mode] = run
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f'{mode}: {run["grading"]["matched"]}/12, {run["elapsed_ms"]:.1f} ms', flush=True)
        for key,row in run['grading']['questions'].items():
            print(f'  {key}: {row["predicted"]} (expected={row["expected"]}, P(expected)={row["expected_probability"]:.4f})', flush=True)


if __name__ == '__main__':
    main()
