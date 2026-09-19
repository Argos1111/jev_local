#!/usr/bin/env python3
"""Compare encoder-only reuse off/on on the same experimental GPU binary."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import socket
import subprocess
import sys
import time
import urllib.request

from image_input import load_image
from scripts.run_local import wait_ready

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, default=ROOT/'.cache/vision-build-gcc/bin/llama-server')
    p.add_argument('--output', type=Path, default=ROOT/'results/vision_image_cache')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--input', type=Path, default=ROOT/'examples/vision_gss_4.json')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--device', default=os.environ.get('GPU_DEVICE', 'ROCm0'))
    p.add_argument('--port', type=int, default=19080)
    p.add_argument('--backend-port', type=int, default=19097)
    p.add_argument('--workers', type=int, default=4, help='Use 1 for serial numerical regression checks')
    args = p.parse_args()
    if args.rounds < 1: p.error('--rounds must be positive')
    if not 1 <= args.workers <= 4: p.error('--workers must be 1..4')
    if args.port == args.backend_port or not all(1 <= x <= 65535 for x in (args.port,args.backend_port)):
        p.error('Ports must be distinct and in 1..65535')
    if not args.server.is_file(): p.error('Experimental binary missing; run python3 scripts/build_image_cache.py')
    if not args.input.is_file() or not args.image.is_file(): p.error('--input and --image must be existing files; sample images are not bundled')
    for port in (args.port,args.backend_port):
        with socket.socket() as probe:
            try: probe.bind(('127.0.0.1',port))
            except OSError: p.error(f'Port {port} is busy; choose unused ports')
    api_url = f'http://127.0.0.1:{args.port}'
    backend_url = f'http://127.0.0.1:{args.backend_port}'
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input
    pic = args.image
    payload = json.loads(source.read_text())
    if not payload.get('questions') or any(q.get('type') != 'choice' for q in payload['questions'].values()):
        p.error('This comparison requires Choice questions (e.g. examples/vision_gss_4.json)')
    payload['images'] = [load_image(pic)]
    body = json.dumps(payload, ensure_ascii=False).encode()
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'image_sha256':hashlib.sha256(pic.read_bytes()).hexdigest(),
              'request_without_image':{k:v for k,v in payload.items() if k != 'images'},
              'binary':str(args.server.resolve()), 'binary_sha256':hashlib.sha256(args.server.read_bytes()).hexdigest(),
              'requested_device':args.device, 'gpu_layers':'all', 'ctx_size':32768, 'slots':4,
              'workers':args.workers, 'order':['off','on','on','off'], 'blocks':[]}
    def save():
        (args.output/'measurement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for index, mode in enumerate(report['order']):
        name = f'{index+1}-{mode}'
        env = dict(os.environ, LLAMA_SERVER=str(args.server.resolve()), GPU_LAYERS='all', GPU_DEVICE=args.device,
                   CTX_SIZE='32768', PORT=str(args.backend_port), JEV_IMAGE_CACHE_MIB='128' if mode == 'on' else '0')
        procs = []
        block = {'mode':mode, 'runs':[]}
        logpath = args.output/f'{name}-backend.log'
        try:
            with logpath.open('w') as log, (args.output/f'{name}-api.log').open('w') as api_log:
                procs.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/start_backend.py'), '--model', 'vision'],env=env,stdout=log,stderr=log))
                wait_ready(backend_url,procs[-1])
                procs.append(subprocess.Popen([sys.executable,str(ROOT/'api_server.py'),'--backend-url',backend_url,'--port',str(args.port),'--workers',str(args.workers)],env=env,stdout=api_log,stderr=api_log))
                wait_ready(api_url,procs[-1])
                with urllib.request.urlopen(backend_url+'/props') as r:
                    props=json.load(r)
                if mode == 'on' and 'jev-vision-cache' not in props.get('build_info',''):
                    raise RuntimeError('Backend is not the experimental encoder-cache build')
                block['backend']={k:props.get(k) for k in ['build_info','model_path','modalities','total_slots']}
                for i in range(args.rounds+1):
                    req=urllib.request.Request(api_url+'/v1/systemone',body,
                        {'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('JEV_API_KEY','local-dev')})
                    before=logpath.read_text().count('JEV_IMAGE_CACHE hit')
                    before_misses=logpath.read_text().count('JEV_IMAGE_CACHE miss')
                    t=time.perf_counter()
                    with urllib.request.urlopen(req,timeout=600) as r:
                        response=json.load(r)
                        headers={k:v for k,v in r.headers.items() if k.lower().startswith('x-jev-')}
                    elapsed=(time.perf_counter()-t)*1000
                    assert headers['X-Jev-Local-State-Cache']=='off-images'
                    logtext=logpath.read_text()
                    row={'elapsed_ms':elapsed,'response':response,'headers':headers,
                         'encoder_cache_hits':logtext.count('JEV_IMAGE_CACHE hit')-before,
                         'encoder_cache_misses':logtext.count('JEV_IMAGE_CACHE miss')-before_misses}
                    if mode == 'on' and row['encoder_cache_hits'] + row['encoder_cache_misses'] == 0:
                        raise RuntimeError('No encoder cache activity; refusing an invalid speed comparison')
                    if i==0: block['first_request']=row
                    else: block['runs'].append(row)
                    print(name,'first' if i==0 else i,f'{elapsed:.1f} ms',row['encoder_cache_hits'],row['encoder_cache_misses'],flush=True)
                report['blocks'].append(block)
                save()
        finally:
            for proc in reversed(procs):
                if proc.poll() is None: proc.terminate()
            for proc in reversed(procs):
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    report['summary']={}
    for mode in ('off','on'):
        values=[r['elapsed_ms'] for b in report['blocks'] if b['mode']==mode for r in b['runs']]
        report['summary'][mode]={'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'n':len(values)}
    report['speedup']=report['summary']['off']['median_ms']/report['summary']['on']['median_ms']
    # Cache has no effect on candidate scoring semantics. Compare all probabilities.
    rows=[r for b in report['blocks'] for r in [b['first_request'],*b['runs']]]
    reference=rows[0]['response']['answers']
    report['max_probability_difference']=max(abs(a['probabilities'][label]-reference[k]['probabilities'][label])
        for row in rows for k,a in row['response']['answers'].items() for label in a['probabilities'])
    report['all_choices_equal']=all(a['choice']==reference[k]['choice'] for row in rows for k,a in row['response']['answers'].items())
    save()
    print(json.dumps({k:report[k] for k in ['summary','speedup','max_probability_difference','all_choices_equal']},indent=2))


if __name__=='__main__': main()
