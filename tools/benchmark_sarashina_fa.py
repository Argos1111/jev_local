#!/usr/bin/env python3
"""Compare FA off/on on one build; no prompt cache, one answer token, fixed slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess

from jev_local import JevLocal, BASELINE_DEMO
from scripts.run_local import wait_ready
from scripts.build_sarashina import build_environment
from scripts.setup_runtime import ROOT, sha256


def default_cases():
    return {name: {'state': BASELINE_DEMO['state'] * repeats,
                   'spec': BASELINE_DEMO['questions']['route']}
            for name, repeats in [('short', 1), ('medium', 5), ('long', 20)]}


def dataset_cases():
    """Fixed JComQA/livedoor inputs used in the stock-runtime FA investigation."""
    import io
    import tarfile
    from systemone import content, to_spec
    from tools.benchmark_jglue import load_data, payload_for
    from tools.evaluate_heldout_tasks import fetch, LIVEDOOR_URL, LIVEDOOR_SHA, LIVEDOOR_LABELS
    rows, _ = load_data('jcommonsenseqa', 'test', ROOT/'.cache/jglue')
    qa = next(row for row in rows if row['q_id'] == 10063)
    data = fetch(LIVEDOOR_URL, LIVEDOOR_SHA, 'ldcc-20140209.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        lines = bundle.extractfile('text/livedoor-homme/livedoor-homme-6011299.txt').read().decode('utf-8', 'replace').splitlines()
    state = {'タイトル': lines[2].strip(), '本文冒頭': ' '.join(line.strip() for line in lines[3:] if line.strip())[:400]}
    question = {'type': 'choice', 'instructions': 'この記事のカテゴリは？', 'criteria': dict(LIVEDOOR_LABELS)}
    requests = {'jcommonsenseqa': payload_for('jcommonsenseqa', qa, 'jev-latest'),
                'livedoor': {'model': 'jev-latest', 'state': state, 'questions': {'answer': question}}}
    return {name: {'state': content(request['state']), 'spec': to_spec(request['questions']['answer']),
                   'request': request} for name, request in requests.items()}


def compare(rows):
    reference = rows[0]
    return {
        'all_choices_equal': all(r['value'] == reference['value'] for r in rows),
        'max_probability_difference': max(abs(p['probability'] - q['probability'])
            for r in rows for p, q in zip(r['probabilities'], reference['probabilities'])),
    }


def runtime_observations(log):
    """Keep evidence of FA selection separate from a speedup claim."""
    return {
        'flash_attn_states': sorted(set(re.findall(r'flash_attn\s*=\s*(\w+)', log))),
        'graph_splits': sorted({int(n) for n in re.findall(r'graph splits\s*=\s*(\d+)', log)}),
        'device_lines': [line for line in log.splitlines() if re.search(r'Device \d+:|using device |compute buffer size', line)],
        'attention_lines': [line for line in log.splitlines() if 'FLASH_ATTN_EXT' in line or 'flash_attn' in line],
        'note': 'FA enabled alone does not prove GPU placement. Inspect backend logs / graph splits / kernel checks.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--cases', type=Path, help='JSON mapping (or {"items": mapping}) of state/spec inputs')
    inputs.add_argument('--dataset-cases', action='store_true', help='Fetch fixed JComQA/livedoor cases from pinned datasets')
    parser.add_argument('--output', type=Path, default=ROOT/'results/sarashina-fa160')
    parser.add_argument('--port', type=int, default=29197)
    parser.add_argument('--device', default=os.environ.get('GPU_DEVICE'), help='e.g. CUDA0 or ROCm1; omitted: llama.cpp chooses available GPUs')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--context', type=int, default=8192)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmups < 1 or not 1 <= args.port <= 65535:
        parser.error('Positive rounds/warmups and a valid port are required')
    if not args.server.is_file() or not args.model.is_file():
        parser.error('--server and --model must exist')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', args.port))
        except OSError:
            parser.error('Port is busy')
    cases = json.loads(args.cases.read_text()) if args.cases else (dataset_cases() if args.dataset_cases else default_cases())
    cases = cases.get('items', cases)
    args.output.mkdir(parents=True, exist_ok=True)
    build = args.server.resolve().parent.parent
    manifest_path = build/'jev-build.json'
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
              'build_manifest': json.loads(manifest_path.read_text()) if manifest_path.is_file() else None,
              'server': str(args.server.resolve()), 'model': str(args.model.resolve()),
              'server_sha256': sha256(args.server), 'model_sha256': sha256(args.model),
              'libraries_sha256': {p.name: sha256(p) for p in sorted(args.server.parent.glob('*.so'))},
              'device': args.device, 'context': args.context, 'slots': 4, 'threads': 8,
              'rounds': args.rounds, 'warmups': args.warmups, 'cases': cases,
              'order': ['off', 'on', 'on', 'off'], 'blocks': [],
              'scope': 'Pretokenized /completion round trip; 1 token, no prompt/encoder cache or mmproj. Not full SystemOne HTTP.'}
    url = f'http://127.0.0.1:{args.port}'
    def save():
        (args.output/'measurement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    save()
    for index, mode in enumerate(report['order']):
        env = build_environment(build, args.server.resolve().parent)
        env['JEV_IMAGE_CACHE_MIB'] = '0'
        env.pop('LLAMA_ARG_MMPROJ', None)
        command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1',
                   '--port', str(args.port), '-ngl', 'all', '-np', '4',
                   '-c', str(args.context), '-t', '8', '-fa', mode, '-ctk', 'f16', '-ctv', 'f16',
                   '--cache-ram', '0', '-lv', '4']
        if args.device:
            command += ['--device', args.device]
        block = {'mode': mode, 'args': command, 'tasks': {}}
        logpath = args.output/f'{index+1}-{mode}.log'
        with logpath.open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                wait_ready(url, process)
                client = JevLocal(url)
                for name, case in cases.items():
                    state, spec = case['state'], case['spec']
                    client.prepare({'state': state, 'questions': {'answer': spec}})
                    prompt = client.post('/tokenize', {'content': client.score_prompt(state, spec),
                                                      'add_special': False, 'parse_special': True})['tokens']
                    if len(prompt) + 1 > args.context // 4:
                        raise ValueError(f'{name}: prompt exceeds per-slot context')
                    task = {'tokens': len(prompt), 'concurrency': {}}
                    for concurrency in (1, 4):
                        rows = []
                        for round_index in range(args.warmups + args.rounds):
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                batch = list(pool.map(lambda slot: client.score(state, spec, cache=False,
                                    prompt_tokens=prompt, slot=slot), range(concurrency)))
                            if any(r['timings'].get('cache_n') != 0 for r in batch):
                                raise AssertionError('Unexpected prompt cache reuse')
                            if round_index >= args.warmups:
                                rows.extend(batch)
                        summary = {'p50_ms': statistics.median(r['elapsed_ms'] for r in rows),
                                   'prompt_p50_ms': statistics.median(r['timings']['prompt_ms'] for r in rows),
                                   **compare(rows)}
                        task['concurrency'][str(concurrency)] = {'summary': summary, 'runs': rows}
                        print(index+1, mode, name, concurrency, summary, flush=True)
                    block['tasks'][name] = task
                # Exercise repeated single-token decode and cached KV positions too.
                block['generation'] = []
                for question in ('日本の首都を一語だけで答えてください。', '1から10までを順番に列挙してください。'):
                    result = client.post('/completion', {'prompt': client.prompt(question), 'n_predict': 80,
                        'temperature': 0, 'cache_prompt': False, 'id_slot': 0})
                    block['generation'].append({'question': question, 'content': result['content'],
                                                'timings': result['timings']})
                block['runtime_observations'] = runtime_observations(logpath.read_text(errors='replace'))
                report['blocks'].append(block)
                save()
            except Exception as exc:
                block['error'] = f'{type(exc).__name__}: {exc}'
                block['runtime_observations'] = runtime_observations(logpath.read_text(errors='replace'))
                report['blocks'].append(block)
                save()
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    report['summary'] = {}
    for name in cases:
        report['summary'][name] = {}
        for concurrency in ('1', '4'):
            by_mode = {mode: [r for b in report['blocks'] if b['mode'] == mode
                             for r in b['tasks'][name]['concurrency'][concurrency]['runs']]
                       for mode in ('off', 'on')}
            medians = {mode: statistics.median(r['elapsed_ms'] for r in rows) for mode, rows in by_mode.items()}
            report['summary'][name][concurrency] = {'p50_ms': medians, 'speedup': medians['off']/medians['on'],
                **compare(by_mode['off'] + by_mode['on'])}
    save()
    print(json.dumps(report['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
