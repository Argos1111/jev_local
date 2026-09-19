#!/usr/bin/env python3
"""Zero-shot JGLUE evaluation through the TypeSafe-compatible HTTP API."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from display import width  # noqa: E402

REVISION = '6f071c09316baae89c3d083a90985b4b1cb9968c'
ROOT = Path(__file__).resolve().parents[1]
LABELS = ['entailment', 'contradiction', 'neutral']

def payload_for(task, row, model):
    if task == 'jnli':
        state = {'前提': row['sentence1'], '仮説': row['sentence2']}
        instructions = '前提が正しいとき、仮説との論理的な関係を判定してください。前提から分からない情報を補わないでください。'
        criteria = dict(zip(LABELS, ['含意：前提から仮説が正しいと必ず言える',
                                    '矛盾：前提から仮説が誤りだと必ず言える',
                                    '中立：前提だけでは仮説が正しいとも誤りとも判断できない']))
    else:
        state = {'質問': row['question']}
        instructions = '質問に対して、常識に基づく最も適切な答えを選択肢から1つ選んでください。'
        criteria = {str(i): row[f'choice{i}'] for i in range(5)}
    return {'model': model, 'state': state, 'questions': {'answer': {
        'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}

def load_data(task, split, directory):
    directory.mkdir(parents=True, exist_ok=True)
    name = f'{task}-{split}-{REVISION}.jsonl'
    path = directory / name
    url = f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{REVISION}/datasets/{task}-v1.3/{split}-v1.3.json'
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=60).read()
        temp = path.with_suffix('.tmp'); temp.write_bytes(data); temp.replace(path)
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    for row in rows:
        gold = str(row['label'])
        assert gold in (LABELS if task == 'jnli' else list('01234')), f'Invalid gold: {gold}'
    return rows, {'url': url, 'revision': REVISION, 'sha256': hashlib.sha256(data).hexdigest(), 'rows': len(rows)}

def clip(text, columns):
    """Truncate to a display width, counting East Asian characters as two columns."""
    text = text.replace('\n', ' ')
    if width(text) <= columns:
        return text + ' ' * (columns - width(text))
    out = ''
    for ch in text:
        if width(out + ch) > columns - 1:
            break
        out += ch
    return out + '…' + ' ' * (columns - width(out) - 1)


def describe(task, row, record):
    """One console line per item; gold and prediction shown as short labels."""
    if task == 'jnli':
        short = {'entailment': '含意', 'contradiction': '矛盾', 'neutral': '中立'}
        text = f"{row['sentence1']} ⇒ {row['sentence2']}"
    else:
        short = {str(i): row[f'choice{i}'] for i in range(5)}
        text = row['question']
    gold = short.get(record['gold'], record['gold'])
    if 'error' in record:
        return f"  !  {record['index']+1:>5}  {'エラー':>7}  {clip(record['error'], 44)}  {clip(text, 60)}"
    pred = short.get(record['prediction'], record['prediction'])
    mark = '✓' if record['correct'] else '✗'
    verdict = clip(pred, 18) if record['correct'] else clip(f'{pred} (正解: {gold})', 40)
    return f"  {mark}  {record['index']+1:>5}  {record['probabilities'][record['prediction']]*100:6.1f}%  {verdict}  {clip(text, 60)}"


def summarize(records, elapsed):
    good = [r for r in records if 'error' not in r]
    n = len(records); correct = sum(r['correct'] for r in good)
    lat = sorted(r['latency_ms'] for r in good)
    confusion = {}
    for r in good:
        confusion.setdefault(r['gold'], Counter())[r['prediction']] += 1
    return {'total': n, 'successful': len(good), 'errors': n-len(good), 'correct': correct,
            'accuracy': correct/n if n else None,
            'accuracy_successful_only': correct/len(good) if good else None,
            'wall_seconds': elapsed, 'examples_per_second': n/elapsed,
            'latency_ms': {'mean': statistics.mean(lat), 'p50': statistics.median(lat),
                           'p95': lat[math.ceil(.95*len(lat))-1]} if lat else {},
            'mean_confidence': statistics.mean(r['confidence'] for r in good) if good else None,
            'mean_gold_probability': statistics.mean(r['probabilities'][r['gold']] for r in good) if good else None,
            'negative_log_likelihood': statistics.mean(-math.log(max(r['probabilities'][r['gold']],1e-300)) for r in good) if good else None,
            'gold_counts': dict(Counter(r['gold'] for r in records)),
            'prediction_counts': dict(Counter(r['prediction'] for r in good)),
            'confusion_matrix_gold_rows': confusion,
            'cache_modes': dict(Counter(r['headers'].get('X-Jev-Local-State-Cache','unknown') for r in good)),
            'models': sorted(set(r['response']['model'] for r in good))}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', default='jev-latest')
    p.add_argument('--tasks', nargs='+', choices=['jnli','jcommonsenseqa'], default=['jnli','jcommonsenseqa'])
    p.add_argument('--split', choices=['valid','test'], default='test')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='0 means all rows; otherwise first N, for smoke tests only')
    p.add_argument('--data-dir', type=Path, default=ROOT/'.cache/jglue')
    p.add_argument('--output', type=Path, default=ROOT/'results/jglue-test')
    p.add_argument('--method', default='zero-shot', help='Method label for the report, e.g. fine-tuned-on-train for the ModernBERT backend')
    args = p.parse_args()
    if args.workers < 1 or args.limit < 0: p.error('workers must be positive; limit must be non-negative')
    if args.output.exists(): p.error('Output already exists; choose a new --output directory')
    datasets = {t: load_data(t,args.split,args.data_dir) for t in args.tasks}
    key = os.environ.get('JEV_API_KEY','local-dev')
    def call(payload):
        req = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone', data=json.dumps(payload,ensure_ascii=False).encode(),
                                     headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=120) as response:
            return json.load(response), {k:v for k,v in response.headers.items() if k.lower().startswith('x-jev-local-')}
    # Synthetic warm-up; excluded from scores and timing. Never use gold labels as input.
    try:
        response,_ = call({'model':args.model,'state':'空は青い。','questions':{'answer':{'type':'choice','instructions':'空の色は？','criteria':{'blue':'青','red':'赤'}}}})
    except urllib.error.URLError as exc:
        p.exit(1, f'Cannot reach the API at {args.url} ({exc.reason}).\n'
                  'Start a backend first in another terminal: ./run.sh (LFM) or ./run_modernbert.sh (ModernBERT), '
                  'then rerun. Use --url if it listens on another port.\n')
    except urllib.error.HTTPError as exc:
        p.exit(1, f'API at {args.url} rejected the warm-up request: HTTP {exc.code} {exc.read().decode(errors="replace")[:300]}\n')
    print(f'API: {args.url}  backend model: {response["model"]}', flush=True)
    args.output.mkdir(parents=True)
    report = {'started_at':datetime.now(timezone.utc).isoformat(), 'api_url':args.url,
              'requested_model':args.model,'split':args.split,'workers':args.workers,'limit':args.limit,
              'method':f'{args.method}; fixed original choice order; API classification; no prompt tuning on test; one item per request',
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'implementation_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['jev_local.py','systemone.py','state_cache.py','api_server.py']},
              'datasets':{},'results':{}}
    color = sys.stdout.isatty() and 'NO_COLOR' not in os.environ
    def paint(line, ok):
        if not color: return line
        return f'\033[32m{line}\033[0m' if ok else f'\033[31m{line}\033[0m'
    for task,(rows,source) in datasets.items():
        report['datasets'][task]=source
        if args.limit: rows=rows[:args.limit]
        print(f"\n{task}  {len(rows)}件  split={args.split}  並列={args.workers}\n" + '─'*100, flush=True)
        def evaluate(pair):
            index,row=pair
            payload=payload_for(task,row,args.model)
            record={'index':index,'id':row.get('sentence_pair_id',row.get('q_id')), 'gold':str(row['label']), 'request':payload}
            start=time.perf_counter()
            try:
                response,headers=call(payload)
                answer=response['answers']['answer']; probs=answer['probabilities']
                assert set(probs)==set(payload['questions']['answer']['criteria'])
                assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5
                pred=answer['choice']; assert pred in probs and probs[pred]>=max(probs.values())-1e-8
                record.update(prediction=pred,correct=pred==record['gold'],probabilities=probs,confidence=answer['confidence'],response=response,headers=headers)
            except Exception as exc: record['error']=f'{type(exc).__name__}: {exc}'
            record['latency_ms']=(time.perf_counter()-start)*1000
            return record
        records=[]; start=time.perf_counter()
        with (args.output/f'{task}.jsonl').open('w') as output, ThreadPoolExecutor(args.workers) as pool:
            correct=0
            for record in pool.map(evaluate,enumerate(rows)):
                records.append(record); output.write(json.dumps(record,ensure_ascii=False)+'\n'); output.flush()
                correct+=record.get('correct',False)
                line=describe(task,rows[record['index']],record) + f"  累計 {correct/len(records):6.2%}"
                print(paint(line, record.get('correct',False)), flush=True)
        report['results'][task]=summarize(records,time.perf_counter()-start)
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        r=report['results'][task]; l=r['latency_ms']
        print('─'*100 + f"\n{task}: 正解 {r['correct']}/{r['total']} = {r['accuracy']:.2%}  エラー {r['errors']}  "
              f"p50 {l.get('p50',0):.1f} ms  p95 {l.get('p95',0):.1f} ms  {r['examples_per_second']:.1f} 件/秒  所要 {r['wall_seconds']:.1f} 秒", flush=True)
    print(f"\n保存先: {args.output}  (report.md / summary.json / *.jsonl)", flush=True)
    lines=['# JGLUE API benchmark', '', f"Split: {args.split}; {args.method}; concurrency: {args.workers}", '',
           '| Task | Correct / Total | Accuracy | Errors | p50 / p95 ms | Examples/s |', '|---|---:|---:|---:|---:|---:|']
    for task,r in report['results'].items():
        l=r['latency_ms']; lines.append(f"| {task} | {r['correct']} / {r['total']} | {r['accuracy']:.2%} | {r['errors']} | {l.get('p50',0):.1f} / {l.get('p95',0):.1f} | {r['examples_per_second']:.2f} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    if any(r['errors'] for r in report['results'].values()): raise SystemExit('Evaluation completed with errors; see JSONL')

if __name__=='__main__': main()
