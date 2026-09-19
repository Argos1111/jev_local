#!/usr/bin/env python3
"""Generalization probe: tasks never used in ModernBERT training, sent through the live API.

Each task is rendered as a System One question and scored via /v1/systemone, so the
same tool measures any backend (LFM zero-shot or the fine-tuned ModernBERT). Tasks
sit at increasing distance from the training distribution:

  livedoor   news topic classification, 9 choices with descriptions (classification, unseen labels)
  jmmlu      4-choice knowledge questions on unseen subjects (MCQ, like JCommonsenseQA in shape)
  wrime-like sentiment: handled elsewhere; omitted (no pinned public split without extra deps)
  synthetic  hand-written customer-support noul/choice items (the README domain; small, indicative only)

Datasets are fetched by pinned URL and verified by SHA256. Nothing here trains anything.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.cache/datasets'
LIVEDOOR_URL = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'
LIVEDOOR_SHA = 'b17606ed8c670013a3809100a9e6104701baab62cc019abc262111bd2acf1063'
JMMLU_REV = '762cbf192c9e4588b574d718f95afd64c96fcbf4'
JMMLU_SUBJECTS = ['anatomy', 'astronomy', 'college_biology', 'high_school_geography', 'world_religions',
                  'nutrition', 'marketing', 'japanese_history', 'computer_security', 'logical_fallacies']

LIVEDOOR_LABELS = {
    'topic-news': 'トピックニュース：社会・事件・芸能などの一般ニュース',
    'sports-watch': 'スポーツ',
    'it-life-hack': 'IT・ライフハック：PCやネットサービスの活用術',
    'kaden-channel': '家電：家電製品のニュースやレビュー',
    'movie-enter': '映画・エンタメ',
    'dokujo-tsushin': '独女通信：独身女性向けのコラム',
    'smax': 'スマートフォン・モバイル端末',
    'livedoor-homme': '男性向けライフスタイル・ビジネス',
    'peachy': '女性向けの恋愛・美容・ライフスタイル',
}

# Hand-written; the README domain the model never trained on. Indicative only.
SYNTHETIC = [
    ('先週注文した商品がまだ届きません。追跡番号も無効です。', {'q': ('noul', '顧客は配送の遅延について問い合わせているか？', None), 'gold': True}),
    ('請求書の宛名を会社名に変更してもらえますか。金額はそのままで結構です。', {'q': ('noul', '顧客は返金を要求しているか？', None), 'gold': False}),
    ('アプリが起動直後にクラッシュします。iOS 18、バージョン3.2.1です。', {'q': ('choice', '担当部署は？', {'billing': '請求・返金', 'technical': '技術的な障害', 'sales': '新規契約・見積'}), 'gold': 'technical'}),
    ('御社のエンタープライズプランの見積をお願いします。50ユーザー想定です。', {'q': ('choice', '担当部署は？', {'billing': '請求・返金', 'technical': '技術的な障害', 'sales': '新規契約・見積'}), 'gold': 'sales'}),
    ('二重に引き落とされています。片方を戻してください。', {'q': ('choice', '担当部署は？', {'billing': '請求・返金', 'technical': '技術的な障害', 'sales': '新規契約・見積'}), 'gold': 'billing'}),
    ('いつも丁寧な対応ありがとうございます。今後もよろしくお願いします。', {'q': ('choice', 'メッセージの主な口調は？', {'calm': '冷静な事実確認', 'frustrated': '不満があり改善を要求', 'appreciative': '感謝や満足を伝えている'}), 'gold': 'appreciative'}),
    ('3回問い合わせても返事がない。これ以上待てないので今月で解約します。', {'q': ('noul', '顧客は解約の意思を示しているか？', None), 'gold': True}),
    ('プランの違いを教えてください。まだ契約は決めていません。', {'q': ('noul', '顧客は解約の意思を示しているか？', None), 'gold': False}),
    ('パスワードリセットのメールが届きません。', {'q': ('choice', '担当部署は？', {'billing': '請求・返金', 'technical': '技術的な障害', 'sales': '新規契約・見積'}), 'gold': 'technical'}),
    ('The invoice shows 2 charges for the same month. Please refund one.', {'q': ('noul', '顧客は返金を要求しているか？', None), 'gold': True}),
    ('Can you send me the receipt for last month? No other issues.', {'q': ('noul', '顧客は返金を要求しているか？', None), 'gold': False}),
    ('電話で説明してほしいです。メールだと分かりにくいので。', {'q': ('choice', '顧客が希望する返答方法は？', {'phone': '電話', 'email': 'メール', 'chat': 'チャット', 'unspecified': '希望の記載がない'}), 'gold': 'phone'}),
    ('返信はメールでお願いします。', {'q': ('choice', '顧客が希望する返答方法は？', {'phone': '電話', 'email': 'メール', 'chat': 'チャット', 'unspecified': '希望の記載がない'}), 'gold': 'email'}),
    ('ログインできない件、至急対応をお願いします。業務が止まっています。', {'q': ('score', '対応の緊急度は？', ['急がない', '早めに対応', '即時対応が必要']), 'gold': 2}),
    ('来月あたりにプラン変更を検討しています。急ぎではありません。', {'q': ('score', '対応の緊急度は？', ['急がない', '早めに対応', '即時対応が必要']), 'gold': 0}),
    ('契約は継続しますが、料金の内訳を一度確認したいです。', {'q': ('noul', '顧客は解約の意思を示しているか？', None), 'gold': False}),
]


def fetch(url, sha, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE/name
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'jev-local'}), timeout=120) as s:
            path.write_bytes(s.read())
    data = path.read_bytes()
    if sha and hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f'Checksum mismatch: {path}')
    return data


def livedoor_items(limit, rng):
    labels = list(LIVEDOOR_LABELS)
    items = []
    with tarfile.open(fileobj=io.BytesIO(fetch(LIVEDOOR_URL, LIVEDOOR_SHA, 'ldcc-20140209.tar.gz'))) as bundle:
        members = [m for m in bundle.getmembers() if m.isfile() and m.name.count('/') == 2 and not m.name.endswith('LICENSE.txt')]
        rng.shuffle(members)
        for m in members[:limit]:
            category = m.name.split('/')[1]
            lines = bundle.extractfile(m).read().decode('utf-8', 'replace').splitlines()
            # Line 0 is the URL, line 1 the timestamp, line 2 the title; body follows.
            title, body = lines[2].strip(), ' '.join(l.strip() for l in lines[3:] if l.strip())[:400]
            items.append({'id': m.name, 'state': {'タイトル': title, '本文冒頭': body},
                          'question': {'type': 'choice', 'instructions': 'この記事のカテゴリは？', 'criteria': dict(LIVEDOOR_LABELS)},
                          'gold': category})
    return items


def jmmlu_items(limit, rng):
    items = []
    per_subject = max(1, limit // len(JMMLU_SUBJECTS))
    for subject in JMMLU_SUBJECTS:
        data = fetch(f'https://raw.githubusercontent.com/nlp-waseda/JMMLU/{JMMLU_REV}/JMMLU/{subject}.csv', None, f'jmmlu-{subject}.csv')
        rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
        rng.shuffle(rows)
        for i, row in enumerate(rows[:per_subject]):
            q, a, b, c, d, gold = [x.strip() for x in row[:6]]
            items.append({'id': f'{subject}:{i}', 'state': {'質問': q},
                          'question': {'type': 'choice', 'instructions': '正しい答えを選択肢から1つ選んでください。',
                                       'criteria': {'A': a, 'B': b, 'C': c, 'D': d}},
                          'gold': gold, 'subject': subject})
    return items


def synthetic_items(limit, rng):
    items = []
    for i, (state, spec) in enumerate(SYNTHETIC):
        kind, instructions, criteria = spec['q']
        q = {'type': kind, 'instructions': instructions}
        if criteria is not None:
            q['criteria'] = criteria
        items.append({'id': f'synthetic:{i}', 'state': state, 'question': q, 'gold': spec['gold']})
    return items


TASKS = {'livedoor': livedoor_items, 'jmmlu': jmmlu_items, 'synthetic': synthetic_items}


def predict(answer, question):
    if question['type'] == 'noul':
        return answer['noul'] >= 0.5, answer['noul']
    if question['type'] == 'choice':
        return answer['choice'], answer['probabilities'][answer['choice']]
    probs = answer['probabilities']
    best = max(probs, key=probs.get)
    return int(best), probs[best]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', default='jev-latest')
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new --output')
    key = os.environ.get('JEV_API_KEY', 'local-dev')

    def call(payload):
        req = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone', json.dumps(payload, ensure_ascii=False).encode(),
                                     {'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(args.url.rstrip('/')+'/health', timeout=10) as r:
            health = json.load(r)
    except urllib.error.URLError as exc:
        p.exit(1, f'Cannot reach the API at {args.url} ({exc.reason}).\n'
                  'Start a backend first in another terminal: ./run.sh (LFM) or ./run_modernbert.sh (ModernBERT).\n')
    print(f'API: {args.url}  backend model: {health.get("model")}', flush=True)
    args.output.mkdir(parents=True)

    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'api_url': args.url, 'seed': args.seed, 'limit': args.limit, 'results': {}}
    for task in args.tasks:
        items = TASKS[task](args.limit, random.Random(args.seed))

        def evaluate(item):
            payload = {'model': args.model, 'state': item['state'], 'questions': {'answer': item['question']}}
            start = time.perf_counter()
            record = {'id': item['id'], 'gold': item['gold'], 'request': payload}
            try:
                response = call(payload)
                pred, prob = predict(response['answers']['answer'], item['question'])
                record.update(prediction=pred, probability=prob, correct=pred == item['gold'], response=response)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['latency_ms'] = (time.perf_counter()-start)*1000
            if 'subject' in item:
                record['subject'] = item['subject']
            return record

        with ThreadPoolExecutor(args.workers) as pool:
            records = list(pool.map(evaluate, items))
        (args.output/f'{task}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records)+'\n')
        good = [r for r in records if 'error' not in r]
        golds = Counter(str(r['gold']) for r in records)
        summary = {'n': len(records), 'errors': len(records)-len(good),
                   'accuracy': sum(r['correct'] for r in good)/len(records),
                   'majority_baseline': golds.most_common(1)[0][1]/len(records),
                   'chance': statistics.mean(1/len(r['request']['questions']['answer'].get('criteria') or [True, False]) for r in records),
                   'prediction_counts': dict(Counter(str(r['prediction']) for r in good)),
                   'gold_counts': dict(golds),
                   'p50_ms': statistics.median(r['latency_ms'] for r in good) if good else None,
                   'models': sorted({r['response']['model'] for r in good})}
        if any('subject' in r for r in good):
            summary['by_subject'] = {s: round(statistics.mean(r['correct'] for r in good if r.get('subject') == s), 3)
                                     for s in sorted({r['subject'] for r in good})}
        report['results'][task] = summary
        print(task, json.dumps(summary, ensure_ascii=False), flush=True)
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
