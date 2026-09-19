"""Public Japanese classification data rendered as System One style questions.

Every example is {'state', 'question', 'choices', 'descriptions', 'answer'} where
'answer' indexes choices. Sources are pinned by revision and SHA256 so the
training set is reproducible. Test splits used for reporting are never loaded here
except through the explicit `held_out` loader used by tools/benchmark_jglue.py's
counterpart in modernbert.evaluate.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/'.cache/jglue'

JGLUE_REVISION = '6f071c09316baae89c3d083a90985b4b1cb9968c'
JCOLA_REVISION = '736d9eef3af04bb17e4cf59adf01325973234ff9'
MORAL_REVISION = '1d232e86b0448a2686939a5e7bc1a3731637fe37'
MASSIVE_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz'

SOURCES = {
    'jnli-train': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jnli-v1.3/train-v1.3.json',
                   'c506354d1310807f7300cfa8ad49482c6359a84fe85eea8d5e667d90d338d1c4'),
    'jnli-valid': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jnli-v1.3/valid-v1.3.json',
                   'ca0353efc7c2eebfb6de4e13f16295053c8b1ee65e7b0849190c90426fbc495f'),
    'jcommonsenseqa-train': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jcommonsenseqa-v1.3/train-v1.3.json',
                             '9b55fae5ecb3aedd6f8ce5bc09196c3b629864668ec6c18eee4d65c0aa48229e'),
    'jcommonsenseqa-valid': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jcommonsenseqa-v1.3/valid-v1.3.json',
                             '0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4'),
    'jsts-train': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jsts-v1.3/train-v1.3.json',
                   'db6bd653270ac9786759920413a48973164d322998f34dd4ae32202af5e66969'),
    'jsts-valid': (f'https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jsts-v1.3/valid-v1.3.json',
                   '7c0bdcb381179f01096c635d058853d96da1e1248d23fe3f5c2beed5dc2d9b1a'),
    'jcola-train': (f'https://raw.githubusercontent.com/osekilab/JCoLA/{JCOLA_REVISION}/data/jcola-v1.0/in_domain_train-v1.0.json',
                    'c665f8a8b371b34c3c8947cde3670637cf0403ac7529ba8bb62370d07f008396'),
    'jcola-valid': (f'https://raw.githubusercontent.com/osekilab/JCoLA/{JCOLA_REVISION}/data/jcola-v1.0/in_domain_valid-v1.0.json',
                    '677fb021886b6829d6a84e8ca9f62808f6bd479fa829439f49eac4f94f23658c'),
    'moral-train': (f'https://raw.githubusercontent.com/Language-Media-Lab/commonsense-moral-ja/{MORAL_REVISION}/data/data_train.csv',
                    '46c01bdb6e2f79c2bb2c553606813bc887bda3670949a188b764ccc70b96c828'),
    'moral-valid': (f'https://raw.githubusercontent.com/Language-Media-Lab/commonsense-moral-ja/{MORAL_REVISION}/data/data_val.csv',
                    '28872666ba47ac057b3ffffb8554bdd692db39425089719f8e137d683e81e7f8'),
    'massive': (MASSIVE_URL, '4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577'),
}

NLI_LABELS = ['entailment', 'contradiction', 'neutral']
NLI_DESCRIPTIONS = ['含意：前提から仮説が正しいと必ず言える',
                    '矛盾：前提から仮説が誤りだと必ず言える',
                    '中立：前提だけでは仮説が正しいとも誤りとも判断できない']
STS_LEVELS = ['全く関係がない', 'ほとんど関係がない', '一部だけ共通する', 'おおむね同じ内容', 'ほぼ同じ内容', '完全に同じ意味']
SCENARIO_JA = {
    'alarm': 'アラーム', 'audio': '音量・音声設定', 'calendar': 'カレンダー・予定', 'cooking': '料理・レシピ',
    'datetime': '日時', 'email': 'メール', 'general': '雑談・一般', 'iot': '家電・IoT操作', 'lists': 'リスト管理',
    'music': '音楽の設定・好み', 'news': 'ニュース', 'play': '再生（音楽・ラジオ・ゲームなど）', 'qa': '質問応答・調べもの',
    'recommendation': 'おすすめ', 'social': 'SNS', 'takeaway': 'テイクアウト注文', 'transport': '交通・移動', 'weather': '天気',
}


def fetch(name, directory=DATA_DIR):
    url, digest = SOURCES[name]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(url.rsplit('/', 1)[1] if name == 'massive' else f'{name}.{url.rsplit(".", 1)[1]}')
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'jev-local-modernbert'})
        with urllib.request.urlopen(request, timeout=120) as source:
            data = source.read()
        temp = path.with_suffix(path.suffix + '.part')
        temp.write_bytes(data)
        temp.replace(path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise RuntimeError(f'Checksum mismatch for {name}: {actual}. Move {path} aside and retry.')
    return data


def jsonl(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def example(state, question, choices, answer, descriptions=None, task=None, weight=1.0):
    if not 0 <= answer < len(choices):
        raise ValueError('answer must index choices')
    return {'state': state, 'question': question, 'choices': choices,
            'descriptions': descriptions or [None]*len(choices), 'answer': answer, 'task': task, 'weight': weight}


def drop_descriptions(rows, rng, rate):
    """API callers may omit criteria descriptions; expose the model to bare labels too."""
    for row in rows:
        if rng.random() < rate:
            row['descriptions'] = [None] * len(row['choices'])
    return rows


def build_jnli(split, rng):
    rows = []
    for row in jsonl(fetch(f'jnli-{split}')):
        label = NLI_LABELS.index(row['label'])
        base = example({'前提': row['sentence1'], '仮説': row['sentence2']},
                       '前提が正しいとき、仮説との論理的な関係を判定してください。前提から分からない情報を補わないでください。',
                       list(NLI_LABELS), label, list(NLI_DESCRIPTIONS), 'jnli')
        rows.append(base)
        # Noul-style rephrasings keep the same knowledge in the true/false shape the API uses most.
        if rng.random() < 0.5:
            rows.append(example({'前提': row['sentence1'], '仮説': row['sentence2']},
                                '前提が正しいとき、仮説も必ず正しいと言えるか？', [True, False], 0 if label == 0 else 1, task='jnli-noul'))
    return drop_descriptions(rows, rng, 0.3)


def build_jcommonsenseqa(split, rng):
    rows = []
    for row in jsonl(fetch(f'jcommonsenseqa-{split}')):
        choices = [row[f'choice{i}'] for i in range(5)]
        gold = int(row['label'])
        rows.append(example({'質問': row['question']}, '質問に対して、常識に基づく最も適切な答えを選択肢から1つ選んでください。',
                            choices, gold, task='jcommonsenseqa'))
        # Yes/no phrasing of the same fact, balanced between the gold answer and a distractor.
        if rng.random() < 0.4:
            positive = rng.random() < 0.5
            named = choices[gold] if positive else rng.choice([c for i, c in enumerate(choices) if i != gold])
            rows.append(example({'質問': row['question']}, f'この質問の答えは「{named}」か？', [True, False],
                                0 if positive else 1, task='jcommonsenseqa-noul'))
    return rows


def build_jsts(split, rng):
    rows = []
    for row in jsonl(fetch(f'jsts-{split}')):
        level = min(5, max(0, int(round(float(row['label'])))))
        rows.append(example({'文1': row['sentence1'], '文2': row['sentence2']}, '2つの文の意味はどの程度近いか？',
                            list(range(6)), level, list(STS_LEVELS), 'jsts'))
    return rows


def build_jcola(split, rng):
    rows = []
    for row in jsonl(fetch(f'jcola-{split}')):
        rows.append(example(row['sentence'], 'この文は日本語として文法的に自然か？', [True, False], 0 if int(row['label']) == 1 else 1,
                            ['文法的に容認できる', '文法的に不自然で容認できない'], 'jcola'))
    return drop_descriptions(rows, rng, 0.5)


def build_moral(split, rng):
    rows = []
    for row in csv.DictReader(io.StringIO(fetch(f'moral-{split}').decode())):
        wrong = int(row['label']) == 1
        rows.append(example(row['sent'], 'この行動は道徳的に問題があるか？', [True, False], 0 if wrong else 1,
                            ['明らかに道徳的に問題がある', '道徳的に許容できる'], 'moral'))
    return drop_descriptions(rows, rng, 0.5)


def build_massive(split, rng):
    with tarfile.open(fileobj=io.BytesIO(fetch('massive'))) as bundle:
        data = bundle.extractfile('1.1/data/ja-JP.jsonl').read()
    partition = {'train': 'train', 'valid': 'dev'}[split]
    scenarios = sorted(SCENARIO_JA)
    rows = []
    for row in jsonl(data):
        if row['partition'] != partition:
            continue
        # The full 18-way question is expensive (18 pairs); most items use a random subset to vary K.
        if rng.random() < 0.2:
            rows.append(example({'発話': row['utt']}, 'この発話が属する分野は？', scenarios, scenarios.index(row['scenario']),
                                [SCENARIO_JA[s] for s in scenarios], 'massive-scenario'))
        subset = rng.sample([s for s in scenarios if s != row['scenario']], k=rng.randint(1, 5)) + [row['scenario']]
        rng.shuffle(subset)
        rows.append(example({'発話': row['utt']}, 'この発話が属する分野は？', subset, subset.index(row['scenario']),
                            [SCENARIO_JA[s] for s in subset], 'massive-scenario-subset'))
        if rng.random() < 0.4:
            positive = rng.random() < 0.5
            named = row['scenario'] if positive else rng.choice([s for s in scenarios if s != row['scenario']])
            rows.append(example({'発話': row['utt']}, f'この発話は{SCENARIO_JA[named]}に関するものか？', [True, False],
                                0 if positive else 1, task='massive-noul'))
    return drop_descriptions(rows, rng, 0.3)


BUILDERS = {'jnli': build_jnli, 'jcommonsenseqa': build_jcommonsenseqa, 'jsts': build_jsts,
            'jcola': build_jcola, 'moral': build_moral, 'massive': build_massive}


def build(split, tasks=None, seed=0):
    rng = random.Random(seed)
    rows = []
    for task in tasks or BUILDERS:
        rows.extend(BUILDERS[task](split, rng))
    rng.shuffle(rows)
    return rows
