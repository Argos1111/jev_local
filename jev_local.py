#!/usr/bin/env python3
"""Typed decisions from first-answer-token logprobs; Python standard library only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys
from display import print_result
import time
import urllib.request

THINK_PREFILL = '<think>\n</think>\n\n'
BASELINE_DEMO = {
    'state': '顧客は有料プランを利用中。請求が二重になったと怒っており、返金を要求している。解決しなければ解約すると明言している。サービス自体は現在使えている。',
    'questions': {
        'refund_requested': {'question': '返金を要求しているか？', 'choices': [True, False]},
        'sentiment': {'question': '顧客の感情は？', 'choices': ['positive', 'neutral', 'negative']},
        'route': {'question': '担当部署は？', 'choices': ['billing', 'technical', 'sales']},
        'churn_risk': {'question': '解約リスクは？', 'choices': ['low', 'medium', 'high']},
    },
}

# Keep the original four-question payload for comparable speed measurements.
DEMO = {
    'state': BASELINE_DEMO['state'],
    'questions': {
        **BASELINE_DEMO['questions'],
        'retention_after_refund': {
            'question': '返金が完了すれば、この顧客は利用を継続するか？',
            'choices': [True, False],
        },
        'reply_channel': {
            'question': '顧客が希望する返答方法は？',
            'choices': ['電話', 'メール', 'チャット'],
        },
        'refund_timing': {
            'question': '顧客が許容できる返金までの待ち時間は？',
            'choices': ['当日中', '3営業日以内', '1週間以内'],
        },
    },
}


def distribution_confidence(probabilities):
    """Concentration over the supplied candidates, not calibrated correctness."""
    if len(probabilities) < 2 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError('Expected at least two finite candidate probabilities')
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError('Candidate probabilities must sum to one')
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    ranked = sorted(probabilities, reverse=True)
    return {
        'confidence': min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities)))),
        'confidence_method': 'one_minus_normalized_entropy',
        'entropy_nats': entropy,
        'top_two_margin': ranked[0] - ranked[1],
    }


class JevLocal:
    def __init__(self, url='http://127.0.0.1:8097', workers=4, empty_think=False):
        self.url, self.workers = url.rstrip('/'), workers
        self.ids = {}
        self.empty_think = empty_think

    def post(self, path, data):
        req = urllib.request.Request(self.url + path, json.dumps(data, ensure_ascii=False).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)

    def prepare(self, payload):
        if not isinstance(payload['state'], str) or not payload['questions']:
            raise ValueError('state must be a string and questions must be nonempty')
        for spec in payload['questions'].values():
            choices = spec['choices']
            if not 2 <= len(choices) <= 26:
                raise ValueError('Each question requires 2..26 choices')
            if len({json.dumps(c, sort_keys=True) for c in choices}) != len(choices):
                raise ValueError('Choices must be unique JSON values')
            for letter in self.labels(spec):
                if letter not in self.ids:
                    tokens = self.post('/tokenize', {'content': letter, 'add_special': False})['tokens']
                    if len(tokens) != 1:
                        raise ValueError(f'{letter!r} is not a single token: {tokens}')
                    self.ids[letter] = tokens[0]

    @staticmethod
    def letters(choices):
        return [chr(65+i) for i in range(len(choices))]

    def labels(self, spec):
        labels = spec.get('tokens', self.letters(spec['choices']))
        if len(labels) != len(spec['choices']) or len(set(labels)) != len(labels):
            raise ValueError('tokens must contain one unique string per choice')
        if not all(isinstance(x, str) and x for x in labels):
            raise ValueError('tokens must be nonempty strings')
        return labels

    def prompt(self, text):
        result = self.post('/apply-template', {'messages': [{'role': 'user', 'content': text}]})['prompt']
        # Instruct uses the native answer position; legacy 8B experiment is opt-in.
        return result + (THINK_PREFILL if self.empty_think else "")

    def score_prompt(self, state, spec):
        letters = self.labels(spec)
        descriptions = spec.get('descriptions', [None] * len(letters))
        options = '\n'.join(f'{a}: {json.dumps(v, ensure_ascii=False)}' +
                            ('' if desc is None else ' — ' + json.dumps(desc, ensure_ascii=False))
                            for a, v, desc in zip(letters, spec['choices'], descriptions))
        instruction = 'Reply with exactly one option token, nothing else.' if 'tokens' in spec else 'Reply with exactly one option letter, nothing else.'
        return self.prompt(f'State:\n{state}\n\nQuestion: {spec["question"]}\n{options}\n{instruction}')

    def score(self, state, spec, cache=False, *, prompt_tokens=None, slot=None, before_attempt=None):
        letters = self.labels(spec)
        prompt = self.score_prompt(state, spec) if prompt_tokens is None else prompt_tokens
        started = time.perf_counter()
        # Never silently treat a candidate missing from top-N as probability zero.
        usage = {'input_tokens': 0, 'output_tokens': 0}
        cache_attempts = []
        for n in (100, 1000, 200000):
            if before_attempt is not None: before_attempt()
            raw = self.post('/completion', {
                'prompt': prompt, 'n_predict': 1, 'n_probs': n,
                'temperature': -1, 'top_k': 0, 'top_p': 1, 'min_p': 0,
                'repeat_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
                'post_sampling_probs': False, 'cache_prompt': cache,
                'seed': 42,
                **({'id_slot': slot} if slot is not None else {}),
            })
            if raw.get('truncated'):
                raise RuntimeError('Prompt exceeded backend context; refusing truncated inference')
            cache_attempts.append(raw.get('timings', {}))
            usage['input_tokens'] += raw.get('tokens_evaluated', 0)
            usage['output_tokens'] += raw.get('tokens_predicted', 0)
            row = raw['completion_probabilities'][0]
            probs = {entry['id']: entry['logprob'] for entry in row['top_logprobs']}
            if all(self.ids[a] in probs for a in letters):
                break
        else:
            raise RuntimeError('Server did not return all candidate logprobs; refusing approximate scores')
        lp = [probs[self.ids[a]] for a in letters]
        if not all(math.isfinite(x) for x in lp):
            raise RuntimeError('Non-finite candidate logprob')
        shift = max(lp)
        weights = [math.exp(x-shift) for x in lp]
        conditional = [x/sum(weights) for x in weights]
        best = max(range(len(lp)), key=lp.__getitem__)
        return {
            'cache_attempts': cache_attempts,
            'usage': usage,
            'question': spec['question'],
            **distribution_confidence(conditional),
            'value': spec['choices'][best],
            'probabilities': [{'value': v, 'probability': p, 'token': a, 'token_id': self.ids[a], 'logprob': l}
                              for v,p,a,l in zip(spec['choices'], conditional, letters, lp)],
            'candidate_mass': sum(math.exp(x) for x in lp),
            'unconstrained_first_token': row['token'],
            'elapsed_ms': (time.perf_counter()-started)*1000,
            'timings': raw['timings'], 'n_probs_requested': n,
        }

    def decide(self, payload, cache=False, workers=None):
        self.prepare(payload)
        started = time.perf_counter()
        items = list(payload['questions'].items())
        with ThreadPoolExecutor(max_workers=workers or self.workers) as pool:
            results = list(pool.map(lambda item: self.score(payload['state'], item[1], cache), items))
        return {'state': payload['state'],
                'decisions': dict(zip((k for k,_ in items), results)),
                'values': {k:r['value'] for (k,_),r in zip(items, results)},
                'elapsed_ms': (time.perf_counter()-started)*1000,
                'cache_prompt': cache, 'workers': workers or self.workers,
                'probability_semantics': 'softmax over option-token logits; not calibrated confidence'}

    def generate_json(self, payload, cache=False):
        schema = {'type': 'object', 'properties': {k: {'enum': s['choices']} for k,s in payload['questions'].items()},
                  'required': list(payload['questions']), 'additionalProperties': False}
        prompt = self.prompt('Return a JSON object with one answer per question.\n' + json.dumps(payload, ensure_ascii=False))
        started = time.perf_counter()
        raw = self.post('/completion', {'prompt': prompt, 'n_predict': 256, 'temperature': -1,
                                        'cache_prompt': cache, 'json_schema': schema, 'seed': 42})
        elapsed = (time.perf_counter()-started)*1000
        if raw['stop_type'] == 'limit':
            raise RuntimeError('JSON baseline exhausted its token budget')
        values = json.loads(raw['content'])
        if set(values) != set(payload['questions']):
            raise RuntimeError('JSON baseline returned wrong keys')
        for k,v in values.items():
            if not any(type(v) is type(c) and v == c for c in payload['questions'][k]['choices']):
                raise RuntimeError('JSON baseline returned value outside schema')
        return {'values': values, 'elapsed_ms': elapsed, 'timings': raw['timings'],
                'tokens_predicted': raw['tokens_predicted'], 'content': raw['content']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo','decide','benchmark','view'])
    parser.add_argument('--format', choices=['pretty','json'], default='pretty', help='Console format; --output always saves JSON')
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--empty-think', action='store_true', help='Legacy reasoning-model experiment only')
    args = parser.parse_args()
    if args.command == 'view':
        if not args.input: parser.error('view requires --input RESULT.json')
        print_result(json.loads(args.input.read_text()), args.format)
        return
    api = JevLocal(args.url, args.workers, args.empty_think)
    payload = json.loads(args.input.read_text()) if args.input else (BASELINE_DEMO if args.command == 'benchmark' else DEMO)
    if args.command == 'benchmark':
        if args.rounds < 1: parser.error('--rounds must be >= 1')
        api.prepare(payload)
        runs = []
        # Alternate execution order. Model resident; prompt cache disabled by default.
        # Both methods use the same prefill policy. Baseline emits VALUES ONLY, no verbal probabilities.
        for i in range(args.rounds):
            run = {}
            for mode in (['logprobs','json'] if i%2 == 0 else ['json','logprobs']):
                run[mode] = api.decide(payload, args.cache) if mode == 'logprobs' else api.generate_json(payload, args.cache)
                print(f'round {i+1} {mode}: {run[mode]["elapsed_ms"]:.1f} ms', flush=True, file=sys.stderr)
            run['agreement'] = run['logprobs']['values'] == run['json']['values']
            runs.append(run)
        medians = {k: statistics.median(r[k]['elapsed_ms'] for r in runs) for k in ('logprobs','json')}
        result = {'runs': runs, 'median_ms': medians, 'speedup': medians['json']/medians['logprobs'],
                  'cache_prompt': args.cache, 'workers': args.workers,
                  'scope': 'Local synthetic demo, not Jev evaluation. JSON baseline returns values only.'}
    else:
        result = api.decide(payload, args.cache)
    result['empty_think'] = args.empty_think
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered+'\n')
    print_result(result, args.format)

if __name__ == '__main__':
    main()
