#!/usr/bin/env python3
"""Small hand-written sanity set, not a calibration benchmark."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
from pathlib import Path
from jev_local import JevLocal
from display import print_result

CASES = [
    ('二重請求されています。返金してください。', True),
    ('返金を希望します。', True),
    ('Please refund the duplicate charge.', True),
    ('I want my money back.', True),
    ('ログインできません。パスワードをリセットしたいです。', False),
    ('返金は不要です。請求書の宛名だけ変更してください。', False),
    ('Please send a receipt. I do not want a refund.', False),
    ('Everything works perfectly, thank you!', False),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['pretty','json'], default='pretty')
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--output', type=Path, default=Path('results/sanity.json'))
    parser.add_argument('--json-baseline', action='store_true')
    parser.add_argument('--semantic-tokens', action='store_true')
    parser.add_argument('--empty-think', action='store_true')
    args = parser.parse_args()
    api = JevLocal(args.url, empty_think=args.empty_think)
    spec = {'question': '顧客は返金を要求しているか？', 'choices': [True,False]}
    if args.semantic_tokens: spec['tokens'] = ['true','false']
    api.prepare({'state':'', 'questions':{'refund':spec}})
    jobs = [(state, expected, reverse) for state,expected in CASES for reverse in (False,True)]
    def run(job):
        state,expected,reverse = job
        question = dict(spec, choices=[False,True] if reverse else [True,False])
        if args.semantic_tokens: question['tokens'] = ['false','true'] if reverse else ['true','false']
        result = api.score(state,question)
        ptrue = next(x['probability'] for x in result['probabilities'] if x['value'] is True)
        return {'state':state,'expected':expected,'reversed':reverse,'p_true':ptrue,'result':result}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run,jobs))
    canonical = rows[::2]
    accuracy = sum(x['result']['value']==x['expected'] for x in canonical)/len(canonical)
    brier = sum((x['p_true']-int(x['expected']))**2 for x in canonical)/len(canonical)
    nll = -sum(math.log(max(1e-15,x['p_true'] if x['expected'] else 1-x['p_true'])) for x in canonical)/len(canonical)
    deltas = [abs(a['p_true']-b['p_true']) for a,b in zip(rows[::2],rows[1::2])]
    flips = sum(a['result']['value']!=b['result']['value'] for a,b in zip(rows[::2],rows[1::2]))
    out = {'scope':'8 hand-written sanity cases; not held-out calibration evidence',
           'semantic_tokens':args.semantic_tokens,'empty_think':args.empty_think,'accuracy':accuracy,'binary_brier':brier,'nll':nll,
           'reversed_choice_mean_abs_probability_delta':sum(deltas)/len(deltas),
           'reversed_choice_max_abs_probability_delta':max(deltas),
           'reversed_choice_prediction_flips':flips,'rows':rows}
    if args.json_baseline:
        baseline = []
        for state, expected in CASES:
            result = api.generate_json({'state':state,'questions':{'refund':spec}})
            baseline.append({'state':state,'expected':expected,'result':result})
        out['json_baseline'] = baseline
        out['json_accuracy'] = sum(x['result']['values']['refund'] == x['expected'] for x in baseline)/len(baseline)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print_result(out, args.format)
    assert all(abs(sum(p['probability'] for p in r['result']['probabilities'])-1)<1e-9 for r in rows)
    assert all(0<=r['result']['candidate_mass']<=1.00001 for r in rows)

if __name__=='__main__': main()
