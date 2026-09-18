#!/usr/bin/env python3
"""Call the local TypeSafe-compatible API with JSON; optionally render a table."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
from display import table, percent, value, duration

def render_response(response, request, elapsed_ms):
    rows=[]
    for key,a in response['answers'].items():
        q=request['questions'][key]
        label=q.get('instructions')
        question=key if label is None else value(label)
        if a['type']=='noul':
            answer=f"{a['noul']:.4f}"
            probability=percent(a['noul'])+' (true)'
        elif a['type']=='choice':
            answer=a['choice']
            probability=percent(a['probabilities'][answer])
        else:
            answer=f"{a['score']:.4f}"
            probability='—（期待値）'
        rows.append([question,a['type'],answer,probability,percent(a['confidence']) if 'confidence' in a else '—（仕様なし）'])
    return '\n'.join(['Jev Local / System One',f"モデル: {response['model']}  ·  {duration(elapsed_ms)}",'',
                     '入力（state）',value(request['state']),'',
                     table(['質問','返答の型','返答','確率','確信度'],rows),'',
                     'Scoreは段階の期待値、NoulはP(true)。確信度はローカル方式の分布集中度です。'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8080')
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'examples/systemone.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--format',choices=['pretty','json'],default='pretty')
    args=p.parse_args()
    payload=json.loads(args.input.read_text())
    req=urllib.request.Request(args.url.rstrip('/')+'/v1/systemone',json.dumps(payload,ensure_ascii=False).encode(),
                               {'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('JEV_API_KEY','local-dev')})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as response: result=json.load(response)
    elapsed=(time.perf_counter()-start)*1000
    raw=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw+'\n')
    print(raw if args.format=='json' else render_response(result,payload,elapsed))

if __name__=='__main__':main()
