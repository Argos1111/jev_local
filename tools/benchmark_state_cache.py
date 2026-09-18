#!/usr/bin/env python3
"""Measure fresh per-request shared prefixes, including save/restore and HTTP costs."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time
from systemone import SystemOne


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend-url',default='http://127.0.0.1:8097')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('results/state_cache_benchmark.json'))
    args=p.parse_args()
    original=json.loads(Path('examples/systemone.json').read_text())
    cases={}
    for name,count in [('short',0),('medium',15),('long',45)]:
        payload=copy.deepcopy(original)
        if count:payload['state']['history']=[f'記録{i}: 顧客は有料プランを利用中。サービスは現在利用できる。' for i in range(count)]
        cases[name]=payload
    results={}
    for name,payload in cases.items():
        runs=[]
        for i in range(args.rounds):
            run={}
            for mode in (['off','shared','auto'] if i%2==0 else ['auto','shared','off']):
                engine=SystemOne(args.backend_url,state_cache=mode)
                diagnostics={}
                try:
                    start=time.perf_counter();response=engine.evaluate(payload,diagnostics=diagnostics)
                    run[mode]={'elapsed_ms':(time.perf_counter()-start)*1000,'diagnostics':diagnostics,'response':response}
                finally:engine.close()
            runs.append(run)
        med={mode:statistics.median(r[mode]['elapsed_ms'] for r in runs) for mode in ('off','shared','auto')}
        results[name]={'median_ms':med,'speedup':med['off']/med['shared'],'runs':runs}
        print(name,med,'speedup',results[name]['speedup'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
