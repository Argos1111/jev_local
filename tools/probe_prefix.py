import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from jev_local import JevLocal
from tools.evaluate import CASES
parser=argparse.ArgumentParser()
parser.add_argument('--output', default='results/prefix_probe_new.json')
args=parser.parse_args()
api=JevLocal()
prefixes=['<think>\n</think>\n\n','<think>\n</think>\n\nAnswer:','<think>\n</think>\n\nThe answer is','<think>\n</think>\n\n{"answer":"']

def run(job):
 prefix,state,truth=job
 prompt=api.post('/apply-template',{'messages':[{'role':'user','content':f'Classify whether the customer explicitly requests a refund.\nCustomer: {state}\nA: yes\nB: no\nRespond with one letter only.'}]})['prompt']+prefix
 r=api.post('/completion',{'prompt':prompt,'n_predict':1,'n_probs':100,'temperature':-1,'cache_prompt':False})
 entries=r['completion_probabilities'][0]['top_logprobs']
 return {'prefix':prefix,'state':state,'truth':truth,'top':entries[:5]}
jobs=[(prefix,state,truth) for prefix in prefixes for state,truth in [CASES[0],CASES[5]]]
with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(run,jobs))
open(args.output,'w').write(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for r in rows: print(repr(r['prefix']),r['truth'],[(p['token'],round(p['logprob'],2)) for p in r['top']])
