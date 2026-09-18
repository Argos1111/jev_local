#!/usr/bin/env python3
"""Live contract smoke test; run using .venv/bin/python for official SDK checks."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--sdk', action='store_true', help='Also test the optional official SDK')
    args=parser.parse_args()
    api_key=os.environ.get('JEV_API_KEY','local-dev')
    payload=json.loads((Path(__file__).resolve().parents[1]/'examples/systemone.json').read_text())
    def call(path,data=None,key=api_key):
        req=urllib.request.Request(args.url+path,None if data is None else data.encode(),
                                   {'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=120) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    assert call('/v1/models',key=api_key+'-invalid')[0]==401
    assert call('/v1/models')[0]==200
    assert call('/v1/systemone','{')[0]==422
    assert call('/v1/systemone',json.dumps(dict(payload,model='bad')))[0]==422
    assert call('/v1/systemone','{"state":NaN}')[0]==422
    invalid=dict(payload,questions={'x':{'type':'score','criteria':['only one']}})
    assert call('/v1/systemone',json.dumps(invalid))[0]==422
    code,raw=call('/v1/systemone',json.dumps(payload))
    assert code==200 and set(raw)=={'model','answers','usage'}
    assert set(raw['answers']['refund'])=={'type','noul'}
    s=raw['answers']['urgency']
    assert abs(s['score']-sum(int(k)*v for k,v in s['probabilities'].items()))<1e-9
    for a in raw['answers'].values():
        if 'probabilities' in a: assert abs(sum(a['probabilities'].values())-1)<1e-9
    print('PASS: HTTP contract, authentication, validation, mixed questions, probability sums')
    if not args.sdk:
        return
    from typesafe_sdk import TypeSafeClient,AsyncTypeSafeClient,Choice,Score,Noul
    questions={'binary':Noul(instructions='Is the state about a refund?',criteria={'true':{'meaning':'refund'},'false':['other']}),
               'large':Choice(instructions='Choose the named number.',criteria={str(i):None for i in range(27)}),
               'rating':Score(instructions={'question':'How urgent?'},criteria=[{'level':'low'},['medium'],'high'])}
    with TypeSafeClient(api_key=api_key,base_url=args.url) as client:
        result=client.system_one(state=['Refund requested.', 'Number 12.'],questions=questions)
        assert len(result.answers['large'].probabilities)==27
        assert set(result.answers['rating'].legend)=={0,1,2}
        assert client.models.list().models
    async def check_async():
        async with AsyncTypeSafeClient(api_key=api_key,base_url=args.url) as client:
            response=await client.system_one(state=payload['state'],questions=payload['questions'])
            assert 0 <= response.answers['refund'].noul <= 1
    asyncio.run(check_async())
    print('PASS: HTTP contract, 401/422, structured JSON, mixed questions, 27 choices, sync + async official SDK')

if __name__=='__main__':main()
