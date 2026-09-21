import json
import unittest
from unittest.mock import patch
from systemone import AdapterBackend, SystemOne, ValidationError, to_spec, validate
from api_server import parse_json
from jev_local import distribution_confidence

MODEL='test-local-model'

class FakeBackend:
    def __init__(self): self.calls=[]
    def prepare(self,payload): self.payload=payload
    def score(self,state,spec):
        self.calls.append((state,spec))
        n=len(spec['choices'])
        probs=([.1,.2,.7] if n==3 else [.25,.75] if n==2 else [1/n]*n)
        best=max(range(n),key=probs.__getitem__)
        return {'value':spec['choices'][best], 'probabilities':[{'value':v,'probability':p} for v,p in zip(spec['choices'],probs)],
                **distribution_confidence(probs), 'usage':{'input_tokens':10,'output_tokens':1}}

class LabelTokenizationTests(unittest.TestCase):
    def test_multidigit_tokenizer_rejects_large_choice_as_validation_error(self):
        backend = AdapterBackend()
        spec = to_spec({'type': 'choice', 'criteria': {str(i): None for i in range(27)}})
        with patch.object(backend, 'post', side_effect=lambda path, data: {'tokens': list(data['content'])}):
            with self.assertRaisesRegex(ValidationError, 'one token') as error:
                backend.prepare({'questions': {'large': spec}})
        self.assertEqual(error.exception.detail[0]['loc'], ['body', 'questions', 'large', 'criteria'])
        self.assertNotIn('10', backend.ids)

    def test_letter_labels_and_supported_numeric_labels_still_work(self):
        for n in (26, 27, 255):
            backend = AdapterBackend()
            spec = to_spec({'type': 'choice', 'criteria': {str(i): None for i in range(n)}})
            with patch.object(backend, 'post', side_effect=lambda path, data: {'tokens': [data['content']]}):
                backend.prepare({'questions': {'q': spec}})
            self.assertEqual(len(backend.ids), n)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.backend=FakeBackend()
        self.engine=SystemOne(model_id=MODEL,backend_factory=lambda:self.backend,state_cache=False)
        self.addCleanup(self.engine.close)
        self.payload={'model':'jev-latest','state':{'customer':'test'},'questions':{
            'yes':{'type':'noul','instructions':'True?','criteria':{'true':{'rule':'yes'},'false':['no']}},
            'team':{'type':'choice','instructions':{'question':'Route?'},'criteria':{'a':None,'b':'B','c':{'description':'C'}}},
            'rating':{'type':'score','criteria':['low',{'level':'middle'},['high']]}}}

    def test_mixed_contract_and_usage(self):
        r=self.engine.evaluate(self.payload)
        self.assertEqual(set(r),{'model','answers','usage'})
        self.assertEqual(r['model'],MODEL)
        self.assertEqual(r['answers']['yes'],{'type':'noul','noul':.25})
        self.assertEqual(r['answers']['team']['choice'],'c')
        score=r['answers']['rating']
        self.assertAlmostEqual(score['score'],1.6)
        self.assertEqual(score['legend']['1'],{'level':'middle'})
        self.assertEqual(set(score['probabilities']),{'0','1','2'})
        self.assertEqual(r['usage'],{'input_tokens':30,'output_tokens':3})
        self.assertTrue(all(json.loads(state)==self.payload['state'] for state,_ in self.backend.calls))

    def test_question_ids_do_not_enter_inference(self):
        self.engine.evaluate(self.payload)
        self.assertNotIn('team',self.backend.payload['questions']['team']['question'])
        first=[spec for _,spec in self.backend.calls]
        self.backend.calls.clear()
        self.payload['questions']={f'renamed{i}':v for i,v in enumerate(self.payload['questions'].values())}
        self.engine.evaluate(self.payload)
        self.assertCountEqual(first,[spec for _,spec in self.backend.calls])

    def test_choice_limit_and_singleton(self):
        for n in [1,26,27,255]:
            q={'type':'choice','criteria':{f'item{i}':None for i in range(n)}}
            self.payload['questions']={'q':q}
            r=self.engine.evaluate(self.payload)['answers']['q']
            self.assertEqual(len(r['probabilities']),n)
            self.assertAlmostEqual(sum(r['probabilities'].values()),1)
        self.payload['questions']['q']['criteria']['overflow']=None
        with self.assertRaises(ValidationError):self.engine.evaluate(self.payload)

    def test_bad_requests_before_inference(self):
        for field,bad in [('model','nonexistent'),('state',False),('questions',{})]:
            payload=dict(self.payload,**{field:bad})
            with self.subTest(field=field),self.assertRaises(ValidationError):self.engine.evaluate(payload)
        for q in [{'type':'unknown'}, {'type':'score','criteria':['one']},
                  {'type':'score','criteria':['x']*11}, {'type':'choice','criteria':[]},
                  {'type':'noul'}, {'type':'noul','instructions':'?','criteria':{'maybe':'?'}}]:
            with self.subTest(q=q),self.assertRaises(ValidationError):
                self.engine.evaluate(dict(self.payload,questions={'q':q}))
        self.assertEqual(self.backend.calls,[])

    def test_json_rejects_nonfinite_and_duplicate_keys(self):
        for raw in ['{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}']:
            with self.assertRaises(ValueError):parse_json(raw)

if __name__=='__main__':unittest.main()
