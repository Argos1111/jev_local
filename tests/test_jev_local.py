"""Regression checks for missing-candidate handling and typed mapping."""
import math
import unittest
from jev_local import JevLocal, distribution_confidence

class Fake(JevLocal):
    def __init__(self, missing_forever=False):
        super().__init__()
        self.ids={'A':41,'B':42}
        self.requests=[]
        self.missing_forever=missing_forever
    def prompt(self,text): return text
    def post(self,path,data):
        self.requests.append(data)
        entries=[{'id':41,'logprob':math.log(.2)}]
        if data['n_probs']>100 and not self.missing_forever:
            entries.append({'id':42,'logprob':math.log(.3)})
        return {'completion_probabilities':[{'token':'X','top_logprobs':entries}],'timings':{}}

class ScoringTests(unittest.TestCase):
    def test_missing_candidate_retries_and_preserves_boolean(self):
        api=Fake()
        result=api.score('state',{'question':'?', 'choices':[True,False]})
        self.assertIs(result['value'],False)
        self.assertAlmostEqual(result['probabilities'][0]['probability'],.4)
        self.assertAlmostEqual(result['candidate_mass'],.5)
        self.assertEqual([r['n_probs'] for r in api.requests],[100,1000])
        self.assertTrue(all(r['post_sampling_probs'] is False for r in api.requests))
    def test_missing_candidate_fails_instead_of_zero_fill(self):
        with self.assertRaises(RuntimeError):
            Fake(True).score('state',{'question':'?', 'choices':[True,False]})

class ConfidenceTests(unittest.TestCase):
    def test_uniform_and_certain_endpoints(self):
        for probs in ([.5, .5], [1/3, 1/3, 1/3]):
            self.assertAlmostEqual(distribution_confidence(probs)['confidence'], 0)
        self.assertEqual(distribution_confidence([1., 0., 0.])['confidence'], 1)

    def test_concentration_and_label_order(self):
        a = distribution_confidence([.9, .1])
        self.assertAlmostEqual(a['confidence'], 0.5310044064)
        self.assertAlmostEqual(a['top_two_margin'], .8)
        self.assertEqual(a, distribution_confidence([.1, .9]))
        self.assertGreater(distribution_confidence([.99, .01])['confidence'], a['confidence'])

    def test_score_contains_confidence_without_extra_requests(self):
        api = Fake()
        result = api.score('state', {'question':'?', 'choices':[True, False]})
        self.assertAlmostEqual(result['confidence'], distribution_confidence([.4,.6])['confidence'])
        self.assertEqual(result['confidence_method'], 'one_minus_normalized_entropy')
        self.assertEqual(len(api.requests), 2)

if __name__=='__main__': unittest.main()
