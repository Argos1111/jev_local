import unittest
from tools.benchmark_jglue import payload_for, summarize

class BenchmarkTests(unittest.TestCase):
    def test_jnli_direction_and_no_gold_leak(self):
        row={'sentence1':'前提文','sentence2':'仮説文','label':'neutral'}
        p=payload_for('jnli',row,'model')
        self.assertEqual(p['state'],{'前提':'前提文','仮説':'仮説文'})
        row['label']='entailment'
        self.assertEqual(p,payload_for('jnli',row,'model'))
        self.assertEqual(list(p['questions']['answer']['criteria']),['entailment','contradiction','neutral'])
    def test_qa_label_mapping(self):
        row={'question':'問','label':3,**{f'choice{i}':f'答{i}' for i in range(5)}}
        p=payload_for('jcommonsenseqa',row,'model')
        self.assertEqual(p['questions']['answer']['criteria']['3'],'答3')
        row['label']=0
        self.assertEqual(p,payload_for('jcommonsenseqa',row,'model'))
    def test_failed_requests_count_in_denominator(self):
        rows=[{'gold':'a','prediction':'a','correct':True,'latency_ms':10,'confidence':.2,
               'probabilities':{'a':.8,'b':.2},'response':{'model':'m'},'headers':{}},
              {'gold':'b','error':'failure'}]
        r=summarize(rows,1)
        self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['errors'],1)
        self.assertEqual(r['accuracy_successful_only'],1)

if __name__=='__main__':unittest.main()
