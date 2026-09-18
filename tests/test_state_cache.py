import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from state_cache import SharedStateCache, common_prefix

class Backend:
    def __init__(self, directory, fail=False):
        self.directory=Path(directory)
        self.context={}
        self.restores=0
        self.fail=fail
    def score_prompt(self,state,spec): return json.dumps(state+[spec])
    def post(self,path,body):
        if path=='/tokenize': return {'tokens':json.loads(body['content'])}
        if path=='/completion':
            self.context[body['id_slot']]=body['prompt'][:]
            return {'timings':{'prompt_n':len(body['prompt'])}}
        slot=int(path.split('/')[2].split('?')[0])
        file=self.directory/body['filename']
        if path.endswith('save'):
            file.write_text(json.dumps(self.context[slot]))
            return {'n_saved':len(self.context[slot]),'n_written':file.stat().st_size}
        self.context[slot]=json.loads(file.read_text())
        self.restores+=1
        return {'n_restored':len(self.context[slot])}
    def score(self,state,spec,cache,*,prompt_tokens,slot,before_attempt=None):
        attempts=[]
        for _ in range(2):
            if before_attempt: before_attempt()
            if cache:
                assert self.context[slot]==state, 'Another question or request contaminated the prefix'
            self.context[slot]=prompt_tokens[:]
            if self.fail: raise RuntimeError('test failure')
            attempts.append({'cache_n':len(state) if cache else 0,'prompt_n':1 if cache else len(prompt_tokens)})
        return {'cache_attempts':attempts}

class StateCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache=SharedStateCache('fake',2,self.tmp.name)
        self.addCleanup(self.cache.close)
        self.pool=ThreadPoolExecutor(4)
        self.addCleanup(self.pool.shutdown)
    def test_prefix_boundaries(self):
        self.assertEqual(common_prefix([[1,2,3],[1,2,4]]),[1,2])
        self.assertEqual(common_prefix([[1,2],[1,2]]),[1])
        self.assertEqual(common_prefix([[1],[2]]),[])
        self.assertEqual(common_prefix([]),[])
    def test_restore_each_retry_and_cleanup(self):
        backend=Backend(self.tmp.name)
        _,profile=self.cache.run(backend,[1,2,3],{'a':4,'b':5},self.pool)
        self.assertEqual(backend.restores,4)
        self.assertEqual(profile['cached_tokens'],12)
        self.assertEqual(list(Path(self.tmp.name).glob('*.bin')),[])
    def test_failure_cleanup_and_slot_return(self):
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            self.cache.run(Backend(self.tmp.name,True),[1,2],{'a':3,'b':4},self.pool)
        self.assertEqual(list(Path(self.tmp.name).glob('*.bin')),[])
        self.assertEqual(self.cache.slots.qsize(),2)
    def test_short_prefix_skips_snapshot(self):
        backend=Backend(self.tmp.name)
        _,profile=self.cache.run(backend,[1,2],{'a':3,'b':4},self.pool,256)
        self.assertEqual(profile['mode'],'off-short')
        self.assertEqual(backend.restores,0)
    def test_concurrent_distinct_states(self):
        backend=Backend(self.tmp.name)
        with ThreadPoolExecutor(2) as requests:
            jobs=[requests.submit(self.cache.run,backend,state,{'a':8,'b':9},self.pool)
                  for state in ([1,2,3],[4,5,6,7])]
            self.assertEqual([j.result()[1]['common_prefix_tokens'] for j in jobs],[3,4])
        self.assertEqual(list(Path(self.tmp.name).glob('*.bin')),[])
    def test_duplicate_owner_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'already has'):
            SharedStateCache('fake',2,self.tmp.name)

if __name__=='__main__': unittest.main()
