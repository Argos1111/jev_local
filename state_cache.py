"""Request-scoped prefix snapshots for llama.cpp hybrid/recurrent models.

The adapter and llama-server must share a directory; inference slots must be
exclusive to this adapter. Save/restore transfers attention AND recurrent state.
"""
from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
from queue import Queue
import uuid
from urllib.error import HTTPError

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / '.cache' / 'slots'


def common_prefix(prompts):
    if not prompts: return []
    n = min(map(len, prompts))
    first = prompts[0]
    for other in prompts[1:]:
        for i in range(n):
            if first[i] != other[i]:
                n = i
                break
    # Keep at least one token for the question's own logits evaluation.
    return first[:min(n, min(map(len, prompts))-1)]


class SharedStateCache:
    def __init__(self, backend_url, slots, directory=DEFAULT_CACHE_DIR):
        if slots < 1: raise ValueError('slots must be positive')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest = hashlib.sha256(backend_url.rstrip('/').encode()).hexdigest()[:16]
        self.lock_file = (self.directory / f'owner-{digest}.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            raise RuntimeError('This backend already has a shared-state adapter; use a dedicated backend')
        self.slots = Queue()
        for slot in range(slots): self.slots.put(slot)

    def close(self):
        self.lock_file.close()

    @contextmanager
    def lease(self):
        slot = self.slots.get()
        try: yield slot
        finally: self.slots.put(slot)

    def run(self, backend, state, specs, pool, min_prefix_tokens=0):
        prompts = {key: backend.post('/tokenize', {
            'content':backend.score_prompt(state,spec), 'add_special':True, 'parse_special':True,
        })['tokens'] for key,spec in specs.items()}
        prefix = common_prefix(list(prompts.values()))
        if len(specs) < 2 or len(prefix) < max(1,min_prefix_tokens):
            return self._uncached(backend,state,specs,prompts,pool)
        filename = 'jev-prefix-' + uuid.uuid4().hex + '.bin'
        path = self.directory / filename
        futures = {}
        try:
            with self.lease() as slot:
                # This installed backend emits one token even with n_predict=0.
                # Force an ASCII token to avoid partial UTF-8/parser failures.
                # Its first sampled token is NOT decoded back into the context.
                warm = backend.post('/completion', {
                    'prompt':prefix,'id_slot':slot,'cache_prompt':False,'n_predict':1,
                    'temperature':-1,'grammar':'root ::= "A"',
                })
                if warm.get('truncated'):
                    raise RuntimeError('Shared prefix exceeded backend context')
                try:
                    saved = backend.post(f'/slots/{slot}?action=save',{'filename':filename})
                except HTTPError as exc:
                    raise RuntimeError('Cannot save shared prefix. Restart llama-server with --slot-save-path matching the adapter --slot-cache-dir, or use --state-cache off') from exc
                if saved.get('n_saved') != len(prefix):
                    raise RuntimeError('Snapshot token count differs from exact common prefix')
                if not path.is_file():
                    raise RuntimeError('Adapter and llama-server must use the same --slot-save-path directory')
            def score(key,spec):
                with self.lease() as slot:
                    def restore():
                        r = backend.post(f'/slots/{slot}?action=restore',{'filename':filename})
                        if r.get('n_restored') != len(prefix):
                            raise RuntimeError('Incomplete shared prefix restore')
                    result = backend.score(state,spec,True,prompt_tokens=prompts[key],slot=slot,before_attempt=restore)
                    if any(t.get('cache_n',0) < len(prefix) for t in result['cache_attempts']):
                        raise RuntimeError('Backend did not reuse the restored prefix; refusing silent fallback')
                    return result
            futures = {key:pool.submit(score,key,spec) for key,spec in specs.items()}
            results = {key:future.result() for key,future in futures.items()}
            attempts = [t for result in results.values() for t in result['cache_attempts']]
            profile = {
                'mode':'shared', 'common_prefix_tokens':len(prefix), 'snapshot_bytes':saved['n_written'],
                'prefill_prompt_tokens':warm['timings']['prompt_n'],
                'cached_tokens':sum(t['cache_n'] for t in attempts),
                'processed_tokens':warm['timings']['prompt_n']+sum(t['prompt_n'] for t in attempts),
                'question_cache_tokens':{k:[t['cache_n'] for t in r['cache_attempts']] for k,r in results.items()},
                'prefill_usage':{'input_tokens':warm.get('tokens_evaluated',0),'output_tokens':warm.get('tokens_predicted',0)},
            }
            return results,profile
        finally:
            # Wait even after errors: no worker may restore a file after deletion.
            for future in futures.values():
                try: future.result()
                except Exception: pass
            path.unlink(missing_ok=True)

    def _uncached(self,backend,state,specs,prompts,pool):
        def score(key,spec):
            with self.lease() as slot:
                return backend.score(state,spec,False,prompt_tokens=prompts[key],slot=slot)
        futures={key:pool.submit(score,key,spec) for key,spec in specs.items()}
        try:
            results={key:f.result() for key,f in futures.items()}
            return results,{'mode':'off-short','common_prefix_tokens':0,'cached_tokens':0,
                            'processed_tokens':sum(t.get('prompt_n',0) for r in results.values() for t in r['cache_attempts'])}
        finally:
            for f in futures.values():
                try:f.result()
                except Exception:pass
