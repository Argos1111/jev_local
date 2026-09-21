"""TypeSafe HTTP contract adapter over a local llama.cpp model, not Jev weights."""
from concurrent.futures import ThreadPoolExecutor
import json
from state_cache import SharedStateCache, DEFAULT_CACHE_DIR
from jev_local import JevLocal
from image_input import validate_images

ALIASES = ('jev-latest', 'jev-preview')

class ValidationError(ValueError):
    def __init__(self, path, message):
        self.detail = [{'loc': ['body', *path], 'msg': message, 'type': 'value_error'}]
        super().__init__(message)


def content(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def validate(payload, model_id):
    def require(ok, path, message):
        if not ok: raise ValidationError(path, message)
    require(isinstance(payload, dict), [], 'Expected an object')
    require(isinstance(payload.get('model'), str), ['model'], 'model is required')
    require(payload['model'] in (*ALIASES, model_id), ['model'], 'Unknown model; see GET /v1/models')
    require(isinstance(payload.get('state'), (str,dict,list)), ['state'], 'Expected string, object or array')
    try:
        validate_images(payload.get('images', []))
    except ValueError as exc:
        raise ValidationError(['images'], str(exc)) from exc
    questions = payload.get('questions')
    require(isinstance(questions,dict) and bool(questions), ['questions'], 'Expected a nonempty question map')
    for key,q in questions.items():
        path = ['questions',key]
        require(isinstance(q,dict), path, 'Expected a question object')
        kind = q.get('type')
        require(kind in ('noul','choice','score'), path+['type'], 'Expected noul, choice or score')
        # SDK accepts omitted/null instructions; descriptions must then carry the task.
        instruction = q.get('instructions')
        require(instruction is None or isinstance(instruction,(str,dict,list)), path+['instructions'], 'Expected string, object, array or null')
        criteria = q.get('criteria')
        if kind == 'choice':
            require(isinstance(criteria,dict) and 1 <= len(criteria) <= 255, path+['criteria'], 'Choice requires 1..255 options')
            for label,desc in criteria.items():
                require(desc is None or isinstance(desc,(str,dict,list)), path+['criteria',label], 'Expected string, object, array or null')
        elif kind == 'score':
            require(isinstance(criteria,list) and 2 <= len(criteria) <= 10, path+['criteria'], 'Score requires 2..10 ordered levels')
            for i,desc in enumerate(criteria):
                require(isinstance(desc,(str,dict,list)), path+['criteria',i], 'Expected string, object or array')
        else:
            require(criteria is None or isinstance(criteria,dict), path+['criteria'], 'Expected true/false descriptions')
            if criteria is not None:
                require(set(criteria) <= {'true','false'}, path+['criteria'], 'Only true and false keys are allowed')
                for label,desc in criteria.items():
                    require(desc is None or isinstance(desc,(str,dict,list)), path+['criteria',label], 'Expected string, object, array or null')
            require(instruction is not None or bool(criteria), path, 'Noul requires instructions or criteria')
    return payload


def to_spec(q):
    kind = q['type']
    criteria = q.get('criteria')
    defaults = {'choice':'Choose the best matching option.', 'score':'Select the most appropriate rubric level.', 'noul':'Is the statement true?'}
    spec = {'question': content(q['instructions']) if q.get('instructions') is not None else defaults[kind]}
    if kind == 'choice':
        spec.update(choices=list(criteria), descriptions=list(criteria.values()))
    elif kind == 'score':
        spec.update(choices=list(range(len(criteria))), descriptions=criteria)
    else:
        spec.update(choices=[True,False], descriptions=[(criteria or {}).get('true'), (criteria or {}).get('false')])
    if len(spec['choices']) > 26:
        spec['tokens'] = [str(i) for i in range(len(spec['choices']))]
    return spec


def answer_for(question, spec, result):
    """Typed answer from candidate probabilities; shared by every local backend."""
    p = [v['probability'] for v in result['probabilities']]
    if question['type'] == 'noul':
        return {'type':'noul','noul':p[0]}
    if question['type'] == 'choice':
        return {'type':'choice','choice':result['value'],
                'probabilities':dict(zip(spec['choices'],p)), 'confidence':result['confidence']}
    return {'type':'score', 'score':sum(i*v for i,v in enumerate(p)),
            'legend':{str(i):v for i,v in enumerate(question['criteria'])},
            'probabilities':{str(i):v for i,v in enumerate(p)}, 'confidence':result['confidence']}


def singleton_answer(spec):
    # A singleton is deterministic by its schema; no inference needed.
    return {'type':'choice','choice':spec['choices'][0], 'probabilities':{spec['choices'][0]:1.0},'confidence':1.0}


class AdapterBackend(JevLocal):
    def prepare(self, payload):
        # Larger Choice sets use numeric labels only if the tokenizer supports
        # them. Sarashina splits multi-digit numbers; reject rather than truncate.
        for key, spec in payload['questions'].items():
            for label in self.labels(spec):
                if label not in self.ids:
                    ids = self.post('/tokenize', {'content':label, 'add_special':False})['tokens']
                    if len(ids) != 1:
                        raise ValidationError(['questions', key, 'criteria'],
                                              f'Backend cannot represent option label {label!r} as one token; reduce the number of choices or use a backend with single-token numeric labels')
                    self.ids[label] = ids[0]


class SystemOne:
    def __init__(self, backend_url='http://127.0.0.1:8097', model_id='lfm2.5-vl-1.6b-q8_0', workers=4, backend_factory=None, state_cache='auto', cache_dir=DEFAULT_CACHE_DIR, cache_min_tokens=256):
        self.model_id = model_id
        self.backend_factory = backend_factory or (lambda: AdapterBackend(backend_url))
        mode = ('shared' if state_cache else 'off') if isinstance(state_cache,bool) else state_cache
        if mode not in ('auto','shared','off'): raise ValueError('Invalid state cache mode')
        self.cache_min_tokens = cache_min_tokens if mode == 'auto' else 0
        self.cache = SharedStateCache(backend_url, workers, cache_dir) if mode != 'off' else None
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def close(self):
        self.pool.shutdown(wait=True)
        if self.cache: self.cache.close()

    def evaluate(self, payload, *, diagnostics=None):
        validate(payload,self.model_id)
        specs = {key:to_spec(q) for key,q in payload['questions'].items()}
        # Per-request tokenizer state avoids mutations shared between HTTP threads.
        backend = self.backend_factory()
        backend.prepare({'questions':specs})
        state = content(payload['state'])
        active = {k:s for k,s in specs.items() if len(s['choices']) > 1}
        results, profile, futures = {}, {'mode':'off'}, {}
        images = payload.get('images', [])
        if images and active:
            backend.prepare_images(images)
            profile['mode'] = 'off-images'
            def score_image(spec):
                # Participate in the same slot leases as text snapshot requests.
                if self.cache:
                    with self.cache.lease() as slot:
                        return backend.score(state, spec, images=images, slot=slot)
                return backend.score(state, spec, images=images)
            futures = {key:self.pool.submit(score_image,spec) for key,spec in active.items()}
        elif self.cache and active:
            results, profile = self.cache.run(backend,state,active,self.pool,self.cache_min_tokens)
        else:
            futures = {key:self.pool.submit(backend.score,state,spec) for key,spec in active.items()}
        answers = {}
        usage = dict(profile.get('prefill_usage', {'input_tokens':0,'output_tokens':0}))
        try:
            for key,q in payload['questions'].items():
                spec = specs[key]
                if len(spec['choices']) == 1:
                    answers[key] = singleton_answer(spec)
                    continue
                result = results[key] if key in results else futures[key].result()
                results[key] = result
                for field in usage: usage[field] += result['usage'][field]
                answers[key] = answer_for(q, spec, result)
        except Exception:
            for future in futures.values(): future.cancel()
            # Do not release request admission while its inference is still running.
            for future in futures.values():
                try: future.result()
                except Exception: pass
            raise
        if diagnostics is not None:
            if profile['mode'] in ('off', 'off-images'):
                profile.update(cached_tokens=0, processed_tokens=sum(t.get('prompt_n',0) for r in results.values() for t in r.get('cache_attempts',[])))
            diagnostics.update(profile)
        return {'model':self.model_id,'answers':answers,'usage':usage}

    def models(self):
        return {'models':[{'name':name,'description':f'Local adapter for {self.model_id}; not the TypeSafe Jev model.',
                           'release_date':'2026-09-19'} for name in (self.model_id,*ALIASES)]}
