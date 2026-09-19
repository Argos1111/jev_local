"""TypeSafe HTTP contract over an in-process ModernBERT cross-encoder; text only."""
from systemone import ALIASES, ValidationError, answer_for, content, singleton_answer, to_spec, validate


class ModernBertSystemOne:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model_id = encoder.model_id

    def close(self):
        pass

    def evaluate(self, payload, *, diagnostics=None):
        validate(payload, self.model_id)
        if payload.get('images'):
            raise ValidationError(['images'], 'This backend is text only; use ./run.sh --model vision for images')
        specs = {key: to_spec(q) for key, q in payload['questions'].items()}
        state = content(payload['state'])
        results = self.encoder.score_many(state, specs)
        answers, usage = {}, {'input_tokens': 0, 'output_tokens': 0}
        for key, q in payload['questions'].items():
            spec = specs[key]
            if len(spec['choices']) == 1:
                answers[key] = singleton_answer(spec)
                continue
            result = results[key]
            for field in usage:
                usage[field] += result['usage'][field]
            answers[key] = answer_for(q, spec, result)
        if diagnostics is not None:
            diagnostics.update(mode='encoder-batch', common_prefix_tokens=0, cached_tokens=0,
                               processed_tokens=usage['input_tokens'],
                               truncated=any(r.get('context_truncated') for r in results.values()))
        return {'model': self.model_id, 'answers': answers, 'usage': usage}

    def models(self):
        return {'models': [{'name': name,
                            'description': f'Local ModernBERT cross-encoder adapter ({self.model_id}); not the TypeSafe Jev model.',
                            'release_date': '2026-09-19'} for name in (self.model_id, *ALIASES)]}
