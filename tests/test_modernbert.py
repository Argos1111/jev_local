"""ModernBERT adapter contract tests; no torch, model download or GPU required."""
import json
import math
import random
import unittest

from jev_local import distribution_confidence
from modernbert.prompting import FORMAT_VERSION, render_candidates, render_context
from modernbert.systemone import ModernBertSystemOne
from systemone import ValidationError, answer_for, to_spec


class FakeEncoder:
    """Deterministic stand-in: longer candidate text gets a higher logit."""
    model_id = 'fake-modernbert'

    def __init__(self):
        self.calls = []

    def score_many(self, state, specs):
        self.calls.append((state, json.loads(json.dumps(specs))))
        results = {}
        for key, spec in specs.items():
            if len(spec['choices']) < 2:
                continue
            cands = render_candidates(spec)
            logits = [float(len(c)) for c in cands]
            shift = max(logits)
            weights = [math.exp(x - shift) for x in logits]
            probs = [w / sum(weights) for w in weights]
            best = max(range(len(logits)), key=logits.__getitem__)
            results[key] = {'question': spec['question'], **distribution_confidence(probs),
                            'value': spec['choices'][best],
                            'probabilities': [{'value': v, 'probability': p} for v, p in zip(spec['choices'], probs)],
                            'usage': {'input_tokens': 7, 'output_tokens': 0}, 'context_truncated': False}
        return results


class PromptingTests(unittest.TestCase):
    def test_context_matches_api_content_rules(self):
        spec = {'question': 'Q?', 'choices': [True, False]}
        self.assertEqual(render_context('plain', spec), '質問: Q?\n状況: plain')
        self.assertEqual(render_context({'a': 1}, spec), '質問: Q?\n状況: {"a": 1}')

    def test_candidates_render_labels_and_descriptions(self):
        spec = to_spec({'type': 'choice', 'instructions': 'x', 'criteria': {'billing': '請求', 'sales': None}})
        self.assertEqual(render_candidates(spec), ['billing — 請求', 'sales'])
        spec = to_spec({'type': 'score', 'criteria': ['low', {'level': 'mid'}]})
        self.assertEqual(render_candidates(spec), ['0 — low', '1 — {"level": "mid"}'])
        spec = to_spec({'type': 'noul', 'instructions': 'x'})
        self.assertEqual(render_candidates(spec), ['true', 'false'])

    def test_format_version_is_pinned(self):
        self.assertTrue(FORMAT_VERSION.startswith('modernbert-jev/'))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FakeEncoder()
        self.engine = ModernBertSystemOne(self.encoder)
        self.payload = {'model': 'jev-latest', 'state': {'customer': 'test'}, 'questions': {
            'yes': {'type': 'noul', 'instructions': 'True?', 'criteria': {'true': 'a longer description', 'false': 'no'}},
            'team': {'type': 'choice', 'instructions': 'Route?', 'criteria': {'a': None, 'bb': None, 'c': 'C description'}},
            'rating': {'type': 'score', 'criteria': ['low', 'middle', 'the highest level']},
            'only': {'type': 'choice', 'criteria': {'single': None}}}}

    def test_contract_shape_and_single_batched_call(self):
        diagnostics = {}
        result = self.engine.evaluate(self.payload, diagnostics=diagnostics)
        self.assertEqual(set(result), {'model', 'answers', 'usage'})
        self.assertEqual(result['model'], 'fake-modernbert')
        self.assertEqual(len(self.encoder.calls), 1)
        answers = result['answers']
        self.assertEqual(set(answers['yes']), {'type', 'noul'})
        self.assertGreater(answers['yes']['noul'], 0.5)
        self.assertEqual(answers['team']['choice'], 'c')
        self.assertAlmostEqual(sum(answers['team']['probabilities'].values()), 1)
        self.assertEqual(answers['rating']['legend']['2'], 'the highest level')
        self.assertAlmostEqual(answers['rating']['score'], sum(int(k) * v for k, v in answers['rating']['probabilities'].items()))
        self.assertEqual(answers['only'], {'type': 'choice', 'choice': 'single', 'probabilities': {'single': 1.0}, 'confidence': 1.0})
        self.assertEqual(result['usage'], {'input_tokens': 21, 'output_tokens': 0})
        self.assertEqual(diagnostics['mode'], 'encoder-batch')

    def test_images_are_rejected_with_validation_error(self):
        payload = dict(self.payload, images=['data:image/png;base64,' + 'A' * 100])
        with self.assertRaises(ValidationError) as ctx:
            self.engine.evaluate(payload)
        self.assertEqual(ctx.exception.detail[0]['loc'], ['body', 'images'])

    def test_question_ids_never_reach_the_encoder(self):
        self.engine.evaluate(self.payload)
        state, specs = self.encoder.calls[0]
        rendered = '\n'.join(render_context(state, s) + '\n'.join(render_candidates(s)) for s in specs.values())
        self.assertNotIn('rating', rendered)
        self.assertNotIn('team', rendered)
        self.assertIn('Route?', rendered)

    def test_models_endpoint_lists_aliases(self):
        names = [m['name'] for m in self.engine.models()['models']]
        self.assertEqual(names, ['fake-modernbert', 'jev-latest', 'jev-preview'])


class TrainingDataTests(unittest.TestCase):
    def test_builders_produce_valid_examples_without_network(self):
        from modernbert import data
        samples = {
            'jnli': [{'sentence1': 'p', 'sentence2': 'h', 'label': 'contradiction'}],
            'jcommonsenseqa': [{'question': 'q', 'label': 4, **{f'choice{i}': f'c{i}' for i in range(5)}}],
            'jsts': [{'sentence1': 'a', 'sentence2': 'b', 'label': 4.6}],
            'jcola': [{'sentence': 's', 'label': 0}],
        }
        def fake_fetch(name, directory=None):
            task = name.split('-')[0]
            if task == 'moral':
                return b',sent,label\n0,x,1\n'
            return '\n'.join(json.dumps(r) for r in samples[task]).encode()
        original = data.fetch
        data.fetch = fake_fetch
        try:
            rng = random.Random(0)
            rows = []
            for task in ['jnli', 'jcommonsenseqa', 'jsts', 'jcola', 'moral']:
                rows.extend(data.BUILDERS[task]('train', rng))
        finally:
            data.fetch = original
        for row in rows:
            self.assertEqual(len(row['choices']), len(row['descriptions']))
            self.assertTrue(0 <= row['answer'] < len(row['choices']))
            render_candidates(row)
            render_context(row['state'], row)
        by_task = {r['task']: r for r in rows}
        self.assertEqual(by_task['jnli']['choices'][by_task['jnli']['answer']], 'contradiction')
        self.assertEqual(by_task['jcommonsenseqa']['choices'][4], 'c4')
        self.assertEqual(by_task['jsts']['answer'], 5)
        self.assertEqual(by_task['jcola']['answer'], 1)
        self.assertEqual(by_task['moral']['answer'], 0)

    def test_answer_for_matches_llama_backend_shapes(self):
        spec = to_spec({'type': 'score', 'criteria': ['a', 'b']})
        result = {'value': 1, 'confidence': 0.5, 'probabilities': [{'value': 0, 'probability': .25}, {'value': 1, 'probability': .75}]}
        self.assertAlmostEqual(answer_for({'type': 'score', 'criteria': ['a', 'b']}, spec, result)['score'], .75)


class SetupTests(unittest.TestCase):
    def test_gfx_target_decoding(self):
        import tempfile
        from pathlib import Path
        from scripts import setup_modernbert
        with tempfile.TemporaryDirectory() as directory:
            nodes = Path(directory)
            for index, version in enumerate([0, 120001, 110000, 90402]):
                (nodes/str(index)).mkdir()
                (nodes/str(index)/'properties').write_text(f'cpu_cores_count 0\ngfx_target_version {version}\n')
            self.assertEqual(setup_modernbert.amd_gfx_targets(nodes), ['gfx1100', 'gfx1201', 'gfx942'])

    def test_pip_arguments_pin_torch_per_backend(self):
        from unittest.mock import patch
        from scripts import setup_modernbert
        with patch.object(setup_modernbert, 'detect_backend', return_value=('rocm', ['gfx1201'])), \
             patch.object(setup_modernbert.sys, 'argv', ['setup', '--dry-run']), \
             patch('builtins.print') as printed:
            setup_modernbert.main()
        commands = [str(call.args[0]) for call in printed.call_args_list]
        self.assertTrue(any('torch[device-gfx1201]==2.13.0+rocm10.0.0' in c for c in commands))


if __name__ == '__main__':
    unittest.main()
