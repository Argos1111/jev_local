"""Text rendering shared by training and inference. Both must use this one code path.

A question becomes one context string; each candidate becomes one short string.
The encoder scores (context, candidate) pairs independently, so results do not
depend on candidate order. FORMAT_VERSION is stored with trained checkpoints and
checked at load time so a model is never served with a different rendering.
"""
import json

FORMAT_VERSION = 'modernbert-jev/1'


def content(value):
    """Match systemone.content: strings verbatim, other JSON values serialized."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def render_context(state, spec):
    # Question first so right-side truncation during training removes state text, never the question.
    return f'質問: {spec["question"]}\n状況: {content(state)}'


def render_candidate(label, description=None):
    text = content(label)
    return text if description is None else f'{text} — {content(description)}'


def render_candidates(spec):
    choices = spec['choices']
    descriptions = spec.get('descriptions') or [None] * len(choices)
    if len(descriptions) != len(choices):
        raise ValueError('descriptions must align with choices')
    return [render_candidate(label, description) for label, description in zip(choices, descriptions)]
