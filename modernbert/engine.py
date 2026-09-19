"""In-process ModernBERT cross-encoder scorer producing the same result shape as JevLocal.score.

Each (context, candidate) pair is encoded as one sequence and mapped to one logit.
Candidate probabilities are the softmax over that question's logits, so the
semantics match the LFM path: normalized over supplied candidates, not calibrated.
"""
import json
import math
from pathlib import Path
import threading
import time

from jev_local import distribution_confidence
from .prompting import FORMAT_VERSION, render_candidates, render_context

MODEL_ID = 'sbintuitions/modernbert-ja-310m'
MODEL_REVISION = '77675fc96a7e445e982e2ba90246b816efc74ec6'
MAX_LENGTH = 512
# Published fine-tuned weights; used when no local checkpoint exists.
HUB_CHECKPOINT = 'argos1111/modernbert-ja-310m-jev'
META_FILE = 'jev_modernbert.json'


def select_device(preference='auto'):
    """cuda covers both NVIDIA and ROCm builds; mps is Apple Silicon (inference only, not verified here)."""
    import torch
    mps_ok = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    if preference == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'mps' if mps_ok else 'cpu')
    device = torch.device(preference)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm device requested but torch reports no GPU')
    if device.type == 'mps' and not mps_ok:
        raise RuntimeError('MPS device requested but torch reports no Apple GPU')
    return device


def load_tokenizer(base_model=MODEL_ID, revision=MODEL_REVISION):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token_id is None or tokenizer.sep_token_id is None:
        raise RuntimeError('Tokenizer must define pad and sep tokens')
    return tokenizer


def encode_pairs(tokenizer, contexts, candidates, max_length=MAX_LENGTH):
    """Encode as <s> context </s><s> candidate </s>; truncate the context side only."""
    return tokenizer(contexts, candidates, padding=True, truncation='only_first',
                     max_length=max_length, return_tensors='pt')


def resolve_checkpoint(checkpoint):
    """Accept a local directory or a Hub repo id (optionally 'repo@revision').

    Returns (source, from_pretrained kwargs, metadata). The metadata file is what
    distinguishes a Jev checkpoint from an arbitrary ModernBERT classifier.
    """
    path = Path(checkpoint)
    if path.is_dir():
        meta_path = path/META_FILE
        if not meta_path.is_file():
            raise RuntimeError(f'Not a Jev ModernBERT checkpoint: {path}')
        return str(path), {}, json.loads(meta_path.read_text())
    text = str(checkpoint)
    repo, _, revision = text.partition('@')
    # Hub ids are exactly 'owner/name'; anything else is a missing local path.
    if repo.count('/') != 1 or text.startswith(('.', '/', '~')) or repo.split('/')[0] in ('models', 'model'):
        raise RuntimeError(f'Checkpoint not found: {checkpoint}. Train with ./train_modernbert.sh or pass a Hub id like {HUB_CHECKPOINT}')
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError
    try:
        meta_path = hf_hub_download(repo, META_FILE, revision=revision or None)
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f'Hub repository not found: {repo}') from exc
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(f'{repo} is not cached and the Hub is unreachable (offline?). Unset HF_HUB_OFFLINE or check the network.') from exc
    except EntryNotFoundError as exc:
        raise RuntimeError(f'{repo} has no {META_FILE}; not a Jev ModernBERT checkpoint') from exc
    return repo, {'revision': revision or None}, json.loads(Path(meta_path).read_text())


class CrossEncoder:
    """Single logit per (context, candidate) pair. Untrained heads are refused at serve time."""

    def __init__(self, checkpoint=None, base_model=MODEL_ID, revision=MODEL_REVISION,
                 device='auto', dtype='bfloat16', allow_untrained=False):
        import torch
        from transformers import AutoModelForSequenceClassification
        self.device = select_device(device)
        # bf16 matmul is only reliable on CUDA/ROCm; MPS and CPU run in float32.
        self.dtype = getattr(torch, dtype) if self.device.type == 'cuda' else torch.float32
        self.meta = {'base_model': base_model, 'revision': revision, 'format_version': FORMAT_VERSION, 'trained': False}
        source = base_model
        kwargs = {'revision': revision}
        if checkpoint is not None:
            source, kwargs, self.meta = resolve_checkpoint(checkpoint)
            if self.meta.get('format_version') != FORMAT_VERSION:
                raise RuntimeError(f'Checkpoint format {self.meta.get("format_version")!r} does not match {FORMAT_VERSION!r}')
        elif not allow_untrained:
            raise RuntimeError('Base ModernBERT has a randomly initialized head. Train with '
                               'python -m modernbert.train or pass allow_untrained for smoke tests.')
        self.tokenizer = load_tokenizer(base_model, revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=1, dtype=self.dtype, **kwargs).to(self.device).eval()
        self.max_length = int(self.meta.get('max_length', MAX_LENGTH))
        self.lock = threading.Lock()
        self.model_id = self.meta.get('model_id', 'modernbert-ja-310m-untrained')

    def logits(self, contexts, candidates):
        import torch
        batch = encode_pairs(self.tokenizer, contexts, candidates, self.max_length)
        overflow = batch['input_ids'].shape[1] >= self.max_length
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.lock, torch.inference_mode():
            out = self.model(**batch).logits[:, 0].float().cpu()
        return out.tolist(), int(batch['attention_mask'].sum().item()), overflow

    def score(self, state, spec, **_ignored):
        candidates = render_candidates(spec)
        if len(candidates) < 2:
            raise ValueError('Scoring requires at least two candidates')
        started = time.perf_counter()
        context = render_context(state, spec)
        raw, tokens, truncated = self.logits([context] * len(candidates), candidates)
        if not all(math.isfinite(x) for x in raw):
            raise RuntimeError('Non-finite candidate logit')
        shift = max(raw)
        weights = [math.exp(x - shift) for x in raw]
        total = sum(weights)
        conditional = [w / total for w in weights]
        best = max(range(len(raw)), key=raw.__getitem__)
        return {
            'question': spec['question'],
            **distribution_confidence(conditional),
            'value': spec['choices'][best],
            'probabilities': [{'value': v, 'probability': p, 'candidate': c, 'logit': l}
                              for v, p, c, l in zip(spec['choices'], conditional, candidates, raw)],
            'usage': {'input_tokens': tokens, 'output_tokens': 0},
            'context_truncated': truncated,
            'elapsed_ms': (time.perf_counter() - started) * 1000,
        }

    def score_many(self, state, specs):
        """One batched forward pass for all questions of a request."""
        import torch
        keys = [k for k, s in specs.items() if len(s['choices']) > 1]
        if not keys:
            return {}
        started = time.perf_counter()
        contexts, candidates, spans = [], [], {}
        for key in keys:
            cands = render_candidates(specs[key])
            spans[key] = (len(contexts), len(cands))
            contexts.extend([render_context(state, specs[key])] * len(cands))
            candidates.extend(cands)
        raw, tokens, truncated = self.logits(contexts, candidates)
        results = {}
        for key in keys:
            start, n = spans[key]
            logits = raw[start:start + n]
            if not all(math.isfinite(x) for x in logits):
                raise RuntimeError('Non-finite candidate logit')
            shift = max(logits)
            weights = [math.exp(x - shift) for x in logits]
            total = sum(weights)
            conditional = [w / total for w in weights]
            best = max(range(n), key=logits.__getitem__)
            spec = specs[key]
            results[key] = {
                'question': spec['question'],
                **distribution_confidence(conditional),
                'value': spec['choices'][best],
                'probabilities': [{'value': v, 'probability': p, 'candidate': c, 'logit': l}
                                  for v, p, c, l in zip(spec['choices'], conditional, candidates[start:start + n], logits)],
                'usage': {'input_tokens': tokens if key == keys[0] else 0, 'output_tokens': 0},
                'context_truncated': truncated,
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            }
        return results
