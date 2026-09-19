#!/usr/bin/env python3
"""Fine-tune ModernBERT-Ja as a listwise cross-encoder over rendered questions.

Each training item is one question with K candidates; the loss is cross-entropy
over the K pair logits, matching how the engine normalizes at inference.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

from .data import build
from .engine import MAX_LENGTH, MODEL_ID, MODEL_REVISION, encode_pairs, load_tokenizer, select_device
from .prompting import FORMAT_VERSION, render_candidates, render_context

ROOT = Path(__file__).resolve().parents[1]


def flatten(items):
    contexts, candidates, spans = [], [], []
    for item in items:
        cands = render_candidates(item)
        spans.append((len(contexts), len(cands), item['answer']))
        contexts.extend([render_context(item['state'], item)] * len(cands))
        candidates.extend(cands)
    return contexts, candidates, spans


def listwise_loss(logits, spans, weights=None):
    import torch
    losses = []
    for i, (start, n, answer) in enumerate(spans):
        row = logits[start:start + n].unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(row, torch.tensor([answer], device=logits.device))
        losses.append(loss * (weights[i] if weights else 1.0))
    return torch.stack(losses).mean()


def batches(items, pair_budget, rng):
    """Group questions so that the total pair count stays under the budget."""
    order = list(items)
    rng.shuffle(order)
    batch, pairs = [], 0
    for item in order:
        k = len(item['choices'])
        if batch and pairs + k > pair_budget:
            yield batch
            batch, pairs = [], 0
        batch.append(item)
        pairs += k
    if batch:
        yield batch


def evaluate(model, tokenizer, items, device, max_length, pair_budget=256):
    import torch
    model.eval()
    per_task = defaultdict(lambda: {'n': 0, 'correct': 0, 'nll': 0.0})
    rng = random.Random(0)
    with torch.inference_mode():
        for batch in batches(items, pair_budget, rng):
            contexts, candidates, spans = flatten(batch)
            enc = encode_pairs(tokenizer, contexts, candidates, max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[:, 0].float()
            for item, (start, n, answer) in zip(batch, spans):
                row = logits[start:start + n]
                log_probs = torch.log_softmax(row, dim=0)
                stats = per_task[item['task']]
                stats['n'] += 1
                stats['correct'] += int(row.argmax().item() == answer)
                stats['nll'] += -log_probs[answer].item()
    report = {task: {'n': s['n'], 'accuracy': s['correct'] / s['n'], 'nll': s['nll'] / s['n']} for task, s in per_task.items()}
    total = sum(s['n'] for s in per_task.values())
    report['_all'] = {'n': total, 'accuracy': sum(s['correct'] for s in per_task.values()) / total,
                      'nll': sum(s['nll'] for s in per_task.values()) / total,
                      'macro_accuracy': statistics.mean(r['accuracy'] for t, r in report.items() if t != '_all')}
    return report


def save(model, tokenizer, output, meta):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output/'jev_modernbert.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'models/modernbert-ja-310m-jev')
    parser.add_argument('--tasks', nargs='+', choices=['jnli', 'jcommonsenseqa', 'jsts', 'jcola', 'moral', 'massive'])
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup', type=float, default=0.06)
    parser.add_argument('--pair-budget', type=int, default=64, help='Pairs per optimizer step')
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--limit', type=int, default=0, help='Use only N training questions (smoke tests)')
    parser.add_argument('--valid-limit', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--base-model', default=MODEL_ID)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--model-id', default='modernbert-ja-310m-jev')
    parser.add_argument('--resume', type=Path, help='Continue from a saved Jev checkpoint')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'{args.output} is not empty; choose a new --output')
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = select_device(args.device)
    # Autocast to bf16 only on CUDA/ROCm; MPS and CPU train in float32 (slow; use a GPU for full runs).
    dtype = getattr(torch, args.dtype) if device.type == 'cuda' else torch.float32
    train_items = build('train', args.tasks, args.seed)
    valid_items = build('valid', args.tasks, args.seed)
    if args.limit:
        train_items = train_items[:args.limit]
    if args.valid_limit:
        valid_items = valid_items[:args.valid_limit]
    print(f'train questions: {len(train_items)}  valid questions: {len(valid_items)}  device: {device} {dtype}', flush=True)
    tokenizer = load_tokenizer(args.base_model, args.revision)
    source = str(args.resume) if args.resume else args.base_model
    kwargs = {} if args.resume else {'revision': args.revision}
    # Full-precision master weights; autocast does the bf16 math.
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1, dtype=torch.float32, **kwargs).to(device)
    model.train()
    steps_per_epoch = math.ceil(sum(len(i['choices']) for i in train_items) / args.pair_budget)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * max(0.0, 1.0 - progress)
    meta = {'format_version': FORMAT_VERSION, 'base_model': args.base_model, 'revision': args.revision,
            'model_id': args.model_id, 'max_length': args.max_length, 'trained': True,
            'tasks': args.tasks or 'all', 'epochs': args.epochs, 'lr': args.lr, 'pair_budget': args.pair_budget,
            'seed': args.seed, 'train_questions': len(train_items), 'total_steps': total_steps}
    step, best, started, history = 0, None, time.perf_counter(), []
    log = open(args.output.with_suffix('.log'), 'a') if args.output.parent.exists() else None
    def record(entry):
        history.append(entry)
        line = json.dumps(entry, ensure_ascii=False)
        print(line, flush=True)
        if log:
            log.write(line + '\n'); log.flush()
    def run_eval(final=False):
        nonlocal best
        report = evaluate(model, tokenizer, valid_items, device, args.max_length)
        model.train()
        record({'step': step, 'eval': report, 'elapsed_s': round(time.perf_counter() - started, 1)})
        score = report['_all']['macro_accuracy']
        if best is None or score > best:
            best = score
            save(model, tokenizer, args.output, {**meta, 'step': step, 'valid': report, 'best': True})
            print(f'saved best (macro acc {score:.4f}) to {args.output}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    epoch = 0
    while step < total_steps:
        epoch += 1
        for batch in batches(train_items, args.pair_budget, rng):
            if step >= total_steps:
                break
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step)
            contexts, candidates, spans = flatten(batch)
            enc = encode_pairs(tokenizer, contexts, candidates, args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                logits = model(**enc).logits[:, 0]
            loss = listwise_loss(logits.float(), spans, [i['weight'] for i in batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                record({'step': step, 'epoch': epoch, 'loss': round(loss.item(), 4), 'lr': lr_at(step),
                        'elapsed_s': round(time.perf_counter() - started, 1)})
            if step % args.eval_every == 0 and step < total_steps:
                run_eval()
    run_eval(final=True)
    (args.output/'training_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=1) + '\n')
    print(f'done: best macro accuracy {best:.4f}; checkpoint {args.output}', flush=True)


if __name__ == '__main__':
    main()
