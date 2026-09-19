#!/usr/bin/env python3
"""Upload a trained Jev ModernBERT checkpoint to the Hugging Face Hub.

Requires a write token in HF_TOKEN (never stored by this script). Uploads the
checkpoint directory plus the model card, then verifies that the published
metadata resolves through the same loader the server uses.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modernbert.engine import HUB_CHECKPOINT, META_FILE, resolve_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'models/modernbert-ja-310m-jev')
    parser.add_argument('--repo', default=HUB_CHECKPOINT)
    parser.add_argument('--card', type=Path, default=ROOT/'modernbert/MODEL_CARD.md')
    parser.add_argument('--message', default='Upload fine-tuned Jev cross-encoder')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('HF_TOKEN'):
        parser.exit(1, 'Set HF_TOKEN to a write token (https://huggingface.co/settings/tokens).\n')
    meta = json.loads((args.checkpoint/META_FILE).read_text())
    if not meta.get('trained'):
        parser.exit(1, f'{args.checkpoint} is not a trained checkpoint.\n')
    files = sorted(p.name for p in args.checkpoint.iterdir() if p.is_file())
    print(f'repo: {args.repo}\ncheckpoint: {args.checkpoint}\nfiles: {files}\ncard: {args.card}', flush=True)
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()['name']
    if not args.repo.startswith(who + '/') and who not in [o['name'] for o in api.whoami().get('orgs', [])]:
        parser.exit(1, f'Token belongs to {who!r}, which cannot write to {args.repo}.\n')
    api.create_repo(args.repo, repo_type='model', exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(args.checkpoint), commit_message=args.message,
                      ignore_patterns=['*.log', 'training_history.json'])
    api.upload_file(repo_id=args.repo, path_or_fileobj=str(args.card), path_in_repo='README.md',
                    commit_message='Add model card')
    source, kwargs, published = resolve_checkpoint(args.repo)
    assert published['format_version'] == meta['format_version'], published
    print(f'published: https://huggingface.co/{args.repo}  (format {published["format_version"]}, step {published.get("step")})')


if __name__ == '__main__':
    main()
