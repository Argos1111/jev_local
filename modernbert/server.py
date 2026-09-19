#!/usr/bin/env python3
"""Serve the TypeSafe-compatible API from an in-process ModernBERT cross-encoder."""
import argparse
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_server import make_handler  # noqa: E402
from modernbert.engine import CrossEncoder  # noqa: E402
from modernbert.systemone import ModernBertSystemOne  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT/'models/modernbert-ja-310m-jev'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:1 ...')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--allow-untrained', action='store_true', help='Serve the base model with a random head (smoke tests only)')
    parser.add_argument('--max-requests', type=int, default=8)
    args = parser.parse_args()
    key = os.environ.get('JEV_API_KEY', 'local-dev')
    if not key:
        parser.error('JEV_API_KEY must not be empty')
    try:
        encoder = CrossEncoder(None if args.allow_untrained else args.checkpoint, device=args.device,
                               dtype=args.dtype, allow_untrained=args.allow_untrained)
    except RuntimeError as exc:
        parser.exit(1, f'{exc}\nTrain first: .venv-modernbert/bin/python -m modernbert.train\n')
    engine = ModernBertSystemOne(encoder)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, key, args.max_requests))
    print(f'TypeSafe-compatible API: http://{args.host}:{args.port}/v1/systemone\n'
          f'Backend: {engine.model_id} on {encoder.device} ({encoder.dtype}); text only', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == '__main__':
    main()
