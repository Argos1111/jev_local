#!/usr/bin/env python3
"""Local TypeSafe-compatible HTTP service. Inference stays on llama-server."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import urllib.error
import urllib.request
from systemone import SystemOne, ValidationError
from state_cache import DEFAULT_CACHE_DIR

MAX_BODY = 24 * 1024 * 1024

def parse_json(raw):
    def reject_constant(s): raise ValueError(f'Invalid JSON constant: {s}')
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result: raise ValueError(f'Duplicate JSON key: {k}')
            result[k] = v
        return result
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique)


def make_handler(engine, api_key, max_requests=8):
    gate = threading.BoundedSemaphore(max_requests)
    class Handler(BaseHTTPRequestHandler):
        server_version = 'JevLocal/1.0'

        def send_json(self, status, value, extra_headers=None):
            body = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Jev-Local-Backend',engine.model_id)
            self.send_header('X-Jev-Local-Confidence','one_minus_normalized_entropy')
            for name,value in (extra_headers or {}).items(): self.send_header(name,str(value))
            if status == 529: self.send_header('Retry-After','1')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if not hmac.compare_digest(self.headers.get('Authorization','').encode(), ('Bearer '+api_key).encode()):
                self.send_json(401, {'detail':'Missing or invalid API key'})
                return False
            return True

        def do_GET(self):
            path = self.path.split('?',1)[0]
            if path == '/health':
                self.send_json(200,{'status':'ok','model':engine.model_id})
            elif path == '/v1/models':
                if self.authorized(): self.send_json(200,engine.models())
            else: self.send_json(404,{'detail':'Not found'})

        def do_POST(self):
            if self.path.split('?',1)[0] != '/v1/systemone':
                self.send_json(404,{'detail':'Not found'}); return
            if not self.authorized(): return
            if self.headers.get_content_type() != 'application/json':
                self.send_json(415,{'detail':'Content-Type must be application/json'}); return
            if self.headers.get('Transfer-Encoding'):
                self.send_json(400,{'detail':'Chunked requests are not supported'}); return
            try:
                length = int(self.headers.get('Content-Length','-1'))
                if not 0 <= length <= MAX_BODY:
                    self.send_json(413 if length > MAX_BODY else 411,{'detail':'Expected Content-Length within 24 MiB'});return
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                if len(raw) != length: raise ValueError('Incomplete request body')
                payload = parse_json(raw)
            except (ValueError, UnicodeError, RecursionError):
                self.send_json(422,{'detail':[{'loc':['body'],'msg':'Invalid JSON body','type':'json_invalid'}]});return
            except (TimeoutError, socket.timeout):
                self.send_json(408,{'detail':'Request body timeout'});return
            if not gate.acquire(blocking=False):
                self.send_json(529,{'detail':'Local inference queue is full'});return
            try:
                diagnostics = {}
                result = engine.evaluate(payload,diagnostics=diagnostics)
                self.send_json(200,result,{
                    'X-Jev-Local-State-Cache':diagnostics['mode'],
                    'X-Jev-Local-Prefix-Tokens':diagnostics.get('common_prefix_tokens',0),
                    'X-Jev-Local-Cached-Tokens':diagnostics.get('cached_tokens',0),
                    'X-Jev-Local-Processed-Tokens':diagnostics.get('processed_tokens',0),
                })
            except ValidationError as exc: self.send_json(422,{'detail':exc.detail})
            except urllib.error.HTTPError as exc:
                if exc.code in (400,413,422):
                    self.send_json(422,{'detail':'Request exceeds backend limits or context capacity'})
                else: self.send_json(502,{'detail':'Inference backend returned an error'})
            except (TimeoutError, socket.timeout): self.send_json(504,{'detail':'Inference backend timed out'})
            except urllib.error.URLError: self.send_json(502,{'detail':'Inference backend unavailable'})
            except RuntimeError as exc: self.send_json(502,{'detail':str(exc)})
            except Exception:
                self.log_error('Unexpected inference failure')
                self.send_json(500,{'detail':'Internal adapter error'})
            finally: gate.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--backend-url',default='http://127.0.0.1:8097')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--state-cache',choices=['auto','shared','off'],default='auto')
    parser.add_argument('--cache-min-tokens',type=int,default=256)
    parser.add_argument('--slot-cache-dir',type=Path,default=DEFAULT_CACHE_DIR)
    args=parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    if args.cache_min_tokens < 0: parser.error('--cache-min-tokens must be non-negative')
    key=os.environ.get('JEV_API_KEY','local-dev')
    if not key: parser.error('JEV_API_KEY must not be empty')
    try:
        with urllib.request.urlopen(args.backend_url.rstrip('/')+'/props',timeout=5) as response:
            props=json.load(response)
        model_id=Path(props['model_path']).stem.lower()
    except (OSError,KeyError,ValueError) as exc:
        parser.exit(1,f'Cannot read backend model. Start ./run_server.sh first. {exc}\n')
    slots = min(args.workers, props.get('total_slots',args.workers))
    engine=SystemOne(args.backend_url,model_id,slots,state_cache=args.state_cache,cache_dir=args.slot_cache_dir,cache_min_tokens=args.cache_min_tokens)
    server=ThreadingHTTPServer((args.host,args.port),make_handler(engine,key))
    print(f'TypeSafe-compatible API: http://{args.host}:{args.port}/v1/systemone\nBackend: {model_id} / state-cache: {args.state_cache}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        engine.close()

if __name__=='__main__': main()
