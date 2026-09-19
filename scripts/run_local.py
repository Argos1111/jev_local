#!/usr/bin/env python3
"""Start both local services and stop them together with Ctrl-C."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(url, process, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited with status {process.returncode}: {url}')
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as response:
                if json.load(response).get('status') == 'ok':
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f'Service did not become ready within {timeout}s: {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['text', 'vision'], help='Override the model chosen at setup')
    parser.add_argument('--port', type=int, default=8080, help='Public local API port')
    parser.add_argument('--backend-port', type=int, default=int(os.environ.get('PORT', '8097')))
    parser.add_argument('--state-cache', choices=['auto', 'shared', 'off'], default='auto')
    args = parser.parse_args()
    if args.port == args.backend_port or not all(1 <= p <= 65535 for p in (args.port, args.backend_port)):
        parser.error('API and backend ports must be distinct and in 1..65535')
    for port in (args.port, args.backend_port):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'Port {port} is unavailable; stop its service or choose another port') from exc
    cache = Path(os.environ.get('SLOT_CACHE_DIR', str(ROOT/'.cache/slots'))).resolve()
    backend_url = f'http://127.0.0.1:{args.backend_port}'
    api_url = f'http://127.0.0.1:{args.port}'
    log_path = ROOT/'.cache/llama-server.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    children = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with log_path.open('w') as log:
            env = dict(os.environ, PORT=str(args.backend_port), SLOT_CACHE_DIR=str(cache))
            backend_command = [str(ROOT/'run_server.sh')]
            if args.model:
                backend_command.extend(['--model', args.model])
            children.append(subprocess.Popen(backend_command, env=env, stdout=log, stderr=log))
            print(f'Loading model. Backend log: {log_path}', flush=True)
            wait_ready(backend_url, children[0])
            children.append(subprocess.Popen([
                sys.executable, str(ROOT/'api_server.py'), '--backend-url', backend_url,
                '--port', str(args.port), '--slot-cache-dir', str(cache), '--state-cache', args.state_cache,
            ]))
            wait_ready(api_url, children[1])
            print(f'Ready: {api_url}\nRun: python3 systemone_client.py --url {api_url}\nCtrl-C stops both services.', flush=True)
            while True:
                for child in children:
                    if child.poll() is not None:
                        raise RuntimeError(f'Service exited with status {child.returncode}; see {log_path}')
                time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nStopping services.', flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as exc:
        sys.exit(f'Cannot start Jev Local: {exc}\nRun ./setup.sh first; see .cache/llama-server.log and docs/SETUP.md.')
