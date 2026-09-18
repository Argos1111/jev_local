"""The persisted setup choice must also control the actual server launch."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import start_backend


class BackendLaunchTests(unittest.TestCase):
    def test_selected_gpu_enables_offload_and_explicit_settings_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.cache/runtime').mkdir(parents=True)
            (root/'models').mkdir()
            (root/'models/LFM2.5-1.2B-Instruct-Q8_0.gguf').touch()
            server = root/'server'
            server.touch()
            server.chmod(0o755)
            (root/'.cache/runtime/selected.json').write_text(json.dumps({
                'variant': 'linux-x86_64-cuda', 'server': 'server', 'directory': '.', 'gpu_layers': 'all'}))
            with patch.object(start_backend, 'ROOT', root), patch.object(start_backend.sys, 'argv', ['start_backend.py']), patch.object(start_backend.os, 'execve') as execute:
                with patch.dict(os.environ, {}, clear=True):
                    start_backend.main()
                binary, args, env = execute.call_args.args
                self.assertEqual(binary, server)
                self.assertEqual(args[args.index('-ngl')+1], 'all')
                with patch.dict(os.environ, {'GPU_LAYERS': '0', 'GPU_DEVICE': 'CUDA1', 'PORT': '19097'}, clear=True):
                    start_backend.main()
                _, args, _ = execute.call_args.args
                self.assertEqual(args[args.index('-ngl')+1], '0')
                self.assertEqual(args[args.index('--device')+1], 'CUDA1')
                self.assertEqual(args[args.index('--port')+1], '19097')
                manual = root/'manual-server'
                manual.touch()
                manual.chmod(0o755)
                with patch.dict(os.environ, {'LLAMA_SERVER': str(manual)}, clear=True):
                    start_backend.main()
                binary, args, _ = execute.call_args.args
                self.assertEqual(binary, manual)
                self.assertEqual(args[args.index('-ngl')+1], '0')
