"""The persisted setup choice must also control the actual server launch."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import start_backend


class BackendLaunchTests(unittest.TestCase):
    def test_sarashina_is_text_only_unless_projector_is_explicit(self):
        lock = json.loads((start_backend.ROOT/'scripts/runtime.json').read_text())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'scripts/runtime.json').write_text(json.dumps(lock))
            (root/'models').mkdir()
            # Even when LFM's projector exists, never attach it to Sarashina.
            (root/'models'/lock['mmproj']['filename']).touch()
            server = root/'server'
            server.touch()
            server.chmod(0o755)
            selection = root/'.cache/runtime/selected.json'
            selection.parent.mkdir(parents=True)
            selection.write_text(json.dumps({'server': 'server', 'directory': '.', 'gpu_layers': 'all'}))
            custom = root/'custom-mmproj.gguf'
            custom.touch()
            for profile, key in [('sarashina', 'sarashina_model'), ('sarashina-q8', 'sarashina_q8_model')]:
                model = root/'models'/lock[key]['filename']
                model.touch()
                with self.subTest(profile=profile), patch.object(start_backend, 'ROOT', root), patch.object(start_backend.sys, 'argv', ['start_backend.py', '--model', profile]), patch.object(start_backend.os, 'execve') as execute:
                    start_backend.main()
                    args = execute.call_args.args[1]
                    self.assertEqual(args[args.index('-m')+1], str(model))
                    self.assertNotIn('--mmproj', args)
                    self.assertNotIn('--model', args)
                    self.assertEqual(args[args.index('-ngl')+1], 'all')
                    with patch.dict(os.environ, {'LFM_MMPROJ': str(custom)}):
                        start_backend.main()
                    args = execute.call_args.args[1]
                    self.assertEqual(args[args.index('--mmproj')+1], str(custom))
                    model.unlink()
                    with self.assertRaisesRegex(SystemExit, f'--model {profile} --model-only'):
                        start_backend.main()

    def test_selected_gpu_enables_offload_and_explicit_settings_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.cache/runtime').mkdir(parents=True)
            (root/'models').mkdir()
            (root/'scripts').mkdir()
            lock = json.loads((start_backend.ROOT/'scripts/runtime.json').read_text())
            (root/'scripts/runtime.json').write_text(json.dumps(lock))
            (root/'models'/lock['model']['filename']).touch()
            projector = root/'models'/lock['mmproj']['filename']
            projector.touch()
            server = root/'server'
            server.touch()
            server.chmod(0o755)
            (root/'.cache/runtime/selected.json').write_text(json.dumps({
                'variant': 'linux-x86_64-cuda', 'server': 'server', 'directory': '.', 'gpu_layers': 'all'}))
            with patch.object(start_backend, 'ROOT', root), patch.object(start_backend.sys, 'argv', ['start_backend.py', '--model', 'vision']), patch.object(start_backend.os, 'execve') as execute:
                with patch.dict(os.environ, {}, clear=True):
                    start_backend.main()
                binary, args, env = execute.call_args.args
                self.assertEqual(binary, server)
                self.assertEqual(args[args.index('-ngl')+1], 'all')
                self.assertEqual(args[args.index('--mmproj')+1], str(projector))
                self.assertNotIn('--no-mmproj-offload', args)
                with patch.dict(os.environ, {'GPU_LAYERS': '0', 'GPU_DEVICE': 'CUDA1', 'PORT': '19097'}, clear=True):
                    start_backend.main()
                _, args, _ = execute.call_args.args
                self.assertEqual(args[args.index('-ngl')+1], '0')
                self.assertIn('--no-mmproj-offload', args)
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
                with patch.dict(os.environ, {'LFM_MODEL': str(root/'models'/lock['model']['filename'])}, clear=True):
                    start_backend.main()
                self.assertNotIn('--mmproj', execute.call_args.args[1])
                text_model = root/'models'/lock['text_model']['filename']
                text_model.touch()
                projector.unlink()
                with patch.dict(os.environ, {}, clear=True), patch.object(start_backend.sys, 'argv', ['start_backend.py', '--model', 'text']):
                    start_backend.main()
                args = execute.call_args.args[1]
                self.assertEqual(args[args.index('-m')+1], str(text_model))
                self.assertNotIn('--mmproj', args)
                self.assertNotIn('--model', args)
                with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(SystemExit, 'Projector not found'):
                    start_backend.main()
