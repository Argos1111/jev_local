"""Check that setup never accepts corrupted files as downloaded dependencies."""
import hashlib
import json
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from scripts import setup_runtime
from scripts.setup_runtime import download, model_profile, model_specs


class DownloadTests(unittest.TestCase):
    def test_verified_download_and_offline_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            target = Path(directory)/'downloads/model'
            source.write_bytes(b'fixture model')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            download(source.as_uri(), target, digest)
            source.unlink()
            download(source.as_uri(), target, digest)
            self.assertEqual(target.read_bytes(), b'fixture model')

    def test_bad_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            target = Path(directory)/'model'
            source.write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                download(source.as_uri(), target, '0'*64)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.part').exists())

    def test_existing_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'model'
            target.write_bytes(b'local data')
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                download('https://invalid.example/model', target, '0'*64)
            self.assertEqual(target.read_bytes(), b'local data')


class ModelSelectionTests(unittest.TestCase):
    def test_default_saved_environment_and_explicit_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            self.assertEqual(model_profile(root), 'text')
            saved = root/'.cache/runtime/model.json'
            saved.parent.mkdir(parents=True)
            saved.write_text(json.dumps({'profile': 'vision'}))
            self.assertEqual(model_profile(root), 'vision')
            with patch.dict(os.environ, {'LFM_PROFILE': 'text'}):
                self.assertEqual(model_profile(root), 'text')
                self.assertEqual(model_profile(root, 'vision'), 'vision')
            with patch.dict(os.environ, {'LFM_PROFILE': 'unknown'}):
                with self.assertRaisesRegex(RuntimeError, 'Unknown model profile'):
                    model_profile(root)

    def test_text_downloads_no_projector(self):
        lock = {'text_model': 'text', 'model': 'vl', 'mmproj': 'projector'}
        self.assertEqual(model_specs(lock, 'text'), ['text'])
        self.assertEqual(model_specs(lock, 'vision'), ['vl', 'projector'])

    def test_model_only_setup_saves_choice_without_changing_runtime(self):
        lock = json.loads((setup_runtime.ROOT/'scripts/runtime.json').read_text())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'scripts/runtime.json').write_text(json.dumps(lock))
            runtime = root/'.cache/runtime/selected.json'
            runtime.parent.mkdir(parents=True)
            runtime.write_text('{"server":"existing"}')
            for profile, count in [('text', 1), ('vision', 2)]:
                with self.subTest(profile=profile), patch.object(setup_runtime, 'ROOT', root), patch.object(setup_runtime.sys, 'argv', ['setup', '--model', profile, '--model-only']), patch.object(setup_runtime, 'download') as download_mock, patch.object(setup_runtime, 'select_runtime') as select:
                    setup_runtime.main()
                    self.assertEqual(download_mock.call_count, count)
                    select.assert_not_called()
                    self.assertEqual(model_profile(root), profile)
                    self.assertEqual(runtime.read_text(), '{"server":"existing"}')
