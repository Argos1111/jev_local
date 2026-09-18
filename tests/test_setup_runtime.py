"""Check that setup never accepts corrupted files as downloaded dependencies."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.setup_runtime import download


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
