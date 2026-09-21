"""Projector publication preparation stays local and includes only reviewed files."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_sarashina_upload as preparation


class ProjectorBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mmproj = self.root/'private-local-name.gguf'
        self.mmproj.write_bytes(b'GGUF test fixture')
        self.digest = preparation.sha256(self.mmproj)
        self.output = self.root/'upload'
        source = json.loads((preparation.ROOT/'native/sarashina-reference.json').read_text())
        self.manifest = {
            'source_repository': source['repository'], 'source_revision': source['revision'],
            'checkpoint_file': source['checkpoint_file'],
            'config_sha256': source['files']['config.json']['sha256'],
            'checkpoint_sha256': source['files'][source['checkpoint_file']]['sha256'],
            'preprocessor_sha256': source['files']['preprocessor_config.json']['sha256'],
            'preprocessing': 'sarashina_official_v1',
            'output_sha256': self.digest, 'tensors': 442, 'projector_type': 'sarashina2vl', 'outtype': 'f16',
        }
        self.manifest_path = self.mmproj.with_suffix('.gguf.json')
        self.save_manifest()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(preparation, 'PROJECTOR_SHA256', self.digest))
        stack.enter_context(patch.object(preparation, 'PROJECTOR_BYTES', self.mmproj.stat().st_size))

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_bundle_contains_only_expected_files_and_valid_checksums(self):
        (self.root/'other-model.gguf').write_bytes(b'do not publish')
        (self.root/'private.log').write_text('do not publish')
        self.manifest['local_private_field'] = 'do not publish'
        self.save_manifest()
        files = preparation.prepare(self.mmproj, self.output)
        self.assertEqual(set(files), {
            preparation.PROJECTOR_NAME, preparation.PROJECTOR_NAME+'.json',
            'README.md', 'NOTICE.md', 'sarashina-reference.json', 'SHA256SUMS',
            'LICENSE.sarashina2.2-vision-3b', 'LICENSE.siglip',
        })
        self.assertEqual(set(files), {p.name for p in self.output.iterdir()})
        for line in (self.output/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(preparation.sha256(self.output/name), digest)
        card = (self.output/'README.md').read_text()
        self.assertTrue(card.startswith('---\nlicense: mit\n'))
        self.assertIn('[SHA256SUMS](SHA256SUMS)', card)
        self.assertIn(self.digest, (self.output/'NOTICE.md').read_text())
        self.assertIn('sarashina2vl', card)
        self.assertNotIn('$source_', card)
        self.assertNotIn('$filename', card)
        copied = self.output/preparation.PROJECTOR_NAME
        self.assertEqual(copied.read_bytes(), self.mmproj.read_bytes())
        self.assertFalse(copied.is_symlink())
        self.assertNotIn('local_private_field', (self.output/(preparation.PROJECTOR_NAME+'.json')).read_text())
        self.assertEqual(self.mmproj.read_bytes(), b'GGUF test fixture')

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output/'local edits'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            preparation.prepare(self.mmproj, self.output)
        self.assertEqual(marker.read_text(), 'keep')

    def test_wrong_artifact_does_not_create_output(self):
        self.mmproj.write_bytes(b'corrupted fixture')
        with self.assertRaisesRegex(ValueError, 'size/SHA256'):
            preparation.prepare(self.mmproj, self.output)
        self.assertFalse(self.output.exists())

    def test_manifest_must_match_projector_and_pinned_source(self):
        for key, wrong in [('output_sha256', '0'*64), ('config_sha256', '0'*64),
                           ('checkpoint_sha256', '0'*64), ('preprocessor_sha256', '0'*64),
                           ('source_repository', 'test/clone'), ('source_revision', '0'*40),
                           ('preprocessing', 'legacy_pillow'), ('tensors', 440),
                           ('projector_type', 'qwen2vl'), ('outtype', 'f32')]:
            with self.subTest(key=key):
                manifest = dict(self.manifest, **{key: wrong})
                self.manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                    preparation.prepare(self.mmproj, self.output)
                self.assertFalse(self.output.exists())

    def test_legacy_manifest_is_not_mislabeled_official(self):
        for key in ('source_repository', 'source_revision', 'preprocessing', 'preprocessor_sha256'):
            self.manifest.pop(key)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            preparation.prepare(self.mmproj, self.output)
        self.assertFalse(self.output.exists())

    def test_explicit_f16_manifest_supported(self):
        self.manifest['outtype'] = 'f16'
        self.save_manifest()
        preparation.prepare(self.mmproj, self.output)
        copied = json.loads((self.output/(preparation.PROJECTOR_NAME+'.json')).read_text())
        self.assertEqual(copied['outtype'], 'f16')

    def test_modified_license_is_rejected_before_copy(self):
        wrong = dict(preparation.LICENSE_HASHES)
        wrong['LICENSE.siglip'] = '0'*64
        with patch.object(preparation, 'LICENSE_HASHES', wrong), self.assertRaisesRegex(ValueError, 'License differs'):
            preparation.prepare(self.mmproj, self.output)
        self.assertFalse(self.output.exists())

    def test_copy_failure_leaves_no_success_marker(self):
        def broken_copy(source, destination):
            Path(destination).write_bytes(b'incomplete')
            raise OSError('disk full')
        with patch.object(preparation.shutil, 'copyfile', side_effect=broken_copy), self.assertRaises(OSError):
            preparation.prepare(self.mmproj, self.output)
        self.assertFalse((self.output/'SHA256SUMS').exists())


if __name__ == '__main__':
    unittest.main()
