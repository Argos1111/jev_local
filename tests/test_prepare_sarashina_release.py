"""Runtime release preparation does not select models, publish, or gate untested GPUs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_sarashina_release as release


class RuntimeReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root/'build'
        directory = self.build/'bin'
        directory.mkdir(parents=True)
        names = release.COMMON | {'libggml-cuda.so', 'test-backend-ops'}
        for name in names:
            (directory/name).write_bytes(name.encode())
        (directory/'libllama.so.0').symlink_to('libllama.so')
        self.manifest = {
            'identity': release.source_identity(), 'backend': 'cuda', 'platform': {},
            'configure': ['cmake', '-S', str(release.ROOT/'source')],
            'sarashina_preprocessing': ['sarashina_official_v1'],
            'runtime_env': {'LD_LIBRARY_PATH': '/private/local/only'},
            'binaries': {n: release.sha256(directory/n) for n in names},
            'fa160_tests': {'status': 'failed'},
        }
        (self.build/'jev-build.json').write_text(json.dumps(self.manifest))

    def test_manifest_integrity_includes_common_library(self):
        files = release.binary_files(self.build, self.manifest)
        self.assertIn('libllama-common.so', [p.name for p in files])
        self.assertIn('libllama.so.0', [p.name for p in files])
        (self.build/'bin/libllama-common.so').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'build artifact'):
            release.binary_files(self.build, self.manifest)

    def test_external_symlink_and_unexpected_library_rejected(self):
        extra = self.build/'bin/libcuda.so'
        extra.write_bytes(b'driver is not part of this distribution')
        with self.assertRaisesRegex(ValueError, 'Unexpected library'):
            release.binary_files(self.build, self.manifest)
        extra.unlink()
        original = self.build/'bin/libggml-cuda.so'
        other = self.root/'outside.so'
        original.rename(other)
        original.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'External'):
            release.binary_files(self.build, self.manifest)

    def test_elf_absolute_rpath_rejected(self):
        header = 'Machine: Advanced Micro Devices X86-64'
        dynamic = '0x1 (NEEDED) Shared library: [libc.so.6]\n0x1d (RUNPATH) Library runpath: [$ORIGIN]'
        versions = 'Name: GLIBC_2.38\nName: GLIBC_2.9\nName: GLIBCXX_3.4.32'
        with patch.object(release.subprocess, 'check_output', side_effect=[header, dynamic, versions]):
            info = release.inspect_elf(Path('fixture'))
        self.assertEqual(info['abi']['GLIBC'], 'GLIBC_2.38')
        self.assertEqual(info['needed'], ['libc.so.6'])
        with patch.object(release.subprocess, 'check_output', side_effect=[header, dynamic.replace('$ORIGIN', '/private/build')]):
            with self.assertRaisesRegex(ValueError, 'CMAKE_INSTALL_RPATH'):
                release.inspect_elf(Path('fixture'))

    def test_prepare_unverified_build_filters_environment_and_excludes_models(self):
        (self.build/'bin/do-not-publish.gguf').write_bytes(b'not a runtime')
        output = self.root/'out'
        with patch.object(release, 'inspect_elf', return_value={'needed': ['libc.so.6'], 'rpath': '$ORIGIN', 'abi': {}}), \
             patch.object(release, 'upstream_notices', return_value={'LICENSE.fixture': 'fixture'}), \
             patch.object(release, 'toolchain_notices', return_value={}):
            archive = release.prepare(self.build, output, 'fixture', self.root/'unused')
        self.assertTrue(archive.is_file())
        public = json.loads((output/'fixture/jev-build.json').read_text())
        self.assertEqual(public['runtime_env'], {})
        self.assertEqual(public['fa160_tests']['status'], 'not_run')
        self.assertNotIn('/private/local', json.dumps(public))
        self.assertNotIn(str(release.ROOT), json.dumps(public))
        self.assertFalse((output/'fixture/bin/do-not-publish.gguf').exists())
        self.assertEqual(release.sha256(output/'fixture/bin/libllama.so.0'), release.sha256(output/'fixture/bin/libllama.so'))
        with self.assertRaises(FileExistsError):
            release.prepare(self.build, output, 'fixture', self.root/'unused')
        second = output/'second.tar.gz'
        release.write_archive(output/'fixture', second)
        self.assertEqual(release.sha256(archive), release.sha256(second))

    def test_failed_verification_does_not_block_packaging_but_mismatched_hashes_do(self):
        report = self.root/'verification.json'
        report.write_text(json.dumps({'input_binaries_sha256': self.manifest['binaries'],
                                      'fa160_tests': {'status': 'failed', 'total': 0}}))
        with patch.object(release, 'inspect_elf', return_value={'needed': [], 'rpath': '$ORIGIN', 'abi': {}}), \
             patch.object(release, 'upstream_notices', return_value={}), \
             patch.object(release, 'toolchain_notices', return_value={}):
            release.prepare(self.build, self.root/'out', 'fixture', self.root/'unused', report)
            public = json.loads((self.root/'out/fixture/jev-build.json').read_text())
            self.assertEqual(public['fa160_tests']['status'], 'failed')
            report.write_text('{"input_binaries_sha256": {}}')
            with self.assertRaisesRegex(ValueError, 'different build'):
                release.prepare(self.build, self.root/'out2', 'fixture', self.root/'unused', report)
            self.assertFalse((self.root/'out2').exists())

    def test_wrong_source_identity_does_not_create_output(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['identity'] = {}
        (self.build/'jev-build.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Source/patch identity'):
            release.prepare(self.build, self.root/'out', 'fixture', self.root/'unused')
        self.assertFalse((self.root/'out').exists())


if __name__ == '__main__':
    unittest.main()
