"""GPU selection, device validation, and fallback must not silently imply GPU use."""
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from scripts import setup_runtime as setup


class RuntimeSelectionTests(unittest.TestCase):
    def test_platform_and_gpu_matrix(self):
        cases = [
            ('Linux', 'x86_64', set(), ['linux-x86_64-cpu']),
            ('Linux', 'x86_64', {'0x10de'}, ['linux-x86_64-cuda', 'linux-x86_64-cpu']),
            ('Linux', 'x86_64', {'0x1002'}, ['linux-x86_64-rocm', 'linux-x86_64-cpu']),
            ('Linux', 'x86_64', {'0x10de', '0x1002'}, ['linux-x86_64-cuda', 'linux-x86_64-rocm', 'linux-x86_64-cpu']),
            ('Linux', 'aarch64', set(), ['linux-aarch64-cpu']),
            ('Darwin', 'arm64', set(), ['darwin-arm64-metal', 'darwin-arm64-cpu']),
            ('Darwin', 'x86_64', set(), ['darwin-x86_64-cpu']),
        ]
        for system, machine, vendors, expected in cases:
            with self.subTest(system=system, machine=machine, vendors=vendors):
                self.assertEqual(setup.candidates(system, machine, 'auto', vendors), expected)

    def test_explicit_cpu_and_unsupported_backend(self):
        self.assertEqual(setup.candidates('Linux', 'x86_64', 'cpu', {'0x10de'}), ['linux-x86_64-cpu'])
        with self.assertRaisesRegex(RuntimeError, 'no pinned build'):
            setup.candidates('Darwin', 'arm64', 'cuda', set())
        with self.assertRaisesRegex(RuntimeError, 'WSL2'):
            setup.candidates('Windows', 'AMD64', 'auto', set())

    def test_gpu_validation_requires_device_not_just_exit_zero(self):
        with patch.object(setup, 'runtime_env', return_value={}), patch.object(setup.subprocess, 'run') as run:
            for backend, device in [('cuda', 'CUDA0'), ('rocm', 'ROCm0'), ('rocm', 'HIP0'), ('metal', 'MTL0'), ('metal', 'Metal0')]:
                run.return_value = subprocess.CompletedProcess([], 0, f'Available devices:\n  {device}: GPU (free memory)', '')
                setup.validate_runtime(Path('/server'), Path('/runtime'), backend)
            run.return_value = subprocess.CompletedProcess([], 0, 'Available devices:\n', 'Failed to load CUDA')
            with self.assertRaises(setup.RuntimeUnavailable):
                setup.validate_runtime(Path('/server'), Path('/runtime'), 'cuda')

    def test_auto_falls_back_but_explicit_gpu_fails(self):
        options = ['linux-x86_64-cuda', 'linux-x86_64-cpu']
        with patch.object(setup, 'install_runtime', return_value=(setup.ROOT/'server', setup.ROOT/'runtime')) as install, patch.object(setup, 'validate_runtime', side_effect=[setup.RuntimeUnavailable('driver missing'), None]):
            result = setup.select_runtime({}, options, automatic=True)
            self.assertEqual(result['variant'], options[1])
            self.assertEqual(result['gpu_layers'], '0')
            self.assertEqual(install.call_count, 2)
        with patch.object(setup, 'install_runtime', return_value=(setup.ROOT/'server', setup.ROOT/'runtime')), patch.object(setup, 'validate_runtime', side_effect=setup.RuntimeUnavailable('driver missing')):
            with self.assertRaises(setup.RuntimeUnavailable):
                setup.select_runtime({}, options[:1], automatic=False)

    def test_gpu_selection_enables_offload(self):
        with patch.object(setup, 'install_runtime', return_value=(setup.ROOT/'server', setup.ROOT/'runtime')), patch.object(setup, 'validate_runtime'):
            result = setup.select_runtime({}, ['darwin-arm64-metal'], automatic=True)
            self.assertEqual(result['gpu_layers'], 'all')

    def test_download_failure_is_not_hidden_by_fallback(self):
        with patch.object(setup, 'install_runtime', side_effect=RuntimeError('Checksum mismatch')):
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                setup.select_runtime({}, ['linux-x86_64-cuda', 'linux-x86_64-cpu'], automatic=True)
