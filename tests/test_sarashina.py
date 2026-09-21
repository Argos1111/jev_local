"""Guard the opt-in experiment without downloading weights or loading a GPU."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_sarashina as builder
from scripts import convert_sarashina_mmproj as converter
from scripts import fetch_sarashina_reference as fetcher
from scripts import run_sarashina as launcher
from scripts import setup_runtime
from tools.benchmark_sarashina_fa import runtime_observations


def config():
    return {'model_type': 'sarashina2_vision', 'text_config': {'hidden_size': 2560},
            'image_token_index': 14, 'start_image_token_index': 102397, 'end_image_token_index': 102398,
            'vision_config': {'embed_dim': 1152, 'hidden_size': 2560, 'num_heads': 16, 'depth': 27,
                'patch_size': 14, 'spatial_merge_size': 2, 'temporal_patch_size': 2,
                'hidden_act': 'gelu_pytorch_tanh', 'in_channels': 3, 'mlp_ratio': 3.7362}}


class ConverterTests(unittest.TestCase):
    def test_strict_dimensions_activations_and_boundaries(self):
        converter.validate_config(config())
        for section, key, value in [('vision_config', 'hidden_act', 'quick_gelu'),
                                     ('vision_config', 'mlp_ratio', 2.222),
                                     ('text_config', 'hidden_size', 1152),
                                     (None, 'start_image_token_index', 10)]:
            wrong = copy.deepcopy(config())
            (wrong[section] if section else wrong)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                converter.validate_config(wrong)

    def test_official_preprocessor_not_reconstructed_qwen_defaults(self):
        path = setup_runtime.ROOT/'native/sarashina-reference.json'
        spec = json.loads(path.read_text())
        self.assertIn('preprocessor_config.json', spec['conversion_files'])
        value = {'image_processor_type': 'Sarashina2VisionImageProcessor',
                 'do_resize': True, 'do_rescale': True, 'do_normalize': True, 'do_convert_rgb': True,
                 'image_mean': [0.5]*3, 'image_std': [0.5]*3, 'rescale_factor': 1/255,
                 'patch_size': 14, 'temporal_patch_size': 2, 'merge_size': 2,
                 'min_pixels': 3136, 'max_pixels': 1016064}
        converter.validate_preprocessor(value)
        for key, wrong in [('max_pixels', 1003520), ('do_normalize', False), ('image_std', [1]*3)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                converter.validate_preprocessor(dict(value, **{key: wrong}))

    def test_source_metadata_requires_pinned_official_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/name for name in ('config.json', 'model.safetensors', 'preprocessor_config.json')]
            for path in paths:
                path.write_bytes(path.name.encode())
            lock = {'repository': 'test/official', 'revision': '0'*40, 'checkpoint_file': 'model.safetensors',
                    'files': {p.name: {'sha256': setup_runtime.sha256(p), 'size': p.stat().st_size} for p in paths}}
            with patch.object(converter, 'reference_lock', return_value=lock):
                result = converter.source_metadata(*paths)
                self.assertEqual(result['source_repository'], 'test/official')
                self.assertEqual(result['preprocessing'], 'sarashina_official_v1')
                self.assertEqual(result['checkpoint_sha256'], lock['files']['model.safetensors']['sha256'])
                paths[1].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'official reference'):
                    converter.source_metadata(*paths)

    def test_all_source_shapes_and_final_norm_are_accounted_for(self):
        layout = converter.tensor_layout()
        self.assertEqual(len(layout), 333)
        self.assertEqual(layout['norm.weight'], ('mm.post_norm.weight', (2560,)))
        self.assertEqual(layout['norm.bias'], ('mm.post_norm.bias', (2560,)))
        self.assertEqual(layout['visual.blocks.26.mlp.fc1.weight'][1], (4304, 1152))
        self.assertEqual(layout['visual.merger.mlp.2.weight'][1], (2560, 4608))
        # QKV weight/bias become 3 tensors each; Conv3D becomes two temporal slices.
        self.assertEqual(sum(dst is not None for dst, _ in layout.values()) + 27*6 + 2, 442)


class FetchTests(unittest.TestCase):
    def test_obsolete_clone_option_does_not_download(self):
        with patch('sys.argv', ['fetch', '--accept-public-clone']), patch.object(fetcher, 'fetch_file') as download, \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                fetcher.main()
            download.assert_not_called()

    def test_fetching_model_never_persists_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'native').mkdir()
            (root/'scripts/runtime.json').write_text((setup_runtime.ROOT/'scripts/runtime.json').read_text())
            names = ('config.json', 'preprocessor_config.json', 'model.safetensors')
            (root/'native/sarashina-reference.json').write_text(json.dumps({
                'notice': 'fixture official', 'repository': 'test/official', 'revision': '0'*40,
                'conversion_files': names,
                'files': {n: {'sha256': '1'*64, 'size': 1} for n in names}}))
            selected = root/'.cache/runtime/selected.json'
            selected.parent.mkdir(parents=True)
            selected.write_text('original selection')
            def download(url, path, digest):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'1')
            with patch('sys.argv', ['fetch', '--model', 'sarashina', '--output', str(root/'reference')]), \
                 patch.object(fetcher, 'ROOT', root), patch.object(fetcher, 'fetch_file') as fetch, \
                 patch.object(fetcher, 'download', side_effect=download), redirect_stdout(io.StringIO()):
                fetcher.main()
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual([call.args[1] for call in fetch.call_args_list], list(names))
            self.assertEqual(selected.read_text(), 'original selection')
            self.assertFalse((selected.parent/'model.json').exists())
            self.assertTrue((root/'models/sarashina2.2-vision-3b.Q4_K_M.gguf').is_file())

    def test_existing_changed_reference_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'config.json'
            path.write_bytes(b'local edit')
            spec = {'files': {path.name: {'size': path.stat().st_size, 'sha256': '0'*64}}}
            with self.assertRaisesRegex(ValueError, 'changed official'):
                fetcher.fetch_file(spec, path.name, root)
            self.assertEqual(path.read_bytes(), b'local edit')

    def test_download_uses_saved_auth_and_pinned_revision(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {'repository': 'test/official', 'revision': '0'*40, 'files': {'config.json': {}}}
            from unittest.mock import Mock
            download = Mock()
            with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(hf_hub_download=download)}), \
                 patch.object(fetcher, 'verify_file') as verify, redirect_stdout(io.StringIO()):
                fetcher.fetch_file(spec, 'config.json', root)
            download.assert_called_once_with(repo_id='test/official', revision='0'*40, filename='config.json',
                                             local_dir=root, token=True)
            verify.assert_called_once_with(root/'config.json', {})

    def test_reference_lock_is_official_revision_and_checksum_pinned(self):
        spec = json.loads((setup_runtime.ROOT/'native/sarashina-reference.json').read_text())
        self.assertEqual(spec['repository'], 'sbintuitions/sarashina2.2-vision-3b')
        self.assertRegex(spec['revision'], r'^[0-9a-f]{40}$')
        self.assertEqual(sum(name.endswith('.safetensors') for name in spec['files']), 1)
        self.assertEqual(spec['checkpoint_file'], 'model.safetensors')
        for name, row in spec['files'].items():
            self.assertEqual(Path(name).name, name)
            self.assertRegex(row['sha256'], r'^[0-9a-f]{64}$')
            self.assertGreater(row['size'], 0)


class SourceBuildTests(unittest.TestCase):
    def test_fresh_source_is_stamped_reused_and_local_changes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root/'native'
            native.mkdir()
            native.joinpath('source.json').write_text(json.dumps({'version': 'fixture', 'url': 'fixture', 'sha256': '0'*64}))
            native.joinpath('vision_embedding_cache.h').write_text('fixture header')
            for name in builder.PATCHES:
                native.joinpath(name).write_text('+++ b/example.txt\n')
            tree = root/'fixture-tree'
            tree.joinpath('tools/mtmd').mkdir(parents=True)
            tree.joinpath('example.txt').write_text('fixture source')
            instances = tree/'ggml/src/ggml-cuda/template-instances'
            instances.mkdir(parents=True)
            instances.joinpath('fixture.cu').write_text('fixture kernel')
            root.joinpath('.cache').mkdir()
            with tarfile.open(root/'.cache/llama-b11042-source.tar.gz', 'w:gz') as bundle:
                bundle.add(tree, arcname='llama.cpp-fixture')
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'download'), \
                 patch.object(builder.subprocess, 'run') as run:
                source, identity = builder.prepare_source(base=root/'experiment', patch_tool='fixture-patch')
                self.assertEqual(run.call_count, len(builder.PATCHES) + 1)
                self.assertEqual(run.call_args_list[0].args[0][0], 'fixture-patch')
                self.assertEqual(json.loads((source/'jev-source.json').read_text())['identity'], identity)
                run.reset_mock()
                self.assertEqual(builder.prepare_source(base=root/'experiment'), (source, identity))
                run.assert_not_called()
                source.joinpath('example.txt').write_text('local edits')
                with self.assertRaisesRegex(RuntimeError, 'Source/patches changed'):
                    builder.prepare_source(base=root/'experiment')
                self.assertEqual(source.joinpath('example.txt').read_text(), 'local edits')


class BackendBuildTests(unittest.TestCase):
    def test_cuda_and_hip_targets_are_user_selected_not_allowlisted(self):
        cuda = builder.backend_options('cuda', cuda_architectures='89')
        self.assertIn('-DGGML_CUDA=ON', cuda)
        self.assertIn('-DGGML_HIP=OFF', cuda)
        self.assertIn('-DGGML_CUDA_FA=ON', cuda)
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=89', cuda)
        hip = builder.backend_options('hip', hip_architectures='gfx1100;gfx1201')
        self.assertIn('-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201', hip)
        self.assertIn('-DGGML_CUDA=OFF', hip)
        self.assertFalse(any('ARCHITECTURES=' in x for x in builder.backend_options('hip')))
        self.assertFalse(any('ARCHITECTURES=' in x for x in builder.backend_options('cuda')))

    def test_cuda_build_writes_manifest_without_gpu_tests_or_despite_failure(self):
        for check in (False, True):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root/'experiment/build-cuda'
                (folder/'bin').mkdir(parents=True)
                names = ['llama-server', 'libllama-server-impl.so', 'libllama-common.so', 'libmtmd.so', 'libllama.so',
                         'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'test_sarashina_embedding',
                         'libggml-cuda.so', 'test-backend-ops']
                for name in names:
                    (folder/'bin'/name).write_bytes(b'fixture')
                options = ['--base', str(root/'experiment')] if check else []
                with patch('sys.argv', ['build', '--backend', 'cuda', '--cuda-architectures', '89'] + options + (['--test-fa'] if check else [])), \
                     patch.object(builder, 'ROOT', root), patch.object(builder, 'BASE', root/('other-base' if check else 'experiment')), \
                     patch.object(builder.sys, 'platform', 'linux'), patch.object(builder.shutil, 'which', side_effect=lambda x: '/tools/'+x), \
                     patch.object(builder, 'prepare_source', return_value=(root/'source', {})), \
                     patch.object(builder.subprocess, 'run') as run, \
                     patch.object(builder, 'run_fa160_tests', return_value={'status': 'failed', 'passed': 1, 'total': 252, 'log': 'fixture.log'}) as tests, \
                     patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    builder.main()
                manifest = json.loads((folder/'jev-build.json').read_text())
                self.assertEqual(manifest['backend'], 'cuda')
                self.assertIn('libggml-cuda.so', manifest['binaries'])
                self.assertNotIn('libggml-hip.so', manifest['binaries'])
                self.assertEqual(manifest['fa160_tests']['status'], 'failed' if check else 'not_run')
                self.assertEqual(tests.call_count, int(check))
                self.assertIn('test-backend-ops', run.call_args_list[1].args[0])
                self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=89', run.call_args_list[0].args[0])

    def test_runtime_environment_uses_manifest_without_overriding_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            build.joinpath('bin').mkdir()
            build.joinpath('jev-build.json').write_text(json.dumps({'runtime_env': {
                'LD_LIBRARY_PATH': '/recorded/cuda', 'ROCBLAS_TENSILE_LIBPATH': '/recorded/tensile'}}))
            with patch.dict(os.environ, {}, clear=True):
                env = builder.build_environment(build)
            self.assertEqual(env['LD_LIBRARY_PATH'], str((build/'bin').resolve())+':/recorded/cuda')
            with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/explicit'}, clear=True):
                env = builder.build_environment(build)
            self.assertEqual(env['LD_LIBRARY_PATH'], str((build/'bin').resolve())+':/explicit')
            self.assertEqual(env['ROCBLAS_TENSILE_LIBPATH'], '/recorded/tensile')


class KernelVerificationTests(unittest.TestCase):
    def test_full_gpu_run_is_accepted_without_architecture_allowlist(self):
        for device, arch in [('ROCm0', 'gfx1201'), ('ROCm1', 'gfx1100'), ('CUDA0', '8.9'), ('CUDA2', '12.0')]:
            log = f'  Device 0: Some GPU, {arch}\nBackend 1/3: {device}\n  252/252 tests passed\n'
            with self.subTest(device=device):
                self.assertTrue(builder.fa160_verified(log, 0, device))
        log = 'Backend 1/3: CUDA0\n  252/252 tests passed\n'
        for output, status, device in [(log, 1, 'CUDA0'), (log, 0, 'CPU'), (log, 0, 'CUDA1'),
                (log.replace('252/252', '0/0'), 0, 'CUDA0'), (log+'FAIL', 0, 'CUDA0'),
                (log+'NOT_SUPPORTED', 0, 'CUDA0'), (log.replace('252/252', '251/252'), 0, 'CUDA0')]:
            with self.subTest(output=output, status=status, device=device):
                self.assertFalse(builder.fa160_verified(output, status, device))

    def test_pass_counts_on_another_backend_do_not_hide_skipped_gpu(self):
        log = 'Backend 1/2: CUDA0\nSkipping\nBackend 2/2: CPU\n  252/252 tests passed\n'
        self.assertFalse(builder.fa160_verified(log, 0, 'CUDA0'))
        result = builder.fa160_test_result(log, 0, 'CUDA0')
        self.assertEqual(result['total'], 0)

    def test_timeout_and_launch_errors_are_reports_not_build_failures(self):
        for error, status in [(subprocess.TimeoutExpired('test', 1, output=b'partial output'), 'timeout'),
                              (FileNotFoundError('missing test binary'), 'error')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                log = build/'fa.log'
                with patch.object(builder.subprocess, 'run', side_effect=error):
                    report = builder.run_fa160_tests(build, 'CUDA0', log, 1)
                self.assertEqual(report['status'], status)
                self.assertTrue(log.is_file())
                self.assertEqual(report['log_sha256'], setup_runtime.sha256(log))

    def test_benchmark_reports_fa_state_and_splits_without_claiming_gpu_placement(self):
        report = runtime_observations('llama_context: flash_attn = enabled\nsched_reserve: graph splits = 66\n'
                                      'using device CUDA0 (RTX 4090)\n')
        self.assertEqual(report['flash_attn_states'], ['enabled'])
        self.assertEqual(report['graph_splits'], [66])
        self.assertEqual(len(report['device_lines']), 1)
        self.assertIn('does not prove GPU placement', report['note'])


class ExperimentalLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.base = self.root/'experiment'
        (self.root/'scripts').mkdir()
        (self.root/'scripts/runtime.json').write_text((setup_runtime.ROOT/'scripts/runtime.json').read_text())
        (self.root/'.cache/runtime').mkdir(parents=True)
        self.selection = self.root/'.cache/runtime/selected.json'
        self.selection.write_text('{"original":true}')
        self.projector = self.root/'projector.gguf'
        self.projector.write_bytes(b'fixture')
        self.projector.with_suffix('.gguf.json').write_text(json.dumps({
            'projector_type': 'sarashina2vl', 'output_sha256': setup_runtime.sha256(self.projector)}))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(launcher, 'ROOT', self.root))
        self.stack.enter_context(patch.object(launcher, 'BASE', self.base))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        for backend in ('cpu', 'hip', 'stock-rocm', 'cuda'):
            folder = self.base/f'build-{backend}'
            (folder/'bin').mkdir(parents=True)
            names = ['llama-server', 'libmtmd.so', 'libllama.so', 'libllama-server-impl.so',
                     'libggml.so', 'libggml-base.so', 'libggml-cpu.so']
            if backend in builder.GPU_PLUGINS:
                names.append(builder.GPU_PLUGINS[backend])
            for name in names:
                (folder/'bin'/name).write_bytes(name.encode())
            (folder/'jev-build.json').write_text(json.dumps({'backend': backend,
                'fa160_tests': {'passed': 252, 'architecture': 'gfx1201', 'device': 'ROCm0'},
                'sarashina_preprocessing': ['legacy_pillow', 'sarashina_official_v1'],
                'runtime_env': {'LD_LIBRARY_PATH': '/fixture/rocm'},
                'binaries': {name: setup_runtime.sha256(folder/'bin'/name) for name in names}}))

    def test_selection_is_never_changed_and_matching_projector_is_explicit(self):
        env = launcher.launch_environment('hip', 'sarashina-q8', self.projector, False, 'auto', 128)
        self.assertEqual(self.selection.read_text(), '{"original":true}')
        self.assertEqual(env['LFM_MMPROJ'], str(self.projector))
        self.assertTrue(env['LFM_MODEL'].endswith('.Q8_0.gguf'))
        self.assertEqual(env['GPU_LAYERS'], 'all')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/fixture/rocm')
        self.assertTrue(env['JEV_BACKEND_LOG'].startswith(str(self.base)))
        env = launcher.launch_environment('cpu', 'sarashina', Path('/missing'), True, 'off', 0)
        self.assertEqual(env['LFM_MMPROJ'], '')
        self.assertEqual(env['GPU_LAYERS'], '0')
        self.assertNotIn('GPU_DEVICE', env)

    def test_custom_release_build_keeps_default_experiment_and_selections(self):
        release = self.root/'release'
        (self.base/'build-cpu').rename(release)
        env = launcher.launch_environment('cpu', 'sarashina', self.projector, False, 'off', 0, build=release)
        self.assertEqual(env['LLAMA_SERVER'], str(release/'bin/llama-server'))
        self.assertEqual(self.selection.read_text(), '{"original":true}')
        self.assertTrue(env['JEV_BACKEND_LOG'].startswith(str(self.base/'external')))
        self.assertNotEqual(env['SLOT_CACHE_DIR'], str(self.base/'slots'))
        with self.assertRaisesRegex(RuntimeError, 'backend mismatch'):
            launcher.launch_environment('cuda', 'sarashina', self.projector, False, 'off', 0, build=release)

    def test_modified_binary_or_projector_fails_closed(self):
        binary = self.base/'build-hip/bin/libggml-hip.so'
        binary.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'build artifact'):
            launcher.launch_environment('hip', 'sarashina', self.projector, False, 'auto', 128)
        self.projector.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Projector/manifest'):
            launcher.launch_environment('cpu', 'sarashina', self.projector, False, 'auto', 128)

    def test_official_projector_requires_implemented_preprocessing_not_gpu_verification(self):
        path = self.projector.with_suffix('.gguf.json')
        info = json.loads(path.read_text())
        info['preprocessing'] = 'sarashina_official_v1'
        path.write_text(json.dumps(info))
        env = launcher.launch_environment('cpu', 'sarashina', self.projector, False, 'off', 128)
        self.assertEqual(env['LFM_MMPROJ'], str(self.projector))
        path = self.base/'build-cpu/jev-build.json'
        manifest = json.loads(path.read_text())
        manifest.pop('sarashina_preprocessing')
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'preprocessing'):
            launcher.launch_environment('cpu', 'sarashina', self.projector, False, 'off', 128)

    def test_gpu_device_and_kv_types_are_not_restricted_by_past_checks(self):
        with patch.dict(os.environ, {'GPU_DEVICE': 'ROCm1', 'LLAMA_ARG_CACHE_TYPE_K': 'q8_0'}):
            env = launcher.launch_environment('hip', 'sarashina', self.projector, False, 'auto', 128)
        self.assertEqual(env['GPU_DEVICE'], 'ROCm1')
        self.assertEqual(env['LLAMA_ARG_CACHE_TYPE_K'], 'q8_0')
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/explicit/rocm'}):
            env = launcher.launch_environment('hip', 'sarashina', self.projector, False, 'auto', 128)
        self.assertEqual(env['LD_LIBRARY_PATH'], '/explicit/rocm')
        self.assertEqual(env['GPU_DEVICE'], 'ROCm0')

    def test_cuda_launch_uses_cuda_plugin_and_device_override(self):
        env = launcher.launch_environment('cuda', 'sarashina', self.projector, False, 'on', 128)
        self.assertEqual(env['GPU_DEVICE'], 'CUDA0')
        self.assertEqual(env['GPU_LAYERS'], 'all')
        self.assertIn('build-cuda', env['LLAMA_SERVER'])
        with patch.dict(os.environ, {'GPU_DEVICE': 'CUDA1'}):
            env = launcher.launch_environment('cuda', 'sarashina', self.projector, False, 'on', 128, 'CUDA2')
        self.assertEqual(env['GPU_DEVICE'], 'CUDA2')

    def test_stock_fa_and_unverified_or_failed_checks_only_warn(self):
        env = launcher.launch_environment('stock-rocm', 'sarashina', self.projector, False, 'on', 128)
        self.assertEqual(env['LLAMA_ARG_FLASH_ATTN'], 'on')
        for backend in ('hip', 'cuda'):
            path = self.base/f'build-{backend}/jev-build.json'
            manifest = json.loads(path.read_text())
            for result in (None, {'status': 'not_run'}, {'status': 'failed', 'passed': 0},
                           {'status': 'passed', 'device': 'ROCm1', 'passed': 252, 'architecture': 'gfx1100'}):
                with self.subTest(backend=backend, result=result):
                    manifest['fa160_tests'] = result
                    path.write_text(json.dumps(manifest))
                    env = launcher.launch_environment(backend, 'sarashina', self.projector, False, 'on', 128)
                    self.assertEqual(env['LLAMA_ARG_FLASH_ATTN'], 'on')

    def test_profile_still_requires_matching_model(self):
        with patch.dict(os.environ, {'LFM_MODEL': '/custom.gguf'}), self.assertRaisesRegex(RuntimeError, 'Unset LFM_MODEL'):
            launcher.launch_environment('cpu', 'sarashina', self.projector, False, 'off', 128)


if __name__ == '__main__':
    unittest.main()
