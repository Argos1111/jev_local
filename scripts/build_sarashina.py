#!/usr/bin/env python3
"""Build the opt-in Sarashina b11042 runtime in an isolated directory (Linux/WSL2)."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

if __package__:
    from .setup_runtime import ROOT, download, runtime_env, sha256
else:
    from setup_runtime import ROOT, download, runtime_env, sha256

BASE = ROOT/'.cache/sarashina-official-runtime'
PATCHES = ('sarashina-vision-b11042.patch', 'sarashina-fa160-b11042.patch',
           'sarashina-official-preprocess-b11042.patch')
BACKENDS = ('cpu', 'stock-rocm', 'hip', 'cuda')
GPU_PLUGINS = {'hip': 'libggml-hip.so', 'stock-rocm': 'libggml-hip.so', 'cuda': 'libggml-cuda.so'}
DEFAULT_DEVICES = {'hip': 'ROCm0', 'stock-rocm': 'ROCm0', 'cuda': 'CUDA0'}
RUNTIME_VARS = ('LD_LIBRARY_PATH', 'ROCBLAS_TENSILE_LIBPATH', 'HIPBLASLT_TENSILE_LIBPATH')
FA160_CASES = 252


def prepare_source(base=BASE, patch_tool='patch'):
    spec = json.loads((ROOT/'native/source.json').read_text())
    identity = {'source': spec, 'patches': {name: sha256(ROOT/'native'/name) for name in PATCHES},
                'cache_header_sha256': sha256(ROOT/'native/vision_embedding_cache.h')}
    source = base/'source'/f'llama.cpp-{spec["version"]}'
    stamp = source/'jev-source.json'
    if source.exists():
        if not stamp.is_file():
            raise RuntimeError(f'Unmanaged source tree: {source}; move it aside before building')
        saved = json.loads(stamp.read_text())
        if saved.get('identity') != identity or any(not (source/p).is_file() or sha256(source/p) != h
                                                  for p, h in saved['files'].items()):
            raise RuntimeError(f'Source/patches changed: {source}; preserve it elsewhere, then rebuild')
        return source, identity
    archive = ROOT/'.cache/llama-b11042-source.tar.gz'
    download(spec['url'], archive, spec['sha256'])
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter='data')
        unpacked = Path(temporary)/source.name
        touched = set()
        for name in PATCHES:
            patch = ROOT/'native'/name
            subprocess.run([patch_tool, '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch',
                            '-p1', '-i', str(patch)], cwd=unpacked, check=True)
            touched.update(line.removeprefix('+++ b/') for line in patch.read_text().splitlines()
                           if line.startswith('+++ b/'))
        header = 'tools/mtmd/vision_embedding_cache.h'
        shutil.copyfile(ROOT/'native/vision_embedding_cache.h', unpacked/header)
        touched.add(header)
        instances = unpacked/'ggml/src/ggml-cuda/template-instances'
        subprocess.run([sys.executable, 'generate_cu_files.py'], cwd=instances, check=True)
        touched.update(str(p.relative_to(unpacked)) for p in instances.glob('*.cu'))
        (unpacked/stamp.name).write_text(json.dumps({'identity': identity,
            'files': {p: sha256(unpacked/p) for p in sorted(touched)}}, indent=2) + '\n')
        unpacked.rename(source)
    return source, identity


def build_environment(build, binary_dir=None):
    """Use this build's libraries; explicit environment values take precedence."""
    directory = binary_dir if binary_dir is not None else build/'bin'
    env = runtime_env(directory)
    manifest = build/'jev-build.json'
    if manifest.is_file():
        saved = json.loads(manifest.read_text()).get('runtime_env', {})
        for key in RUNTIME_VARS:
            if not os.environ.get(key) and saved.get(key):
                env[key] = saved[key]
    libraries = str(directory.resolve())
    paths = env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys([libraries, *(p for p in paths if p)]))
    return env


def fa160_test_result(output, returncode, device):
    """Grade the requested backend, not another GPU or a CPU-only/empty run."""
    output = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', output)
    section = re.search(rf'^Backend \d+/\d+: {re.escape(device)}\n(.*?)(?=^Backend \d+/\d+:|\Z)',
                        output, re.MULTILINE | re.DOTALL)
    text = section[1] if section else ''
    counts = re.search(r'^\s*(\d+)/(\d+) tests passed\s*$', text, re.MULTILINE)
    passed, total = map(int, counts.groups()) if counts else (0, 0)
    verified = bool(re.fullmatch(r'(?:CUDA|ROCm|HIP)\d+', device) and returncode == 0
                    and passed == total == FA160_CASES and not re.search(r'FAIL|NOT[_ ]SUPPORTED', text))
    description = re.search(r'^\s*Device description: (.+)$', text, re.MULTILINE)
    return {'status': 'passed' if verified else 'failed', 'device': device,
            'description': description[1] if description else None,
            'passed': passed, 'total': total, 'expected': FA160_CASES, 'returncode': returncode}


def fa160_verified(output, returncode, device):
    return fa160_test_result(output, returncode, device)['status'] == 'passed'


def run_fa160_tests(build, device, log, timeout=600):
    env = build_environment(build)
    env['LLAMA_TEST_FA_VEC_DISABLE'] = '1'
    command = [str((build/'bin/test-backend-ops').resolve()), 'test', '-b', device,
               '-o', 'FLASH_ATTN_EXT', '-p', 'hsk=160']
    status = None
    try:
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        output, returncode = result.stdout, result.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        output += f'\nFA160 check timed out after {timeout}s\n'
        returncode, status = None, 'timeout'
    except OSError as exc:
        output, returncode, status = str(exc) + '\n', None, 'error'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output)
    report = fa160_test_result(output, returncode, device)
    if status:
        report['status'] = status
    report.update(command=command, log=str(log.resolve()), log_sha256=sha256(log),
                  kv_types=['f16', 'f16'], vector_slice_tests=False)
    return report


def backend_options(backend, cuda_architectures=None, hip_architectures=None):
    options = ['-DLLAMA_BUILD_TESTS=' + ('ON' if backend in ('hip', 'cuda') else 'OFF'),
               '-DGGML_HIP=' + ('ON' if backend == 'hip' else 'OFF'),
               '-DGGML_CUDA=' + ('ON' if backend == 'cuda' else 'OFF')]
    if backend == 'hip':
        options += ['-DGGML_HIP_GRAPHS=ON', '-DGGML_HIP_NO_VMM=ON', '-DGGML_CUDA_FA=ON']
        if hip_architectures:
            options.append(f'-DCMAKE_HIP_ARCHITECTURES={hip_architectures}')
    if backend == 'cuda':
        options += ['-DGGML_CUDA_GRAPHS=ON', '-DGGML_CUDA_FA=ON']
        if cuda_architectures:
            options.append(f'-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}')
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, default='stock-rocm',
                        help='hip/cuda build FA160; stock-rocm reuses the unpatched official GPU plugin')
    parser.add_argument('--base', type=Path, help='Isolated source/build directory (default: .cache/sarashina-official-runtime)')
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--cuda-architectures', help='CMake CUDA targets, e.g. 89 (RTX 4090) or "86;89"; default: upstream selection')
    parser.add_argument('--hip-architectures', help='CMake HIP targets, e.g. gfx1100 or "gfx1100;gfx1201"; default: compiler detection')
    parser.add_argument('--test-fa', action='store_true', help='Run optional GPU numerical checks after building; failure does not block use')
    parser.add_argument('--test-device', help='Device for optional checks, default: CUDA0 or ROCm0')
    parser.add_argument('--test-timeout', type=float, default=600)
    parser.add_argument('--cmake-arg', action='append', default=[], help='Extra -D option, e.g. --cmake-arg=-DCMAKE_CUDA_COMPILER=/path/to/nvcc')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('This shared-library recipe uses Linux paths; on Windows use WSL2')
    if args.jobs < 1 or args.test_timeout <= 0:
        parser.error('--jobs and --test-timeout must be positive')
    if args.test_fa and args.backend not in ('hip', 'cuda'):
        parser.error('--test-fa requires a source-built hip or cuda backend')
    def tool(variable, default):
        path = shutil.which(os.environ.get(variable, default))
        if not path:
            parser.error(f'Install {default} or set {variable}; see docs/SARASHINA.md')
        return path
    cmake, ninja = tool('CMAKE', 'cmake'), tool('NINJA', 'ninja')
    cc, cxx = tool('CC', 'gcc'), tool('CXX', 'g++')
    patch_tool = tool('PATCH', 'patch')
    plugin = ROOT/'.cache/runtime/b11042/linux-x86_64-rocm/llama-b11042/libggml-hip.so'
    if args.backend == 'stock-rocm' and not plugin.is_file():
        parser.error('Pinned b11042 ROCm plugin is required; see docs/SARASHINA.md')
    base = args.base.resolve() if args.base else BASE
    source, identity = prepare_source(base=base, patch_tool=patch_tool)
    build = base/f'build-{args.backend}'
    command = [cmake, '-S', str(source), '-B', str(build), '-G', 'Ninja',
               f'-DCMAKE_MAKE_PROGRAM={ninja}', f'-DCMAKE_C_COMPILER={cc}', f'-DCMAKE_CXX_COMPILER={cxx}',
               '-DCMAKE_BUILD_TYPE=Release', '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
               '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF',
               '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_APP=OFF']
    command += backend_options(args.backend, args.cuda_architectures, args.hip_architectures) + args.cmake_arg
    subprocess.run(command, check=True)
    targets = ['llama-server'] + (['test-backend-ops'] if args.backend in ('hip', 'cuda') else [])
    subprocess.run([cmake, '--build', str(build), '--target', *targets, '-j', str(args.jobs)], check=True)
    if args.backend == 'stock-rocm':
        link = build/'bin/libggml-hip.so'
        if link.is_symlink() and link.resolve() == plugin.resolve():
            pass
        elif link.exists() or link.is_symlink():
            raise RuntimeError(f'Refusing to replace {link}')
        else:
            link.symlink_to(plugin)
    subprocess.run([cxx, '-std=c++17', '-O2', f'-I{source}/include', f'-I{source}/ggml/include',
                    f'-I{source}/tools/mtmd', str(ROOT/'native/test_sarashina_embedding.cpp'),
                    f'-L{build}/bin', '-Wl,-rpath,$ORIGIN', '-lmtmd', '-lggml', '-lggml-base',
                    '-o', str(build/'bin/test_sarashina_embedding')], check=True)
    subprocess.run([cxx, '-std=c++17', '-pthread', str(ROOT/'native/test_vision_embedding_cache.cpp'),
                    '-o', str(build/'test_vision_embedding_cache')], check=True)
    subprocess.run([str(build/'test_vision_embedding_cache')], check=True)
    tests = {'status': 'not_run'}
    if args.test_fa:
        tests = run_fa160_tests(build, args.test_device or DEFAULT_DEVICES[args.backend],
                               build/'fa160-test.log', args.test_timeout)
        print(f'FA160 check: {tests["status"]} ({tests["passed"]}/{tests["total"]}); see {tests["log"]}', flush=True)
    files = ['llama-server', 'libllama-server-impl.so', 'libllama-common.so', 'libmtmd.so', 'libllama.so',
             'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'test_sarashina_embedding']
    if args.backend in GPU_PLUGINS:
        files.append(GPU_PLUGINS[args.backend])
    if args.backend in ('hip', 'cuda'):
        files.append('test-backend-ops')
    manifest = {'identity': identity, 'backend': args.backend, 'configure': command, 'fa160_tests': tests,
                'sarashina_preprocessing': ['legacy_pillow', 'sarashina_official_v1'],
                'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
                'toolchain_env': {k: os.environ[k] for k in ('CMAKE_PREFIX_PATH', 'ROCM_PATH', 'HIP_PATH', 'HIP_PLATFORM',
                                                          'CUDACXX', 'CUDAHOSTCXX', 'CUDAToolkit_ROOT')
                                  if os.environ.get(k)},
                'runtime_env': {k: os.environ[k] for k in RUNTIME_VARS if os.environ.get(k)},
                'binaries': {p: sha256(build/'bin'/p) for p in files}}
    (build/'jev-build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Built: {build}/bin/llama-server\nNormal runtime/model selections were not changed.')
    if args.backend in ('hip', 'cuda') and tests['status'] != 'passed':
        print('WARNING: FA160 numerical checks have not passed on this build. Use is not blocked.\n'
              f'Optional check: python3 -m tools.verify_sarashina_fa --backend {args.backend}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f'Sarashina build failed: {exc}')
