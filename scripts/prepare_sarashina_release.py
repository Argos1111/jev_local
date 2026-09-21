#!/usr/bin/env python3
"""Package a relocatable Sarashina runtime locally. No login, upload or model files."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from string import Template
import tarfile
import tempfile

if __package__:
    from .build_sarashina import PATCHES, GPU_PLUGINS
    from .setup_runtime import ROOT, sha256
else:
    from build_sarashina import PATCHES, GPU_PLUGINS
    from setup_runtime import ROOT, sha256

COMMON = {'llama-server', 'libllama-server-impl.so', 'libllama-common.so', 'libmtmd.so',
          'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'test_sarashina_embedding'}
SYSTEM_LIBRARIES = {'ld-linux-x86-64.so.2', 'libc.so.6', 'libm.so.6', 'libstdc++.so.6',
                    'libgcc_s.so.1', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1'}
GPU_LIBRARIES = {'cpu': set(), 'hip': {'libamdhip64.so.7', 'libhipblas.so.3', 'librocblas.so.5'},
                 'cuda': {'libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13', 'libcuda.so.1'}}
SOURCE_INPUTS = ['native/source.json', 'native/vision_embedding_cache.h',
                 'native/test_sarashina_embedding.cpp', 'native/test_vision_embedding_cache.cpp',
                 'scripts/build_sarashina.py', 'scripts/setup_runtime.py',
                 *('native/'+name for name in PATCHES)]


def source_identity():
    return {'source': json.loads((ROOT/'native/source.json').read_text()),
            'patches': {name: sha256(ROOT/'native'/name) for name in PATCHES},
            'cache_header_sha256': sha256(ROOT/'native/vision_embedding_cache.h')}


def binary_files(build, manifest):
    backend = manifest.get('backend')
    if backend not in GPU_LIBRARIES:
        raise ValueError('Only source-built CPU/HIP/CUDA releases are supported')
    required = COMMON | ({GPU_PLUGINS[backend], 'test-backend-ops'} if backend != 'cpu' else set())
    recorded = manifest.get('binaries', {})
    if not required <= recorded.keys():
        raise ValueError('Incomplete build manifest; rebuild with the current build script')
    directory = (build/'bin').resolve()
    for name, digest in recorded.items():
        if Path(name).name != name or not (directory/name).is_file() or sha256(directory/name) != digest:
            raise ValueError(f'Changed or missing build artifact: {name}')
    files = {directory/name for name in required} | set(directory.glob('*.so*'))
    for path in files:
        if not path.is_file() or path.resolve().parent != directory:
            raise ValueError(f'External or broken binary link: {path.name}')
        if path.is_symlink() and (os.path.isabs(os.readlink(path)) or Path(os.readlink(path)).name != os.readlink(path)):
            raise ValueError(f'Non-relocatable binary link: {path.name}')
        if path.name not in required and not re.fullmatch(r'lib(?:llama(?:-common)?|mtmd|ggml(?:-base)?)\.so(?:\.[0-9]+)*', path.name):
            raise ValueError(f'Unexpected library: {path.name}')
    return sorted(files)


def inspect_elf(path):
    env = dict(os.environ, LC_ALL='C')
    header = subprocess.check_output(['readelf', '-h', str(path)], env=env, text=True)
    if 'Advanced Micro Devices X86-64' not in header:
        raise ValueError(f'Expected Linux x86_64 ELF: {path.name}')
    dynamic = subprocess.check_output(['readelf', '-d', str(path)], env=env, text=True)
    rpaths = re.findall(r'\((?:RUNPATH|RPATH)\).*?\[([^\]]*)\]', dynamic)
    if rpaths != ['$ORIGIN']:
        raise ValueError(f'{path.name}: build with CMAKE_BUILD_WITH_INSTALL_RPATH=ON and CMAKE_INSTALL_RPATH=$ORIGIN')
    versions = subprocess.check_output(['readelf', '--version-info', str(path)], env=env, text=True)
    abi = {}
    for prefix in ('GLIBC', 'GLIBCXX', 'CXXABI'):
        values = set(re.findall(r'Name: ('+prefix+r'_[\d.]+)\b', versions))
        if values:
            abi[prefix] = max(values, key=lambda s: tuple(map(int, s.split('_')[-1].split('.'))))
    return {'needed': re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', dynamic), 'rpath': '$ORIGIN', 'abi': abi}


def upstream_notices(archive):
    spec = source_identity()['source']
    if sha256(archive) != spec['sha256']:
        raise ValueError('Upstream source archive hash mismatch')
    with tarfile.open(archive) as bundle:
        def text(name):
            with bundle.extractfile(f'llama.cpp-{spec["version"]}/'+name) as stream:
                return stream.read().decode()
        notices = {name: text(path) for name, path in {
            'LICENSE.llama.cpp': 'LICENSE', 'LICENSE.cpp-httplib': 'vendor/cpp-httplib/LICENSE',
            'LICENSE.nlohmann-json': 'licenses/LICENSE-jsonhpp', 'LICENSE.xxhash': 'vendor/hash/xxhash/LICENSE',
            'LICENSE.rotate-bits': 'vendor/hash/rotate-bits/LICENSE.md', 'LICENSE.sha256': 'vendor/hash/sha256/LICENSE',
        }.items()}
        for name, path, start in (
            ('LICENSE.stb_image', 'vendor/stb/stb_image.h', 'This software is available under 2 licenses'),
            ('LICENSE.miniaudio', 'vendor/miniaudio/miniaudio.h', 'This software is available as a choice'),
        ):
            notices[name] = text(path).split(start, 1)[1].rsplit('*/', 1)[0]
            notices[name] = start + notices[name]
        notices['LICENSE.base64'] = text('common/base64.hpp').split('*/', 1)[0].removeprefix('/*\n')
        notices['LICENSE.subprocess'] = text('vendor/sheredom/subprocess.h').split('/*', 2)[2].split('*/', 1)[0]
        notices['NOTICE.sha1'] = text('vendor/hash/sha1/sha1.c').split('*/', 1)[0].removeprefix('/*\n')
        sgemm = text('ggml/src/ggml-cpu/llamafile/sgemm.cpp').split('// SOFTWARE.', 1)[0] + '// SOFTWARE.\n'
        notices['LICENSE.tinyblas'] = '\n'.join(line.removeprefix('//').lstrip() for line in sgemm.splitlines()) + '\n'
        json_text = text('vendor/nlohmann/json.hpp')
        attributions = sorted(set(line.removeprefix('// ') for line in json_text.splitlines()
                                  if line.startswith('// SPDX-FileCopyrightText:')))
        mit = notices['LICENSE.nlohmann-json'].split('Permission is hereby granted', 1)[1]
        notices['NOTICE.nlohmann-components'] = ('Hedley, Grisu2 and UTF-8 decoder notices from vendor/nlohmann/json.hpp\n\n'
            + '\n'.join(line for line in attributions if 'Abseil' not in line)
            + '\n\nPermission is hereby granted' + mit)
        notices['NOTICE.nlohmann-Abseil'] = ('Copyright 2018 The Abseil Authors.\n'
            'The C++11 fallback in nlohmann/json references Abseil under Apache-2.0.\n'
            'This C++17 build uses the standard-library implementation instead.\n'
            'The original source attribution is retained in the pinned upstream archive.\n')
        notices['LICENSE.YaRN'] = ('MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.\n'
            'Source: ggml/src/ggml-cpu/ops.cpp and ggml/src/ggml-cuda/rope.cu\n\nPermission is hereby granted' + mit)
        return notices


def toolchain_notices(backend):
    directory = ROOT/'licenses/runtime'
    lock = json.loads((directory/'sources.json').read_text())
    result = {}
    for name, spec in lock.items():
        if backend not in spec['backends']:
            continue
        if Path(name).name != name or sha256(directory/name) != spec['sha256']:
            raise ValueError(f'Changed toolchain notice: {name}')
        result[name] = (directory/name).read_text()
    return result


def write_archive(folder, destination):
    with destination.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as bundle:
            for path in [folder, *sorted(folder.rglob('*'))]:
                info = bundle.gettarinfo(path, arcname=str(path.relative_to(folder.parent)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o755 if path.is_dir() or (path.is_file() and path.parent.name == 'bin') else 0o644
                if path.is_symlink():
                    info.mode = 0o777
                if info.isfile():
                    with path.open('rb') as stream:
                        bundle.addfile(info, stream)
                else:
                    bundle.addfile(info)


def prepare(build, output, name, source_archive, verification=None):
    build, output = Path(build).resolve(), Path(output).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]+', name):
        raise ValueError('Use a plain archive name without slashes')
    folder, archive = output/name, output/(name+'.tar.gz')
    for path in (folder, archive):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite {path}')
    manifest = json.loads((build/'jev-build.json').read_text())
    if manifest.get('identity') != source_identity():
        raise ValueError('Source/patch identity differs from this checkout')
    if 'sarashina_official_v1' not in manifest.get('sarashina_preprocessing', []):
        raise ValueError('Build lacks official preprocessing v1')
    files = binary_files(build, manifest)
    elf = {p.name: inspect_elf(p) for p in files if not p.is_symlink()}
    external = set().union(*(set(r['needed']) for r in elf.values())) - {p.name for p in files}
    backend = manifest['backend']
    unexpected = external - SYSTEM_LIBRARIES - GPU_LIBRARIES[backend]
    if unexpected:
        raise ValueError(f'Unaccounted external libraries: {sorted(unexpected)}')
    notices = {**upstream_notices(source_archive), **toolchain_notices(backend)}
    checks = {'status': 'not_run'}
    if verification:
        checks = json.loads(Path(verification).read_text())
        if checks.get('input_binaries_sha256') != manifest['binaries']:
            raise ValueError('Verification report belongs to a different build')
    # Do not publish local search paths or the full build environment.
    public = {key: manifest[key] for key in ('identity', 'backend', 'platform', 'sarashina_preprocessing')}
    public.update(runtime_env={}, toolchain_env={},
                  configure=[arg.replace(str(ROOT), '${JEV_LOCAL}') for arg in manifest['configure']],
                  fa160_tests=checks.get('fa160_tests', {'status': 'not_run'}),
                  packaging={'external_libraries': sorted(external), 'elf': elf,
                             'source_inputs_sha256': {p: sha256(ROOT/p) for p in SOURCE_INPUTS},
                             'models_included': False, 'gpu_runtime_shared_libraries_included': False})
    folder.mkdir(parents=True)
    (folder/'bin').mkdir()
    for path in files:
        shutil.copy2(path, folder/'bin'/path.name, follow_symlinks=False)
    for path in files:
        if sha256(path) != sha256(folder/'bin'/path.name):
            raise ValueError(f'Binary changed during copy: {path.name}')
    for filename, digest in manifest['binaries'].items():
        if sha256(folder/'bin'/filename) != digest:
            raise ValueError(f'Copied binary differs from build manifest: {filename}')
    public['binaries'] = {p.name: sha256(folder/'bin'/p.name) for p in files}
    (folder/'jev-build.json').write_text(json.dumps(public, indent=2) + '\n')
    (folder/'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    (folder/'licenses').mkdir()
    for filename, content in notices.items():
        (folder/'licenses'/filename).write_text(content)
    for path in SOURCE_INPUTS:
        target = folder/'source-inputs'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    for filename in ('README.md', 'NOTICE.md'):
        template = Template((ROOT/'native/sarashina-runtime'/f'{filename}.in').read_text())
        (folder/filename).write_text(template.substitute(name=name, backend=backend))
    hashes = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    (folder/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p, h in sorted(hashes.items())))
    write_archive(folder, archive)
    # Re-extract with the safe filter and verify the complete file/link closure.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter='data')
        extracted = Path(temporary)/name
        for path, digest in hashes.items():
            if sha256(extracted/path) != digest:
                raise ValueError(f'Archive verification failed: {path}')
        if sha256(extracted/'SHA256SUMS') != sha256(folder/'SHA256SUMS'):
            raise ValueError('Archive checksum-list mismatch')
        for path in files:
            if path.is_symlink() and os.readlink(extracted/'bin'/path.name) != os.readlink(path):
                raise ValueError(f'Archive symlink mismatch: {path.name}')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/sarashina-prerelease/assets')
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-archive', type=Path, default=ROOT/'.cache/llama-b11042-source.tar.gz')
    parser.add_argument('--verification', type=Path, help='Optional report tied to exact build hashes; pass/fail never gates packaging')
    args = parser.parse_args()
    archive = prepare(args.build, args.output, args.name, args.source_archive, args.verification)
    print(f'{sha256(archive)}  {archive}')
    print('Prepared locally. No upload, git operation or visibility change performed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'Cannot prepare runtime release: {exc}')
