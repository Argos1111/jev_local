#!/usr/bin/env python3
"""Build the optional b11042 encoder cache; Linux/GCC, reusing pinned ROCm plugin."""
import argparse
import json
import platform
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

if __package__:
    from .setup_runtime import ROOT, download, sha256
else:
    from setup_runtime import ROOT, download, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1: parser.error('--jobs must be positive')
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('This experimental ROCm build recipe targets Linux x86_64')
    def tool(variable, default):
        requested = os.environ.get(variable, default)
        value = shutil.which(requested)
        if not value:
            parser.error(f'{requested} not found. Install GCC/g++, CMake and Ninja, or set {variable}; see docs/IMAGE_CACHE.md')
        return value
    cmake, ninja = tool('CMAKE', 'cmake'), tool('NINJA', 'ninja')
    cc, cxx = tool('CC', 'gcc'), tool('CXX', 'g++')
    plugin=ROOT/'.cache/runtime/b11042/linux-x86_64-rocm/llama-b11042/libggml-hip.so'
    if not plugin.is_file():
        parser.error('Pinned b11042 ROCm runtime is required. Run ./setup.sh --backend rocm first; see docs/IMAGE_CACHE.md')
    spec=json.loads((ROOT/'native/source.json').read_text())
    archive=ROOT/'.cache/llama-b11042-source.tar.gz'
    download(spec['url'],archive,spec['sha256'])
    source=ROOT/'.cache/vision-source'/f'llama.cpp-{spec["version"]}'
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as bundle:bundle.extractall(source.parent,filter='data')
    if 'jev_vision_embedding_cache' not in (source/'tools/mtmd/clip.cpp').read_text():
        subprocess.run([sys.executable,str(ROOT/'scripts/patch_vision_cache.py'),str(source)],check=True)
    shutil.copyfile(ROOT/'native/vision_embedding_cache.h',source/'tools/mtmd/vision_embedding_cache.h')
    env=dict(os.environ)
    build=ROOT/'.cache/vision-build-gcc'
    subprocess.run([cmake,'-S',str(source),'-B',str(build),'-G','Ninja',f'-DCMAKE_MAKE_PROGRAM={ninja}',
        f'-DCMAKE_C_COMPILER={cc}',f'-DCMAKE_CXX_COMPILER={cxx}','-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF','-DGGML_OPENMP=OFF','-DLLAMA_CURL=OFF',
        '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_UI=OFF'],env=env,check=True)
    subprocess.run([cmake,'--build',str(build),'--target','llama-server','-j',str(args.jobs)],env=env,check=True)
    link=build/'bin/libggml-hip.so'
    if link.is_symlink() and link.resolve() != plugin.resolve(): link.unlink()
    if not link.exists(): link.symlink_to(plugin)
    test=build/'test_vision_embedding_cache'
    subprocess.run([cxx,'-std=c++17','-pthread',str(ROOT/'native/test_vision_embedding_cache.cpp'),'-o',str(test)],env=env,check=True)
    subprocess.run([str(test)],env=env,check=True)
    manifest={'source':spec,'cache_header_sha256':sha256(ROOT/'native/vision_embedding_cache.h'),
              'patch_script_sha256':sha256(ROOT/'scripts/patch_vision_cache.py'),
              'server_sha256':sha256(build/'bin/llama-server'),'mtmd_sha256':sha256(build/'bin/libmtmd.so'),
              'rocm_plugin_sha256':sha256(plugin)}
    (build/'jev-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built: {build}/bin/llama-server. Enable with JEV_IMAGE_CACHE_MIB=128; 0 disables.')


if __name__=='__main__':main()
