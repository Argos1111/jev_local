#!/usr/bin/env bash
# Experimental b11042 ROCm build; see docs/IMAGE_CACHE.md.
set -euo pipefail
project_root="$(cd "$(dirname "$0")" && pwd)"
export LLAMA_SERVER="$project_root/.cache/vision-build-gcc/bin/llama-server"
if [[ ! -x "$LLAMA_SERVER" || ! -f "$project_root/.cache/vision-build-gcc/jev-build.json" ]]; then
    echo 'Run python3 scripts/build_image_cache.py first; see docs/IMAGE_CACHE.md.' >&2
    exit 1
fi
rocm_library_dir="${ROCM_LIBRARY_DIR:-$HOME/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v4}"
if [[ -d "$rocm_library_dir" ]]; then
    export LD_LIBRARY_PATH="$rocm_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export GPU_LAYERS="${GPU_LAYERS:-all}"
export GPU_DEVICE="${GPU_DEVICE:-ROCm0}"
export CTX_SIZE="${CTX_SIZE:-32768}"
export JEV_IMAGE_CACHE_MIB="${JEV_IMAGE_CACHE_MIB:-128}"
exec "$project_root/run.sh" --model vision "$@"
