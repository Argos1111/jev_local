#!/usr/bin/env bash
# Fine-tune ModernBERT-Ja-310M as a Jev-style cross-encoder on public Japanese data.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
python="$root/.venv-modernbert/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Missing $python. Run ./setup_modernbert.sh first." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$root/.cache/hf}"
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$python" -m modernbert.train "$@"
