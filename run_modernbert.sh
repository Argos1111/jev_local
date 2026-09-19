#!/usr/bin/env bash
# Serve the TypeSafe-compatible API from the fine-tuned ModernBERT-Ja cross-encoder.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
python="$root/.venv-modernbert/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Missing $python. Run ./setup_modernbert.sh first." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$root/.cache/hf}"
# AMD wheels route some ops through Triton JIT, which needs a C compiler; the eager kernels are sufficient here.
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"
exec "$python" -m modernbert.server "$@"
