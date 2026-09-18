#!/usr/bin/env bash
set -euo pipefail
export GPU_LAYERS="${GPU_LAYERS:-all}"
exec "$(dirname "$0")/run_server.sh" "$@"
