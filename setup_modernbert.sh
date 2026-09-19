#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/scripts/setup_modernbert.py" "$@"
