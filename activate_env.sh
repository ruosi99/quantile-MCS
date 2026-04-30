#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH. Please open a shell with conda available before running experiments."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate py12
