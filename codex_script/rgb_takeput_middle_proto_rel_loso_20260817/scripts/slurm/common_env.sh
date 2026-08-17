#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${PACKAGE_ROOT}/run.py"
EXP_CONFIG="${PACKAGE_ROOT}/config/experiment_config.json"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python}"
export PACKAGE_ROOT RUNNER EXP_CONFIG PYTHON_BOOTSTRAP

