#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 JOB.slurm [sbatch options...]" >&2
  exit 2
fi
JOB="$1"
shift
cd "${SCRIPT_DIR}"
sbatch "$@" "${JOB}"
