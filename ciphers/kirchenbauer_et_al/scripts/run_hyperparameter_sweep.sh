#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIRECTORY}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

exec conda run --no-capture-output -n stego \
  python -m ciphers.kirchenbauer_et_al.src.sweep_kl_fineweb "$@"
