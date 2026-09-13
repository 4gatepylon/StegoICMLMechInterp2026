#!/usr/bin/env bash

set -euo pipefail

readonly TRAINING_MODULE="ciphers.kirchenbauer_et_al.src.train_kl_fineweb"
readonly CONFIGURATION_DIRECTORY="ciphers/kirchenbauer_et_al/experiments"
readonly WANDB_PROJECT="stego-kirchenbauer-prefix-kl"
readonly BIT_CONFIGURATIONS=(eight four two one)
readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIRECTORY}" rev-parse --show-toplevel)"

cd "${REPO_ROOT}"

for bit_configuration in "${BIT_CONFIGURATIONS[@]}"; do
  conda run --no-capture-output -n stego \
    python -m "${TRAINING_MODULE}" \
    --config "${CONFIGURATION_DIRECTORY}/${bit_configuration}_bit_training_run.yaml" \
    --wandb-project "${WANDB_PROJECT}"
done
