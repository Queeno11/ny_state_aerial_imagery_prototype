#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
workers="${1:-${WORKERS:-4}}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash run_overnight_multicore.sh [positive-worker-count]"
  exit 2
fi
if [[ ! -f run.py ]]; then
  echo "Extract this bundle into ordinal_mc_suite, beside run.py."
  exit 1
fi

# Independent CPU fits run in separate processes, each with one numerical thread.
plan_stage() {
  local config="configs/overnight_${1}.json"
  if [[ "$1" == change ]]; then config="configs/change_recovery.json"; fi
  python -u run.py --config "$config" \
    --experiments "$2" \
    --workers "$workers" --threads 1 --plan
}
plan_stage measurement 1
plan_stage nn 3
plan_stage links 4
plan_stage change 5

run_stage() {
  local stage="$1" experiment="$2"
  # New scientific version: preserve all earlier overnight results.
  local output="${OUTPUT_PREFIX:-benchmarked}_${stage}"
  local config="configs/overnight_${stage}.json"
  if [[ "$stage" == change ]]; then config="configs/change_recovery.json"; fi
  local resume=()
  if [[ -f "$output/checkpoint.sqlite" ]]; then
    resume=(--resume)
  fi
  python -u run.py --config "$config" \
    --experiments "$experiment" --out "$output" \
    --workers "$workers" --threads 1 "${resume[@]}"
}

run_stage measurement 1
run_stage nn 3
run_stage links 4
run_stage change 5
