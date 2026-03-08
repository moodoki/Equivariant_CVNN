#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

declare -a EXPERIMENT_LABELS=(
    "cvnn_aps"
    "cvnn_lpf"
    "cvnn_mlp_lps"
    "cvnn_poly_lps"
)

declare -a EXPERIMENT_CONFIGS=(
    "configs/sf_alos2/config_sf_alos2_cvnn_aps.yml"
    "configs/sf_alos2/config_sf_alos2_cvnn_lpf.yml"
    "configs/sf_alos2/config_sf_alos2_cvnn_mlp_lps.yml"
    "configs/sf_alos2/config_sf_alos2_cvnn_poly_lps.yml"
)

RUN_TESTS=1
LIST_ONLY=0
START_AT=""
ONLY=""
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/tmp/reconstruction_reproduction}"
LOG_ROOT="${LOG_ROOT:-/data/equiv-cvnn/logs}"

usage() {
    cat <<'EOF'
Usage: scripts/run_reconstruction_reproduction.sh [--list] [--skip-test] [--start-at <label>] [--only <label>]

Runs the paper-facing ALOS2 reconstruction experiments sequentially on the current Aim-enabled worktree.

Labels:
  cvnn_aps
  cvnn_lpf
  cvnn_mlp_lps
  cvnn_poly_lps

Environment overrides:
  RUN_ROOT
  LOG_ROOT
  AIM_REPO
  EXPERIMENT_LOGGER_BACKEND
  WANDB_DISABLED
  WANDB_MODE
EOF
}

while (($# > 0)); do
    case "$1" in
        --list)
            LIST_ONLY=1
            shift
            ;;
        --skip-test)
            RUN_TESTS=0
            shift
            ;;
        --start-at)
            START_AT="${2:-}"
            shift 2
            ;;
        --only)
            ONLY="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${AIM_REPO:-}" ]]; then
    if [[ -d /data/equiv-cvnn ]]; then
        AIM_REPO="/data/equiv-cvnn/aimlogs"
    else
        AIM_REPO="$PROJECT_ROOT/aimlogs"
    fi
fi

export AIM_REPO
export EXPERIMENT_LOGGER_BACKEND="${EXPERIMENT_LOGGER_BACKEND:-aim}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

list_experiments() {
    local idx
    for idx in "${!EXPERIMENT_LABELS[@]}"; do
        printf '%s\t%s\n' "${EXPERIMENT_LABELS[$idx]}" "${EXPERIMENT_CONFIGS[$idx]}"
    done
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    list_experiments
    exit 0
fi

if [[ -n "$ONLY" && -n "$START_AT" ]]; then
    printf 'Use either --only or --start-at, not both.\n' >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
mkdir -p "$LOG_ROOT"
summary_file="$RUN_ROOT/run_summary.tsv"
printf 'label\tconfig\teffective_config\tlogdir\taim_repo\ttrain_log\ttest_log\n' > "$summary_file"

started=1
if [[ -n "$START_AT" ]]; then
    started=0
fi

run_count=0

for idx in "${!EXPERIMENT_LABELS[@]}"; do
    label="${EXPERIMENT_LABELS[$idx]}"
    config="${EXPERIMENT_CONFIGS[$idx]}"

    if [[ -n "$ONLY" && "$label" != "$ONLY" ]]; then
        continue
    fi

    if [[ "$started" -eq 0 ]]; then
        if [[ "$label" == "$START_AT" ]]; then
            started=1
        else
            continue
        fi
    fi

    run_count=$((run_count + 1))
    logdir_file="$RUN_ROOT/${label}.logdir"
    effective_config="$RUN_ROOT/${label}.config.yml"
    train_log="$RUN_ROOT/${label}.train.log"
    test_log="$RUN_ROOT/${label}.test.log"

    python - "$config" "$effective_config" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
effective_path = Path(sys.argv[2])
log_root = sys.argv[3]

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.setdefault("logging", {})["logdir"] = log_root

with effective_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

    printf '\n[%d/%d] Training %s with %s\n' \
        "$run_count" "${#EXPERIMENT_LABELS[@]}" "$label" "$effective_config"
    python -m torchtmpl.main train "$effective_config" "$logdir_file" | tee "$train_log"

    if [[ ! -s "$logdir_file" ]]; then
        printf 'Training did not produce a logdir file for %s\n' "$label" >&2
        exit 1
    fi

    logdir="$(<"$logdir_file")"
    printf 'Resolved logdir for %s: %s\n' "$label" "$logdir"

    if [[ "$RUN_TESTS" -eq 1 ]]; then
        printf '[%d/%d] Testing %s from %s\n' \
            "$run_count" "${#EXPERIMENT_LABELS[@]}" "$label" "$logdir"
        python -m torchtmpl.main test "$logdir" | tee "$test_log"
    else
        : > "$test_log"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" \
        "$config" \
        "$effective_config" \
        "$logdir" \
        "$AIM_REPO" \
        "$train_log" \
        "$test_log" \
        >> "$summary_file"
done

printf '\nCompleted reconstruction reproduction suite.\n'
printf 'Summary: %s\n' "$summary_file"
