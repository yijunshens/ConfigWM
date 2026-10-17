#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: bash launch/train_configwm.sh [options]

Required:
  --env ENV                 pusht, tworoom, or ogbcube
  --gpus IDS                GPU IDs, e.g. 0 or 0,1
  --data-root PATH          Dataset root
  --dino-path PATH          DINOv2 model ID or local path
  --encoder-checkpoint PATH Configurable Encoder checkpoint directory
Optional:
  --output-dir PATH         Output directory (default: outputs/configwm/...)
  --batch-size N            Batch size (default: 512)
  --variant LEVEL           Distraction level (default: d0)
  -h, --help                Show this help
EOF
}

ENV_NAME=""
GPU_IDS=""
DATA_ROOT=""
DINO_PATH=""
ENCODER_CHECKPOINT=""
OUTPUT_DIR=""
BATCH_SIZE="512"
VARIANT="d0"

# Keep existing positional commands working.
if [[ $# -gt 0 && "$1" != -* ]]; then
  if [[ $# -lt 5 || $# -gt 8 ]]; then
    usage >&2
    exit 2
  fi
  ENV_NAME="${1}"
  GPU_IDS="${2}"
  DATA_ROOT="${3}"
  DINO_PATH="${4}"
  ENCODER_CHECKPOINT="${5}"
  OUTPUT_DIR="${6:-}"
  BATCH_SIZE="${7:-512}"
  VARIANT="${8:-d0}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --env|--gpus|--data-root|--dino-path|--encoder-checkpoint|--output-dir|--batch-size|--variant)
        if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
          echo "Missing value for $1" >&2
          exit 2
        fi
        ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    case "$1" in
      --env) ENV_NAME="$2" ;;
      --gpus) GPU_IDS="$2" ;;
      --data-root) DATA_ROOT="$2" ;;
      --dino-path) DINO_PATH="$2" ;;
      --encoder-checkpoint) ENCODER_CHECKPOINT="$2" ;;
      --output-dir) OUTPUT_DIR="$2" ;;
      --batch-size) BATCH_SIZE="$2" ;;
      --variant) VARIANT="$2" ;;
    esac
    shift 2
  done
fi

[[ -n "${ENV_NAME}" ]] || { echo "Missing required option: --env" >&2; exit 2; }
[[ -n "${GPU_IDS}" ]] || { echo "Missing required option: --gpus" >&2; exit 2; }
[[ -n "${DATA_ROOT}" ]] || { echo "Missing required option: --data-root" >&2; exit 2; }
[[ -n "${DINO_PATH}" ]] || { echo "Missing required option: --dino-path" >&2; exit 2; }
[[ -n "${ENCODER_CHECKPOINT}" ]] || { echo "Missing required option: --encoder-checkpoint" >&2; exit 2; }

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/configwm/${ENV_NAME}/${VARIANT}/${RUN_NAME}}"

case "${ENV_NAME}" in
  pusht)
    ACTION_DIM=2
    QUERY="gray T block"
    ;;
  tworoom)
    ACTION_DIM=2
    QUERY="red ball"
    ;;
  ogbcube)
    ACTION_DIM=5
    QUERY="silver arm"
    ;;
  *)
    echo "unknown environment: ${ENV_NAME}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "ConfigWM output: ${OUTPUT_DIR}"

VARIANT_DIR="$DATA_ROOT/$ENV_NAME/$VARIANT"
QUERY_ARGS=(--query-metadata "$VARIANT_DIR/query_metadata.jsonl")
if [[ -f "$DATA_ROOT/dataset.json" ]]; then
  VARIANT_DIR="$DATA_ROOT/$ENV_NAME/trajectories/$VARIANT"
  QUERY_ARGS=()
fi

torchrun --standalone --nproc_per_node=1 -m src.world_model.train \
  --config "$ROOT/configs/configwm.yaml" \
  --data-root "$VARIANT_DIR" \
  --encoder-type configurable_encoder \
  --encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --dino-path "$DINO_PATH" \
  --query "$QUERY" \
  "${QUERY_ARGS[@]}" \
  --sample-mode random \
  --normalize-actions \
  --action-dim "$ACTION_DIM" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-24}" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
