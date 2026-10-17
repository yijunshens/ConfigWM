#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: bash launch/train_configurable_encoder.sh [options]

Required:
  --env ENV                 pusht, tworoom, ogbcube, unified, or realworld
  --gpus IDS                GPU IDs, e.g. 0 or 0,1
  --data-root PATH          Dataset root
  --vision-model PATH       Vision model ID or local path (optional with --backbone)
  --predictor-model PATH    Predictor backbone model ID or local path
Optional:
  --output-dir PATH         Output directory (default: outputs/configurable_encoder/...)
  --backbone MODEL          Preset name, supported model ID/official URL, or local model path
  --backbone-code-dir PATH   Override the standard third_party source directory
  --backbone-checkpoint PATH Override automatically downloaded V-JEPA weights
  --resolution PIXELS       Override the backbone input resolution
  --batch-size N            Per-GPU batch size
  --group-size N            Real-world contrastive group size (default: 8)
  --steps N                 Stop after N optimizer updates
  --prepare-only            Generate configs without starting training
  --master-port PORT        Distributed launch port (default: 29731)
  -h, --help                Show this help
EOF
}

ENV_NAME=""
GPU_IDS=""
DATA_ROOT=""
VISION_MODEL=""
PREDICTOR_MODEL=""
OUTPUT_DIR=""
MASTER_PORT="29731"
BACKBONE=""
EXTRA_ARGS=()
PREPARE_ONLY=false

# Keep existing positional commands working.
if [[ $# -gt 0 && "$1" != -* ]]; then
  if [[ $# -lt 5 || $# -gt 7 ]]; then
    usage >&2
    exit 2
  fi
  ENV_NAME="${1}"
  GPU_IDS="${2}"
  DATA_ROOT="${3}"
  VISION_MODEL="${4}"
  PREDICTOR_MODEL="${5}"
  OUTPUT_DIR="${6:-}"
  MASTER_PORT="${7:-29731}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --prepare-only) PREPARE_ONLY=true; shift; continue ;;
      --env|--gpus|--data-root|--vision-model|--predictor-model|--output-dir|--master-port|--backbone|--backbone-code-dir|--backbone-checkpoint|--resolution|--batch-size|--group-size|--steps)
        if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
          echo "Missing value for $1" >&2
          exit 2
        fi
        ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    case "$1" in
      --backbone) BACKBONE="$2"; EXTRA_ARGS+=("$1" "$2") ;;
      --backbone-code-dir|--backbone-checkpoint|--resolution|--batch-size|--group-size|--steps) EXTRA_ARGS+=("$1" "$2") ;;
      --env) ENV_NAME="$2" ;;
      --gpus) GPU_IDS="$2" ;;
      --data-root) DATA_ROOT="$2" ;;
      --vision-model) VISION_MODEL="$2" ;;
      --predictor-model) PREDICTOR_MODEL="$2" ;;
      --output-dir) OUTPUT_DIR="$2" ;;
      --master-port) MASTER_PORT="$2" ;;
    esac
    shift 2
  done
fi

[[ -n "${ENV_NAME}" ]] || { echo "Missing required option: --env" >&2; exit 2; }
[[ -n "${GPU_IDS}" ]] || { echo "Missing required option: --gpus" >&2; exit 2; }
[[ -n "${DATA_ROOT}" ]] || { echo "Missing required option: --data-root" >&2; exit 2; }
[[ -n "${VISION_MODEL}" || -n "${BACKBONE}" ]] || { echo "Provide --vision-model or --backbone" >&2; exit 2; }
[[ -n "${PREDICTOR_MODEL}" ]] || { echo "Missing required option: --predictor-model" >&2; exit 2; }

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/configurable_encoder/${ENV_NAME}/${RUN_NAME}}"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PWD/$OUTPUT_DIR"
fi
CONFIG_DIR="${OUTPUT_DIR}/launch_configs"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_LIST[@]}"

if [[ -v MODEL_SEED || -v DATA_SEED ]]; then
  echo "Configurable Encoder uses one SEED. Unset MODEL_SEED and DATA_SEED before launching." >&2
  exit 2
fi
VISION_ARGS=()
[[ -z "${VISION_MODEL}" ]] || VISION_ARGS=(--vision-model "${VISION_MODEL}")
SEED_ARGS=()
if [[ -v SEED ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

mkdir -p "${OUTPUT_DIR}"
echo "Configurable Encoder output: ${OUTPUT_DIR}"

python "${ROOT}/tools/render_configs.py" \
  --package-root "${ROOT}" \
  --env "${ENV_NAME}" \
  --data-root "${DATA_ROOT}" \
  "${VISION_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --llama-model "${PREDICTOR_MODEL}" \
  "${SEED_ARGS[@]}" \
  --world-size "${WORLD_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  --config-dir "${CONFIG_DIR}"

if [[ "${PREPARE_ONLY}" == true ]]; then
  echo "Configs ready: ${CONFIG_DIR}"
  exit 0
fi

export PYTHONPATH="${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export CONFIGWM_DETERMINISTIC_DINO_POS_INTERP=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd "${ROOT}"
exec python -m torch.distributed.run \
  --standalone --nproc_per_node="${WORLD_SIZE}" --master_port="${MASTER_PORT}" \
  -m src.train.run \
  --where_am_i local \
  --data_config_path "${CONFIG_DIR}/data_config.yaml" \
  --model_config_path "${CONFIG_DIR}/model_config.yaml" \
  --trainer_config_path "${CONFIG_DIR}/trainer_config.yaml" \
  --output_dir "${OUTPUT_DIR}" \
  --run_description "configurable-encoder-${ENV_NAME}"
