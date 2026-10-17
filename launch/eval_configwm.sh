#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m src.eval.run "$@"
