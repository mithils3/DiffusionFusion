#!/bin/bash
#SBATCH -J JiT_decoder_attr
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-user=mithils3@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load python/3.11.9
module load cuda/13.1.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion
git pull

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1

echo "===== Decoder attribution preflight ====="
echo "date=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$(which python)"
module list
nvidia-smi
echo "===== End preflight ====="

# Keep this checkpoint/key aligned with sbatch/jit_training.sbatch.
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"
DECODER_CHECKPOINT_KEY="${DECODER_CHECKPOINT_KEY:-model_ema}"

FEATURE_ROOT="${FEATURE_ROOT:-/work/hdd/bgnp/msalunkhe/data/imagenet}"
IMAGE_DATA_PATH="${IMAGE_DATA_PATH:-ILSVRC/imagenet-1k}"
SPLIT="${SPLIT:-validation}"
LATENT_DIR_NAME="${LATENT_DIR_NAME:-imagenet256_latents}"
DINO_DIR_NAME="${DINO_DIR_NAME:-imagenet256_dinov3_features}"

NUM_IMAGES="${NUM_IMAGES:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET="${TARGET:-mean}"
SAVE_NPZ="${SAVE_NPZ:-0}"
HEATMAP_PERCENTILE="${HEATMAP_PERCENTILE:-98}"
HEATMAP_GAMMA="${HEATMAP_GAMMA:-0.65}"

HF_REPO_ID="${HF_REPO_ID:-Mithilss/decoder_patch_attribution}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-runs}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"

CHECKPOINT_NAME="$(basename "$(dirname "${DECODER_CHECKPOINT}")")"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/${CHECKPOINT_NAME}_patch_attribution_${RUN_TAG}}"

if [[ ! -f "${DECODER_CHECKPOINT}" ]]; then
  echo "Decoder checkpoint not found: ${DECODER_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

OPTIONAL_ARGS=()
if [[ -n "${PATCHES:-}" ]]; then
  IFS=';' read -r -a PATCH_ARRAY <<< "${PATCHES}"
  for PATCH in "${PATCH_ARRAY[@]}"; do
    if [[ -n "${PATCH}" ]]; then
      OPTIONAL_ARGS+=(--patch "${PATCH}")
    fi
  done
fi
if [[ "${SAVE_NPZ}" == "1" || "${SAVE_NPZ}" == "true" ]]; then
  OPTIONAL_ARGS+=(--save-npz)
fi
if [[ "${UPLOAD_TO_HF}" == "1" || "${UPLOAD_TO_HF}" == "true" ]]; then
  OPTIONAL_ARGS+=(
    --upload-to-hf
    --hf-repo-id "${HF_REPO_ID}"
    --hf-repo-type "${HF_REPO_TYPE}"
    --hf-path-in-repo "${HF_PATH_IN_REPO}/${RUN_TAG}"
  )
fi

OMP_NUM_THREADS=16 python JiT/eval/decoder_patch_attribution.py \
  --checkpoint "${DECODER_CHECKPOINT}" \
  --feature-root "${FEATURE_ROOT}" \
  --image-data-path "${IMAGE_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --latent-dir-name "${LATENT_DIR_NAME}" \
  --dino-dir-name "${DINO_DIR_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-images "${NUM_IMAGES}" \
  --target "${TARGET}" \
  --heatmap-percentile "${HEATMAP_PERCENTILE}" \
  --heatmap-gamma "${HEATMAP_GAMMA}" \
  --checkpoint-key "${DECODER_CHECKPOINT_KEY}" \
  "${OPTIONAL_ARGS[@]}"
