#!/bin/bash
#SBATCH -J JiT_decoder_eval
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

module load python/3.11.9
module load cuda/12.9.0

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion
git pull

CHECKPOINT="${CHECKPOINT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"
FEATURE_ROOT="${FEATURE_ROOT:-/work/hdd/bgnp/msalunkhe/data/imagenet}"
IMAGE_DATA_PATH="${IMAGE_DATA_PATH:-ILSVRC/imagenet-1k}"
SPLIT="${SPLIT:-validation}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-100}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-256}"
FID_NUM_WORKERS="${FID_NUM_WORKERS:-4}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-auto}"

CHECKPOINT_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/${CHECKPOINT_NAME}_eval}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Decoder checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  JiT/eval/eval_decoder.py \
  --checkpoint "${CHECKPOINT}" \
  --feature-root "${FEATURE_ROOT}" \
  --image-data-path "${IMAGE_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-images "${NUM_IMAGES}" \
  --checkpoint-key "${CHECKPOINT_KEY}" \
  --fid-batch-size "${FID_BATCH_SIZE}" \
  --fid-num-workers "${FID_NUM_WORKERS}"
