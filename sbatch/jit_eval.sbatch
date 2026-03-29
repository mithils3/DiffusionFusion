#!/bin/bash
#SBATCH -J JiT_eval
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Requires a checkpoint saved after the native velocity-prediction migration.

module load python/3.11.9
module load cuda/12.9.0

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT
git pull

IMAGENET_PATH="${IMAGENET_PATH:-/work/hdd/bgnp/msalunkhe/data/imagenet/}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/jit_base_patch_size_2_dino_latent_large_hidden_size}"
RESUME_DIR="${RESUME_DIR:-${OUTPUT_DIR}}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"

MODEL="${MODEL:-JiT-B/2-4C}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GEN_BSZ="${GEN_BSZ:-64}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
CFG="${CFG:-2.9}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"
CLASS_NUM="${CLASS_NUM:-1000}"

if [[ ! -f "${RESUME_DIR}/checkpoint-last.pth" ]]; then
  echo "JiT checkpoint not found: ${RESUME_DIR}/checkpoint-last.pth" >&2
  exit 1
fi

if [[ ! -f "${DECODER_CHECKPOINT}" ]]; then
  echo "Decoder checkpoint not found: ${DECODER_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=auto \
  --nnodes=1 \
  --node_rank=0 \
  main_jit.py \
  --model "${MODEL}" \
  --proj_dropout 0.0 \
  --P_mean -0.8 \
  --P_std 0.8 \
  --noise_scale 1.0 \
  --batch_size "${BATCH_SIZE}" \
  --gen_bsz "${GEN_BSZ}" \
  --num_images "${NUM_IMAGES}" \
  --cfg "${CFG}" \
  --interval_min "${INTERVAL_MIN}" \
  --interval_max "${INTERVAL_MAX}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${RESUME_DIR}" \
  --data_path "${IMAGENET_PATH}" \
  --evaluate_gen \
  --decoder_checkpoint "${DECODER_CHECKPOINT}" \
  --decoder_checkpoint_key model_ema \
  --class_num "${CLASS_NUM}"
