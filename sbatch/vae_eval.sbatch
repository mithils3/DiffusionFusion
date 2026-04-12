#!/bin/bash
#SBATCH -J custom_vae_eval
#SBATCH -A betw-dtai-gh
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

module load python/3.11.9
module load cuda/12.9.0

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion
git pull

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/custom_base_patch_size_2_higher_lr}"
RESUME_DIR="${RESUME_DIR:-${TRAIN_OUTPUT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/vae_eval}"
FID_STATS_PATH="${FID_STATS_PATH:-/u/msalunkhe/DiffusionFusion/custom/fid_stats/custom_in256_stats.npz}"
VAE_PRETRAINED_PATH="${VAE_PRETRAINED_PATH:-stabilityai/sdxl-vae}"

GEN_BSZ="${GEN_BSZ:-128}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
CFG="${CFG:-2.9}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"

mkdir -p "${OUTPUT_DIR}"

LOCAL_FILES_FLAG=()
KEEP_IMAGES_FLAG=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_FLAG+=(--local-files-only)
fi
if [[ "${KEEP_IMAGES:-0}" == "1" ]]; then
  KEEP_IMAGES_FLAG+=(--keep-images)
fi

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=auto \
  --nnodes=1 \
  --node_rank=0 \
  -m custom.eval.vae_eval \
  --resume-dir "${RESUME_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fid-stats-path "${FID_STATS_PATH}" \
  --vae-pretrained-path "${VAE_PRETRAINED_PATH}" \
  --gen-bsz "${GEN_BSZ}" \
  --num-images "${NUM_IMAGES}" \
  --cfg "${CFG}" \
  --interval-min "${INTERVAL_MIN}" \
  --interval-max "${INTERVAL_MAX}" \
  "${LOCAL_FILES_FLAG[@]}" \
  "${KEEP_IMAGES_FLAG[@]}"
