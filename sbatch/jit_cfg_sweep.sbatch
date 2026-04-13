#!/bin/bash
#SBATCH -J JiT_cfg_sweep
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
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

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/jit_base_patch_size_2_higher_lr}"
RESUME_DIR="${RESUME_DIR:-${TRAIN_OUTPUT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/cfg_sweep_vae_eval}"
FID_STATS_PATH="${FID_STATS_PATH:-/u/msalunkhe/DiffusionFusion/JiT/fid_stats/jit_in256_stats.npz}"
VAE_PRETRAINED_PATH="${VAE_PRETRAINED_PATH:-stabilityai/sdxl-vae}"

GEN_BSZ="${GEN_BSZ:-128}"
SWEEP_NUM_IMAGES="${SWEEP_NUM_IMAGES:-5000}"
FINAL_NUM_IMAGES="${FINAL_NUM_IMAGES:-50000}"
BASE_CFG="${BASE_CFG:-}"
CFG_VALUES="${CFG_VALUES:-}"
COARSE_OFFSETS="${COARSE_OFFSETS:--0.8,-0.4,0.0,0.4,0.8}"
REFINE_STEP="${REFINE_STEP:-0.1}"
REFINE_COUNT="${REFINE_COUNT:-2}"
CFG_MIN="${CFG_MIN:-1.0}"
CFG_MAX="${CFG_MAX:-4.5}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"

OPTIONAL_ARGS=()

if [[ -n "${BASE_CFG}" ]]; then
  OPTIONAL_ARGS+=(--base-cfg "${BASE_CFG}")
fi
if [[ -n "${CFG_VALUES}" ]]; then
  OPTIONAL_ARGS+=(--cfg-values "${CFG_VALUES}")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--local-files-only)
fi
if [[ "${KEEP_IMAGES:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--keep-images)
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--force)
fi
if [[ "${SKIP_FINAL:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--skip-final)
fi

python JiT/eval/cfg_sweep.py \
  --resume-dir "${RESUME_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-key "${CHECKPOINT_KEY}" \
  --fid-stats-path "${FID_STATS_PATH}" \
  --vae-pretrained-path "${VAE_PRETRAINED_PATH}" \
  --gen-bsz "${GEN_BSZ}" \
  --sweep-num-images "${SWEEP_NUM_IMAGES}" \
  --final-num-images "${FINAL_NUM_IMAGES}" \
  --coarse-offsets="${COARSE_OFFSETS}" \
  --refine-step "${REFINE_STEP}" \
  --refine-count "${REFINE_COUNT}" \
  --cfg-min "${CFG_MIN}" \
  --cfg-max "${CFG_MAX}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  "${OPTIONAL_ARGS[@]}"
