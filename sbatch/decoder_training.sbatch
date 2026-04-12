#!/bin/bash
#SBATCH -J custom_decoder_latent_dino
#SBATCH -A bgnp-dtai-gh
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

IMAGE_DATA_PATH="ILSVRC/imagenet-1k"

FEATURE_ROOT="/work/hdd/bgnp/msalunkhe/data/imagenet"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/custom_decoder_small_gan_lower_noise}"
FID_STATS_PATH="${FID_STATS_PATH:-/work/nvme/betw/msalunkhe/data/custom_in256_stats.npz}"
LATENT_DIR_NAME="${LATENT_DIR_NAME:-imagenet256_latents}"
DINO_DIR_NAME="${DINO_DIR_NAME:-imagenet256_dinov3_features}"
BATCH_SIZE="${BATCH_SIZE:-72}"
EPOCHS="${EPOCHS:-16}"
EVAL_FREQ="${EVAL_FREQ:-2}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-custom_decoder_latent_dino}"

# Optional paper-inspired ablations. Leave unset to use config defaults.
NOISE_TAU="${NOISE_TAU:-}"
DISC_START="${DISC_START:-}"
DISC_UPD_START="${DISC_UPD_START:-}"
DISC_FREEZE_BACKBONE="${DISC_FREEZE_BACKBONE:-}"
DECODER_HIDDEN_SIZE="${DECODER_HIDDEN_SIZE:-}"
DECODER_DEPTH="${DECODER_DEPTH:-}"
mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${NOISE_TAU}" ]]; then
  EXTRA_ARGS+=(--decoder_noise_tau "${NOISE_TAU}")
fi
if [[ -n "${DISC_START}" ]]; then
  EXTRA_ARGS+=(--decoder_disc_start "${DISC_START}")
fi
if [[ -n "${DISC_UPD_START}" ]]; then
  EXTRA_ARGS+=(--decoder_disc_upd_start "${DISC_UPD_START}")
fi
if [[ -n "${DECODER_HIDDEN_SIZE}" ]]; then
  EXTRA_ARGS+=(--decoder_hidden_size "${DECODER_HIDDEN_SIZE}")
fi
if [[ -n "${DECODER_DEPTH}" ]]; then
  EXTRA_ARGS+=(--decoder_depth "${DECODER_DEPTH}")
fi
if [[ -n "${DISC_FREEZE_BACKBONE}" ]]; then
  if [[ "${DISC_FREEZE_BACKBONE}" == "1" || "${DISC_FREEZE_BACKBONE}" == "true" ]]; then
    EXTRA_ARGS+=(--decoder_disc_freeze_backbone)
  elif [[ "${DISC_FREEZE_BACKBONE}" == "0" || "${DISC_FREEZE_BACKBONE}" == "false" ]]; then
    EXTRA_ARGS+=(--no_decoder_disc_freeze_backbone)
  else
    echo "DISC_FREEZE_BACKBONE must be one of: 1, 0, true, false" >&2
    exit 1
  fi
fi

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=auto \
  --nnodes=1 \
  --node_rank=0 \
  -m custom.main_decoder \
  --config decoder/default_config.yaml \
  --data_path "${FEATURE_ROOT}" \
  --image_data_path "${IMAGE_DATA_PATH}" \
  --latent_dir_name "${LATENT_DIR_NAME}" \
  --dino_dir_name "${DINO_DIR_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --decoder_eval_fid_stats "${FID_STATS_PATH}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --eval_freq "${EVAL_FREQ}" \
  --num_images "${NUM_IMAGES}" \
  "${EXTRA_ARGS[@]}"
