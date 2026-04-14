#!/bin/bash
#SBATCH -J JiT_LF_DINO
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
#SBATCH --mail-user=mithils3@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load python/3.11.9
module load cuda/12.9.0

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT

IMAGENET_PATH="${IMAGENET_PATH:-/work/hdd/bgnp/msalunkhe/data/imagenet/}"
MODEL="${MODEL:-JiT-B/2-4C}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/jit_latent_forcing_base_dino_first_100ep}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-jit_latent_forcing_base_dino_first_100ep}"
FID_STATS_PATH="${FID_STATS_PATH:-/u/msalunkhe/DiffusionFusion/JiT/fid_stats/jit_in256_stats.npz}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
EVAL_FREQ="${EVAL_FREQ:-20}"
EPOCHS="${EPOCHS:-100}"
CFG="${CFG:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_ITER="${ACCUM_ITER:-4}"
GEN_BSZ="${GEN_BSZ:-64}"

mkdir -p "${OUTPUT_DIR}"

# Paper-style latent forcing defaults adapted to JiT's dual-stream setup:
# - DINO denoises first as the semantic scratchpad branch.
# - SDXL latent denoises second as the rendering/detail branch.
# - 25/25 Heun split, 100 epochs, constant LR, effective batch 1024.
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
  --accum_iter "${ACCUM_ITER}" \
  --blr 5e-5 \
  --epochs "${EPOCHS}" \
  --warmup_epochs 5 \
  --lr_schedule constant \
  --weight_decay 0.0 \
  --ema_decay1 0.9999 \
  --ema_decay2 0.9996 \
  --sampling_method heun \
  --num_sampling_steps 50 \
  --cfg "${CFG}" \
  --interval_min 0.0 \
  --interval_max 1.0 \
  --latent_forcing \
  --lf_order dino_first \
  --lf_phase_prob 0.4 \
  --lf_dino_mean -1.2 \
  --lf_dino_std 1.0 \
  --lf_latent_mean -0.8 \
  --lf_latent_std 0.8 \
  --lf_context_beta 0.25 \
  --lf_first_phase_steps 25 \
  --latent_loss_weight 1.0 \
  --dino_loss_weight 0.333 \
  --gen_bsz "${GEN_BSZ}" \
  --num_images "${NUM_IMAGES}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --data_path "${IMAGENET_PATH}" \
  --online_eval \
  --fid_stats_path "${FID_STATS_PATH}" \
  --decoder_checkpoint "${DECODER_CHECKPOINT}" \
  --decoder_checkpoint_key model_ema \
  --class_num 1000 \
  --eval_freq "${EVAL_FREQ}"
