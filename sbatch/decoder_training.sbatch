#!/bin/bash
#SBATCH -J JiT_decoder
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
cd /u/msalunkhe/DiffusionFusion/JiT
git pull

IMAGE_DATA_PATH="ILSVRC/imagenet-1k"

FEATURE_ROOT="/work/hdd/bgnp/msalunkhe/data/imagenet"
OUTPUT_DIR="/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan"
mkdir -p "${OUTPUT_DIR}"

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=auto \
  --nnodes=1 \
  --node_rank=0 \
  main_decoder.py \
  --config decoder/default_config.yaml \
  --data_path "${FEATURE_ROOT}" \
  --image_data_path "${IMAGE_DATA_PATH}" \
  --latent_dir_name imagenet256_latents \
  --dino_dir_name imagenet256_dinov3_features \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${OUTPUT_DIR}" \
  --batch_size 64 \
  --epochs 16 \
  --eval_freq 2 \
  --num_images 2048
