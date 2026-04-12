#!/bin/bash
#SBATCH -J SDXL_VAE_eval
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
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

IMAGE_DATA_PATH="ILSVRC/imagenet-1k"
OUTPUT_DIR="/work/nvme/bgnp/msalunkhe/outputs/sdxl_vae_eval"
mkdir -p "${OUTPUT_DIR}"

OMP_NUM_THREADS=16 torchrun \
  --nproc_per_node=auto \
  --nnodes=1 \
  --node_rank=0 \
  -m custom.eval.eval_sdxl_vae \
  --data-path "${IMAGE_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split validation \
  --image-size 224 \
  --num-images 50000   \
  --batch-size 128 \
  --fid-batch-size 256 \
  --num-workers 4
