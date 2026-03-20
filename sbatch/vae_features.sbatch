#!/bin/bash
#SBATCH -J VAE_features
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
cd /u/msalunkhe/DiffusionFusion/JiT
IMAGENET_PATH="ILSVRC/imagenet-1k"
OUTPUT_DIR="/work/hdd/bgnp/msalunkhe/data/imagenet"
SPLIT="${SPLIT:-validation}"
# Avoid thread explosion across DDP ranks + DataLoader workers.
torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
feature_extraction/vae.py \
--data-path ${IMAGENET_PATH}  \
--features-path ${OUTPUT_DIR} \
--split "${SPLIT}" \
--num-workers 4
