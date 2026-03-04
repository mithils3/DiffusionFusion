#!/bin/bash
#SBATCH -J VAE_features
#SBATCH -A betw-dtai-gh
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
module load cuda/12.6.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT
git pull
IMAGENET_PATH="timm/mini-imagenet"
OUTPUT_DIR="/work/nvme/betw/msalunkhe/data/mini-imagenet"
# Avoid thread explosion across DDP ranks + DataLoader workers.
torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
feature_extraction/dinov3.py \
--data-path ${IMAGENET_PATH}  \
--features-path ${OUTPUT_DIR} \
--num-workers 4 \
--hf-dataset-name imagenet256_dinov3_features 
