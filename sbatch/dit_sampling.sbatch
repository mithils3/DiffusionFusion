#!/bin/bash
#SBATCH -J DiT_FID
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail


module load python/3.11.9
module load cuda/12.6.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/DiT

# Use all GPUs on the node
torchrun --nnodes=1 --nproc_per_node=auto \
  sample_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 50000 \
  --sample-dir /projects/betw/msalunkhe/DiT_generations
