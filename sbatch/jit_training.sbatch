#!/bin/bash
#SBATCH -J DiT_FID
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
module load cuda/12.6.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT
git pull
IMAGENET_PATH="/work/nvme/betw/msalunkhe/data/imagenet256_latents/"
OUTPUT_DIR="/work/nvme/betw/msalunkhe/outputs/jit_large_patch_size_4"
OMP_NUM_THREADS=32 torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-L/4-4C \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 32 --noise_scale 1.3 \
--batch_size 256 --blr 5e-5 \
--epochs 300 --warmup_epochs 5 \
--gen_bsz  128 --num_images 5000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval --num_workers 8
