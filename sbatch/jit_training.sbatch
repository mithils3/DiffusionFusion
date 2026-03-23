#!/bin/bash
#SBATCH -J DiT_FID
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
IMAGENET_PATH="/work/hdd/bgnp/msalunkhe/data/imagenet/"
OUTPUT_DIR="/work/nvme/bgnp/msalunkhe/outputs/jit_base_patch_size_2_dino_latent_large_hidden_size"
mkdir -p "${OUTPUT_DIR}"
OMP_NUM_THREADS=16 torchrun \
--nproc_per_node=auto \
--nnodes=1 \
--node_rank=0 \
main_jit.py \
--model JiT-B/2-4C \
--proj_dropout 0.0 \
--P_mean -0.8 \
--P_std 0.8 \
--noise_scale 1.0 \
--batch_size 96 \
--blr 5e-5 \
--epochs 100 \
--warmup_epochs 5 \
--gen_bsz 64 \
--num_images 5000 \
--cfg 2.9 \
--interval_min 0.1 \
--interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--online_eval \
--class_num 1000 \
--eval_freq 25
