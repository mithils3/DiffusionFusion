#!/bin/bash
#SBATCH -J DiT_FID
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
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
IMAGENET_PATH="/work/nvme/betw/msalunkhe/data/mini-imagenet/"
OUTPUT_DIR="/work/nvme/betw/msalunkhe/outputs/jit_base_patch_size_2_dino_latent_naive"
rm -rf  ${OUTPUT_DIR}
TORCHDYNAMO_DISABLE=1 torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/2-4C \
--proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 \
    --noise_scale 0.7 \
--batch_size 128 --blr 5e-5 \
--epochs 900 --warmup_epochs 5 \
--gen_bsz  128 --num_images 10000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval --num_workers 2 --class_num 100
