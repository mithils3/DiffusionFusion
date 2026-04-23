#!/bin/bash
#SBATCH -J JiT_80ep
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
module load cuda/13.1.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1

echo "===== Slurm/CUDA preflight ====="
echo "date=$(date)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "node_list=${SLURM_JOB_NODELIST:-unset}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "python=$(which python)"
echo "torchrun=$(which torchrun)"
module list
nvidia-smi

python - <<'PY'
import os
import sys
import torch

print(f"PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
try:
    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
except Exception as exc:
    sys.exit(f"CUDA preflight failed before torchrun: {type(exc).__name__}: {exc}")

print(f"torch.cuda.is_available()={available}")
print(f"torch.cuda.device_count()={device_count}")
if device_count < 1:
    sys.exit("CUDA preflight failed before torchrun: no CUDA devices are visible to PyTorch")
for idx in range(device_count):
    print(f"cuda:{idx}={torch.cuda.get_device_name(idx)}")
PY
echo "===== End preflight ====="


IMAGENET_PATH="/work/hdd/bgnp/msalunkhe/data/imagenet/"
OUTPUT_DIR="/work/nvme/bgnp/msalunkhe/outputs/jit_dual_patch_size_2_constant_80ep"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-jit_dual_patch_size_2_constant_80ep}"
FID_STATS_PATH="${FID_STATS_PATH:-/u/msalunkhe/DiffusionFusion/JiT/fid_stats/jit_in256_stats.npz}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
EVAL_FREQ="${EVAL_FREQ:-20}"
EPOCHS="${EPOCHS:-80}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_ITER="${ACCUM_ITER:-2}"
mkdir -p "${OUTPUT_DIR}"
OMP_NUM_THREADS=16 torchrun \
--nproc_per_node="${NPROC_PER_NODE}" \
--nnodes=1 \
--node_rank=0 \
main_jit.py \
--model JiT-Dual-B/2-4C-896 \
--proj_dropout 0.0 \
--P_mean -0.8 \
--P_std 0.8 \
--noise_scale 1.0 \
--batch_size "${BATCH_SIZE}" \
--accum_iter "${ACCUM_ITER}" \
--blr 5e-5 \
--epochs "${EPOCHS}" \
--warmup_epochs 5 \
--lr_schedule "${LR_SCHEDULE}" \
--gen_bsz 128 \
--num_images "${NUM_IMAGES}" \
--cfg 2.9 \
--interval_min 0.1 \
--interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--wandb_run_name "${WANDB_RUN_NAME}" \
--data_path ${IMAGENET_PATH} \
--online_eval \
--fid_stats_path "${FID_STATS_PATH}" \
--decoder_checkpoint /work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth \
--decoder_checkpoint_key model_ema \
--class_num 1000 \
--eval_freq "${EVAL_FREQ}"
