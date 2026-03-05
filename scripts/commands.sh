srun -A betw-dtai-gh --time=02:00:00 --nodes=1 --ntasks-per-node=32 \
--partition=ghx4-interactive --gpus=2 --mem=256g --pty /bin/bash

apptainer run --nv \
    --bind /projects \
    /sw/user/NGC_containers/pytorch_25.08-py3.sif