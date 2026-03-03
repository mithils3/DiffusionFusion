srun -A betw-dtai-gh --time=01:00:00 --nodes=1 --ntasks-per-node=16 \
--partition=ghx4-interactive --gpus=1 --mem=256g --pty /bin/bash

apptainer run --nv \
    --bind /projects \
    /sw/user/NGC_containers/pytorch_25.08-py3.sif