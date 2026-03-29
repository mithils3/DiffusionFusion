# Requires a checkpoint saved after the native velocity-prediction migration.
torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16  \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
