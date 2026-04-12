# Requires a CustomDiT checkpoint.
torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
-m custom.main_custom \
--model CustomDiT-B/2-4C  \
--latent_size 32 --noise_scale 1.0  \
--gen_bsz 256 --num_images 50000 --cfg 3.0 \
--sampling_method euler --num_sampling_steps 250 --timestep_shift 0.3 \
--interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
