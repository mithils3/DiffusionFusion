import torch
import torch.distributed as dist
import os
import shutil
from JiT.main_jit import collate_fn
import timm
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets import Array3D, Dataset, Features, Value, concatenate_datasets, load_dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import numpy as np
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size(
    ) == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        tmp_shard_dir = os.path.join(args.features_path, "_tmp_hf_dino_shards")
        if os.path.exists(tmp_shard_dir):
            shutil.rmtree(tmp_shard_dir)
        os.makedirs(tmp_shard_dir, exist_ok=True)
    dist.barrier()
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        features_only=True,
    )
    model = model.eval()
    local_batch_size = args.global_batch_size // dist.get_world_size()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    dataset = load_dataset(args.data_path, split="train")
    dataset = dataset.with_transform(
        lambda examples: {
            "image": [transforms(image.convert("RGB")) for image in examples["image"]],
            "label": examples["label"],
        }
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=4,
    )

    tmp_shard_dir = os.path.join(args.features_path, "_tmp_hf_latent_shards")
    rank_shard_path = os.path.join(tmp_shard_dir, f"rank_{rank:05d}.arrow")
    patches = 256 // 16
    hf_features = Features(
            {
                "feature": Array3D(shape=(args.hidden_size, patches, patches), dtype="float32"),
                "label": Value("int64"),
                # used to restore deterministic global ordering after rank-wise writes
                "sample_id": Value("int64"),
            }
        )
    shard_writer = ArrowWriter(path=rank_shard_path, features=hf_features)

    train_steps = 0
    for batch in tqdm(loader, total=len(loader), desc=f"Rank {rank}"):
        x = batch["image"]
        y = batch["label"]

        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(x)[-1]  # take last feature map (B, C, H, W)
        output = output.detach().cpu().numpy()
        y = y.detach().cpu().numpy()    # (bs,)
        for i in range(x.shape[0]):
            # save_num = NUM_SAMPLES * rank + train_steps * local_batch_size + i
            sample_id = train_steps * args.global_batch_size + dist.get_world_size() * \
                i + rank
            shard_writer.write(
                {
                    "feature": x[i].astype(np.float32, copy=False),
                    "label": int(y[i]),
                    "sample_id": int(sample_id),
                }
            )
