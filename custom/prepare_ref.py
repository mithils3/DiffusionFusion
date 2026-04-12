import os
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from custom.util.image_transforms import build_center_crop_normalize_transform


class HFDatasetAdapter(Dataset):
    def __init__(self, dataset, image_column, transform):
        self.dataset = dataset
        self.image_column = image_column
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.image_column not in sample:
            raise KeyError(
                f'Image column "{self.image_column}" not found in HF sample keys: {list(sample.keys())}'
            )

        image = sample[self.image_column]
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise TypeError(
                f'Unsupported HF image type: {type(image)}. '
                'Expected PIL.Image.Image or numpy.ndarray.'
            )

        return self.transform(pil_image.convert("RGB")), -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='ImageNet root directory (local) or HF dataset ID (with --use_hf_dataset)')
    parser.add_argument('--output_path', type=str, default='imagenet-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Resolution to center-crop and resize')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size used while exporting PNG files')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--use_hf_dataset', action='store_true',
                        help='Load data via datasets.load_dataset instead of torchvision ImageFolder')
    parser.add_argument('--hf_split', type=str, default='train',
                        help='HF split name used when --use_hf_dataset is set')
    parser.add_argument('--hf_image_column', type=str, default='image',
                        help='HF column containing image objects')
    parser.add_argument('--hf_config_name', type=str, default=None,
                        help='Optional HF dataset config name (passed as load_dataset(name=...))')
    parser.add_argument('--hf_cache_dir', type=str, default=None,
                        help='Optional HF cache directory')
    args = parser.parse_args()

    transform_train = build_center_crop_normalize_transform(args.img_size)

    if args.use_hf_dataset:
        from datasets import load_dataset
        hf_dataset = load_dataset(
            args.data_path,
            name=args.hf_config_name,
            split=args.hf_split,
            cache_dir=args.hf_cache_dir,
        )
        dataset_train = HFDatasetAdapter(
            hf_dataset,
            image_column=args.hf_image_column,
            transform=transform_train,
        )
    else:
        imagenet_train_dir = os.path.join(args.data_path, 'train')
        if not os.path.isdir(imagenet_train_dir):
            raise FileNotFoundError(
                f'Expected local ImageFolder directory: "{imagenet_train_dir}". '
                'If you want a Hugging Face dataset, pass --use_hf_dataset.'
            )
        dataset_train = datasets.ImageFolder(
            imagenet_train_dir,
            transform=transform_train
        )

    data_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )

    os.makedirs(args.output_path, exist_ok=True)

    to_pil = transforms.ToPILImage()
    global_idx = 0

    from tqdm import tqdm
    for batch_images, batch_labels in tqdm(data_loader):
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]

            pil_img = to_pil(img_tensor)
            out_path = os.path.join(
                args.output_path,
                f"transformed_{global_idx:08d}.png"
            )
            pil_img.save(out_path, format='PNG', compress_level=0)
            global_idx += 1

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")


if __name__ == "__main__":
    main()
