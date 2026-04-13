import numpy as np
from PIL import Image

from torchvision import transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor()
])


def transform(examples, image_size=256):

    images = examples["image"]
    if isinstance(images, (list, tuple)):
        examples["image"] = [transform_train(
            image.convert("RGB")) for image in images]
    else:
        examples["image"] = transform_train(images.convert("RGB"))
    return examples
