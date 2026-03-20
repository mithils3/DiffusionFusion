import math

import torch
import torch.nn.functional as F
from torch import nn


class DiscriminatorAugment(nn.Module):
    """StyleGAN-T/RAE discriminator augmentation."""

    def __init__(
        self,
        prob: float = 1.0,
        cutout: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        horizontal_flip: bool = False,
    ) -> None:
        super().__init__()
        self.grids: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout
        self.img_channels = -1
        self.last_blur_radius = -1
        self.last_blur_kernel_h = None
        self.last_blur_kernel_w = None

    def get_grids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (batch_size, height, width)
        if key not in self.grids:
            self.grids[key] = torch.meshgrid(
                torch.arange(batch_size, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self.grids[key]

    def aug(self, images: torch.Tensor, warmup_blur_schedule: float = 0.0) -> torch.Tensor:
        if images.dtype != torch.float32:
            images = images.float()

        if warmup_blur_schedule > 0:
            self.img_channels = images.shape[1]
            sigma0 = (images.shape[-2] * 0.5) ** 0.5
            sigma = sigma0 * warmup_blur_schedule
            blur_radius = math.floor(sigma * 3)
            if blur_radius >= 1:
                if self.last_blur_radius != blur_radius:
                    self.last_blur_radius = blur_radius
                    gaussian = torch.arange(-blur_radius, blur_radius + 1, dtype=torch.float32, device=images.device)
                    gaussian = gaussian.mul_(1 / sigma).square_().neg_().exp2_()
                    gaussian.div_(gaussian.sum())
                    kernel_h = gaussian.view(1, 1, 2 * blur_radius + 1, 1)
                    kernel_w = gaussian.view(1, 1, 1, 2 * blur_radius + 1)
                    self.last_blur_kernel_h = kernel_h.repeat(self.img_channels, 1, 1, 1).contiguous()
                    self.last_blur_kernel_w = kernel_w.repeat(self.img_channels, 1, 1, 1).contiguous()
                images = F.pad(images, [blur_radius, blur_radius, blur_radius, blur_radius], mode="reflect")
                images = F.conv2d(images, self.last_blur_kernel_h, bias=None, groups=self.img_channels)
                images = F.conv2d(images, self.last_blur_kernel_w, bias=None, groups=self.img_channels)

        if self.prob < 1e-6:
            return images

        trans, color, cut = (torch.rand(3, device=images.device) <= self.prob).tolist()
        if not (trans or color or cut):
            return images

        batch_size, _, raw_h, raw_w = images.shape
        rand01 = torch.rand(7, batch_size, 1, 1, device=images.device)

        if trans:
            ratio = 0.125
            delta_h = round(raw_h * ratio)
            delta_w = round(raw_w * ratio)
            translation_h = rand01[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
            translation_w = rand01[1].mul(delta_w + delta_w + 1).floor().long() - delta_w
            grid_b, grid_h, grid_w = self.get_grids(batch_size, raw_h, raw_w, images.device)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
            padded = F.pad(images, [1, 1, 1, 1, 0, 0, 0, 0])
            images = padded.permute(0, 2, 3, 1).contiguous()[grid_b, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()

        if color:
            images = images.add(rand01[2].unsqueeze(-1).sub(0.5))
            image_mean = images.mean(dim=1, keepdim=True)
            images = images.sub(image_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(image_mean)
            image_mean = images.mean(dim=(1, 2, 3), keepdim=True)
            images = images.sub(image_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(image_mean)

        if self.using_cutout and cut:
            ratio = self.cutout
            cutout_h = round(raw_h * ratio)
            cutout_w = round(raw_w * ratio)
            offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()
            grid_b, grid_h, grid_w = self.get_grids(batch_size, cutout_h, cutout_w, images.device)
            grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
            grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
            mask = torch.ones(batch_size, raw_h, raw_w, dtype=images.dtype, device=images.device)
            mask[grid_b, grid_h, grid_w] = 0
            images = images.mul(mask.unsqueeze(1))

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return images
        return self.aug(images)
