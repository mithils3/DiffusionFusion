from torch import nn
from model_jit import BottleneckPatchEmbed
import torch 

class Decoder(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, bottleneck_dim: int, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.latent_tokenizer = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        num_patches = self.latent_tokenizer.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=False)
    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs