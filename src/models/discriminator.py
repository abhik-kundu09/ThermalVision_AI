"""
Pix2Pix 70x70 PatchGAN Discriminator for Landsat Translation.

Architecture:
- Evaluates N x N local image patches as real or fake rather than a global scalar.
- Input: Concatenated (Thermal IR, RGB) tensor [B, 4, 256, 256]
- Output: Patch probability logits [B, 1, 30, 30]
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 receptive field PatchGAN Discriminator.
    """

    def __init__(self, in_channels: int = 4, num_filters: int = 64):
        super().__init__()

        def disc_block(in_c: int, out_c: int, normalize: bool = True, stride: int = 2):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: [B, 4, 256, 256] -> [B, 64, 128, 128]
            *disc_block(in_channels, num_filters, normalize=False, stride=2),
            # -> [B, 128, 64, 64]
            *disc_block(num_filters, num_filters * 2, normalize=True, stride=2),
            # -> [B, 256, 32, 32]
            *disc_block(num_filters * 2, num_filters * 4, normalize=True, stride=2),
            # -> [B, 512, 31, 31]
            *disc_block(num_filters * 4, num_filters * 8, normalize=True, stride=1),
            # Final 1-channel prediction map -> [B, 1, 30, 30]
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights according to the original Pix2Pix paper (Normal(0, 0.02))."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, ir_input: torch.Tensor, rgb_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_input: Thermal IR tensor [B, 1, 256, 256]
            rgb_target: Target or Fake RGB tensor [B, 3, 256, 256]

        Returns:
            logits: PatchGAN prediction logits [B, 1, 30, 30]
        """
        x = torch.cat([ir_input, rgb_target], dim=1)
        return self.model(x)
