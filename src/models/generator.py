"""
Pix2Pix U-Net 256 Generator for Landsat Thermal IR -> RGB Translation.

Architecture:
- 8 Downsampling Encoder blocks
- Bottleneck (1x1 feature map)
- 8 Upsampling Decoder blocks with Skip Connections (U-Net concatenation)
- Input: Single-channel Thermal IR [B, 1, 256, 256] in [-1.0, 1.0]
- Output: 3-channel Visible RGB [B, 3, 256, 256] in [-1.0, 1.0] with Tanh activation
"""

from typing import Optional
import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    """Encoder Downsampling Block: Conv2d -> (Norm) -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Decoder Upsampling Block: ConvTranspose2d -> Norm -> (Dropout) -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip_input], dim=1)


class UNetGenerator(nn.Module):
    """
    8-layer U-Net Generator for 256x256 image translation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, num_filters: int = 64):
        super().__init__()

        # Encoder (Downsampling)
        # Input: [B, 1, 256, 256]
        self.down1 = UNetDownBlock(in_channels, num_filters, normalize=False)        # -> [B, 64, 128, 128]
        self.down2 = UNetDownBlock(num_filters, num_filters * 2)                     # -> [B, 128, 64, 64]
        self.down3 = UNetDownBlock(num_filters * 2, num_filters * 4)                 # -> [B, 256, 32, 32]
        self.down4 = UNetDownBlock(num_filters * 4, num_filters * 8)                 # -> [B, 512, 16, 16]
        self.down5 = UNetDownBlock(num_filters * 8, num_filters * 8)                 # -> [B, 512, 8, 8]
        self.down6 = UNetDownBlock(num_filters * 8, num_filters * 8)                 # -> [B, 512, 4, 4]
        self.down7 = UNetDownBlock(num_filters * 8, num_filters * 8)                 # -> [B, 512, 2, 2]
        self.down8 = UNetDownBlock(num_filters * 8, num_filters * 8, normalize=False)# -> [B, 512, 1, 1]

        # Decoder (Upsampling with skip connections)
        self.up1 = UNetUpBlock(num_filters * 8, num_filters * 8, dropout=0.5)        # 512 -> 512 + 512 skip = 1024 -> [B, 1024, 2, 2]
        self.up2 = UNetUpBlock(num_filters * 16, num_filters * 8, dropout=0.5)       # 1024 -> 512 + 512 skip = 1024 -> [B, 1024, 4, 4]
        self.up3 = UNetUpBlock(num_filters * 16, num_filters * 8, dropout=0.5)       # 1024 -> 512 + 512 skip = 1024 -> [B, 1024, 8, 8]
        self.up4 = UNetUpBlock(num_filters * 16, num_filters * 8)                    # 1024 -> 512 + 512 skip = 1024 -> [B, 1024, 16, 16]
        self.up5 = UNetUpBlock(num_filters * 16, num_filters * 4)                    # 1024 -> 256 + 256 skip = 512  -> [B, 512, 32, 32]
        self.up6 = UNetUpBlock(num_filters * 8, num_filters * 2)                     # 512  -> 128 + 128 skip = 256  -> [B, 256, 64, 64]
        self.up7 = UNetUpBlock(num_filters * 4, num_filters)                         # 256  -> 64  + 64 skip  = 128  -> [B, 128, 128, 128]

        # Final output layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, out_channels, kernel_size=4, stride=2, padding=1), # -> [B, 3, 256, 256]
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights according to the original Pix2Pix paper (Normal(0, 0.02))."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward pass with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        output = self.final_up(u7)
        return output
