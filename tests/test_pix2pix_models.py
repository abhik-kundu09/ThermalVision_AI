"""
Unit Tests for Pix2Pix U-Net Generator, PatchGAN Discriminator, and Loss Functions.
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.models.losses import Pix2PixLoss


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model tests")
def test_unet_generator_forward_and_shapes():
    """Verify U-Net Generator input/output tensor shapes and dynamic range."""
    net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=32)
    batch_size = 2
    dummy_ir = torch.randn(batch_size, 1, 256, 256)

    fake_rgb = net_g(dummy_ir)

    assert fake_rgb.shape == (batch_size, 3, 256, 256)
    # Output must be bounded in [-1.0, 1.0] via Tanh
    assert fake_rgb.min().item() >= -1.0
    assert fake_rgb.max().item() <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model tests")
def test_unet_generator_gradient_flow():
    """Verify backward pass and gradient backpropagation through skip connections."""
    net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=16)
    dummy_ir = torch.randn(1, 1, 256, 256, requires_grad=True)
    target_rgb = torch.randn(1, 3, 256, 256)

    fake_rgb = net_g(dummy_ir)
    loss = torch.nn.functional.l1_loss(fake_rgb, target_rgb)
    loss.backward()

    assert dummy_ir.grad is not None
    assert net_g.down1.block[0].weight.grad is not None
    assert net_g.final_up[0].weight.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model tests")
def test_patchgan_discriminator_shapes():
    """Verify PatchGAN Discriminator produces 30x30 receptive field logits."""
    net_d = PatchGANDiscriminator(in_channels=4, num_filters=32)
    batch_size = 2
    dummy_ir = torch.randn(batch_size, 1, 256, 256)
    dummy_rgb = torch.randn(batch_size, 3, 256, 256)

    pred = net_d(dummy_ir, dummy_rgb)

    # 256x256 input through 4-layer stride convs -> [B, 1, 30, 30]
    assert pred.shape == (batch_size, 1, 30, 30)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model tests")
def test_pix2pix_loss_computation():
    """Verify Generator and Discriminator loss calculations."""
    loss_fn = Pix2PixLoss(lambda_l1=100.0)

    disc_pred_fake = torch.randn(2, 1, 30, 30)
    disc_pred_real = torch.randn(2, 1, 30, 30)
    real_rgb = torch.randn(2, 3, 256, 256)
    fake_rgb = torch.randn(2, 3, 256, 256)

    # 1. Generator Loss
    loss_g, g_metrics = loss_fn.compute_generator_loss(disc_pred_fake, real_rgb, fake_rgb)
    assert loss_g.ndim == 0  # Scalar loss
    assert loss_g.item() > 0
    assert "loss_g_total" in g_metrics
    assert "loss_g_l1" in g_metrics

    # 2. Discriminator Loss
    loss_d, d_metrics = loss_fn.compute_discriminator_loss(disc_pred_real, disc_pred_fake)
    assert loss_d.ndim == 0
    assert loss_d.item() > 0
    assert "loss_d_total" in d_metrics
