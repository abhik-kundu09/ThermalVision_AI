"""
Loss Functions for Pix2Pix Conditional GAN Training.

Implements:
- Least Squares / Binary Cross Entropy Adversarial Loss (L_GAN)
- L1 Reconstruction / Color Fidelity Loss (L_L1)
- Composite Generator Loss: L_G = L_GAN + lambda * L_L1 (default lambda = 100.0)
- Discriminator Loss: L_D = 0.5 * (L_D_real + L_D_fake)
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn


class Pix2PixLoss(nn.Module):
    """
    Composite Loss Manager for Pix2Pix training.
    """

    def __init__(self, lambda_l1: float = 100.0, use_lsgan: bool = False):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.use_lsgan = use_lsgan

        if use_lsgan:
            self.criterion_gan = nn.MSELoss()
        else:
            self.criterion_gan = nn.BCEWithLogitsLoss()

        self.criterion_l1 = nn.L1Loss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Creates target label tensor matching the prediction spatial shape."""
        target_val = 1.0 if target_is_real else 0.0
        return torch.full_like(prediction, target_val, device=prediction.device)

    def compute_generator_loss(
        self,
        disc_pred_fake: torch.Tensor,
        real_rgb: torch.Tensor,
        fake_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Total Generator Loss:
        L_G = L_cGAN(D(x, G(x)), 1) + lambda_l1 * L1(y, G(x))
        """
        # Adversarial loss: Generator wants Discriminator to believe fake is real (target=1.0)
        target_real = self.get_target_tensor(disc_pred_fake, target_is_real=True)
        loss_gan = self.criterion_gan(disc_pred_fake, target_real)

        # L1 Pixel-level reconstruction loss
        loss_l1 = self.criterion_l1(fake_rgb, real_rgb)

        # Total loss
        loss_g_total = loss_gan + self.lambda_l1 * loss_l1

        metrics = {
            "loss_g_total": float(loss_g_total.item()),
            "loss_g_gan": float(loss_gan.item()),
            "loss_g_l1": float(loss_l1.item())
        }
        return loss_g_total, metrics

    def compute_discriminator_loss(
        self,
        disc_pred_real: torch.Tensor,
        disc_pred_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Discriminator Loss:
        L_D = 0.5 * [ L_cGAN(D(x, y), 1) + L_cGAN(D(x, G(x)), 0) ]
        """
        target_real = self.get_target_tensor(disc_pred_real, target_is_real=True)
        target_fake = self.get_target_tensor(disc_pred_fake, target_is_real=False)

        loss_d_real = self.criterion_gan(disc_pred_real, target_real)
        loss_d_fake = self.criterion_gan(disc_pred_fake, target_fake)

        loss_d_total = 0.5 * (loss_d_real + loss_d_fake)

        metrics = {
            "loss_d_total": float(loss_d_total.item()),
            "loss_d_real": float(loss_d_real.item()),
            "loss_d_fake": float(loss_d_fake.item())
        }
        return loss_d_total, metrics
