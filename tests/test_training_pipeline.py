"""
Unit Tests for Pix2Pix Training Engine and Checkpoint Saving.
"""

import os
import tempfile
import pytest

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.training.train import Pix2PixTrainer, run_smoke_test


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for training tests")
def test_trainer_single_epoch_and_checkpoint():
    """Verify single training epoch and checkpoint serialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=8)
        net_d = PatchGANDiscriminator(in_channels=4, num_filters=8)

        # Synthetic in-memory dataset of 4 patches
        dummy_ir = torch.randn(4, 1, 256, 256)
        dummy_rgb = torch.randn(4, 3, 256, 256)
        loader = DataLoader(TensorDataset(dummy_ir, dummy_rgb), batch_size=2)

        trainer = Pix2PixTrainer(
            net_g=net_g,
            net_d=net_d,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=tmpdir,
            use_amp=False
        )

        metrics = trainer.train_epoch(epoch=1)
        assert metrics["loss_g"] > 0.0
        assert metrics["loss_d"] > 0.0

        val_l1 = trainer.evaluate_val()
        assert val_l1 > 0.0

        # Save checkpoint
        trainer.save_checkpoints(epoch=1, is_best=True)

        latest_path = os.path.join(tmpdir, "generator_latest.pth")
        best_path = os.path.join(tmpdir, "generator_best.pth")
        state_path = os.path.join(tmpdir, "training_state_latest.pth")

        assert os.path.exists(latest_path)
        assert os.path.exists(best_path)
        assert os.path.exists(state_path)

        # Verify saved state dict can be re-loaded into generator
        new_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=8)
        state_dict = torch.load(best_path, map_location="cpu")
        new_g.load_state_dict(state_dict)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for training tests")
def test_run_smoke_test_execution():
    """Verify smoke test completes without raising exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        success = run_smoke_test(data_dir=tmpdir)
        assert success is True
