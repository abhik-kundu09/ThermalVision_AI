"""
PyTorch Dataset and DataLoader for Landsat Paired Thermal IR -> RGB Translation.

Loads normalized 256x256 .npy patch pairs:
- Input (IR): (1, 256, 256) float32 in range [-1.0, 1.0]
- Target (RGB): (3, 256, 256) float32 in range [-1.0, 1.0]
"""

import os
import json
import logging
from typing import Tuple, Optional, List, Dict, Any
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal stub if torch is not installed locally
    class Dataset:
        pass

logger = logging.getLogger("ps10.data.dataset")


class LandsatPatchDataset(Dataset):
    """
    PyTorch Dataset for paired Landsat Thermal IR -> Visible RGB patches.
    """

    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        manifest_path: Optional[str] = None,
        transform: Optional[Any] = None
    ):
        """
        Args:
            data_dir: Base directory containing dataset (e.g. 'data').
            split: Dataset split ('train', 'val', or 'test').
            manifest_path: Optional explicit path to dataset_manifest.json.
            transform: Optional spatial augmentation callable (e.g. random flips).
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.patch_pairs: List[Tuple[str, str]] = []

        m_path = manifest_path or os.path.join(data_dir, "dataset_manifest.json")

        if os.path.exists(m_path):
            with open(m_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            split_entries = manifest.get("splits", {}).get(split, [])
            for entry in split_entries:
                ir_p = os.path.join(data_dir, entry["ir_file"])
                rgb_p = os.path.join(data_dir, entry["rgb_file"])
                if os.path.exists(ir_p) and os.path.exists(rgb_p):
                    self.patch_pairs.append((ir_p, rgb_p))
        else:
            # Fallback: scan split directory directly
            ir_dir = os.path.join(data_dir, split, "ir")
            rgb_dir = os.path.join(data_dir, split, "rgb")
            if os.path.exists(ir_dir) and os.path.exists(rgb_dir):
                ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(".npy")])
                for ir_f in ir_files:
                    rgb_f = ir_f.replace("_ir.npy", "_rgb.npy")
                    ir_full = os.path.join(ir_dir, ir_f)
                    rgb_full = os.path.join(rgb_dir, rgb_f)
                    if os.path.exists(rgb_full):
                        self.patch_pairs.append((ir_full, rgb_full))

        logger.info(f"Loaded {len(self.patch_pairs)} patch pairs for split '{split}'.")

    def __len__(self) -> int:
        return len(self.patch_pairs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Returns:
            ir_tensor: Tensor of shape (1, 256, 256) in [-1.0, 1.0].
            rgb_tensor: Tensor of shape (3, 256, 256) in [-1.0, 1.0].
        """
        ir_path, rgb_path = self.patch_pairs[idx]

        # Load numpy arrays
        ir_arr = np.load(ir_path)    # Shape: (256, 256)
        rgb_arr = np.load(rgb_path)  # Shape: (256, 256, 3)

        # Expand IR to (1, H, W) and transpose RGB to (3, H, W)
        if ir_arr.ndim == 2:
            ir_arr = np.expand_dims(ir_arr, axis=0)  # (1, 256, 256)
        elif ir_arr.ndim == 3 and ir_arr.shape[-1] == 1:
            ir_arr = np.transpose(ir_arr, (2, 0, 1))

        if rgb_arr.ndim == 3 and rgb_arr.shape[-1] == 3:
            rgb_arr = np.transpose(rgb_arr, (2, 0, 1))  # (3, 256, 256)

        if HAS_TORCH:
            ir_tensor = torch.from_numpy(ir_arr.astype(np.float32))
            rgb_tensor = torch.from_numpy(rgb_arr.astype(np.float32))

            # Apply random horizontal/vertical flips for training
            if self.transform is not None:
                ir_tensor, rgb_tensor = self.transform(ir_tensor, rgb_tensor)

            return ir_tensor, rgb_tensor
        else:
            return ir_arr.astype(np.float32), rgb_arr.astype(np.float32)


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 16,
    num_workers: int = 0
) -> Dict[str, Any]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    train_ds = LandsatPatchDataset(data_dir=data_dir, split="train")
    val_ds = LandsatPatchDataset(data_dir=data_dir, split="val")
    test_ds = LandsatPatchDataset(data_dir=data_dir, split="test")

    if not HAS_TORCH:
        return {"train": train_ds, "val": val_ds, "test": test_ds}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
