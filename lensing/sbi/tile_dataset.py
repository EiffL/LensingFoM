"""Parquet tile loading, normalization, and 3-way split for field-level SBI."""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}


def load_tiles_parquet(parquet_dir):
    """Load tile images + cosmo params from parquet shards on the volume.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing shard_*.parquet files
        (e.g. /results/hf_dataset/lmax_200_des_y3/).

    Returns
    -------
    tiles : np.ndarray, shape (N, 4, nside, nside)
    theta : np.ndarray, shape (N, 2)  — [Omega_m, S8]
    sim_ids : np.ndarray, shape (N,)  — simulation ID for each tile
    """
    from pathlib import Path
    shards = sorted(Path(parquet_dir).glob("shard_*.parquet"))
    ds = load_dataset("parquet", data_files=[str(s) for s in shards], split="train")
    ds = ds.with_format("numpy")
    tiles = np.array(ds["kappa"], dtype=np.float32)
    theta = np.column_stack([ds["Omega_m"], ds["S8"]]).astype(np.float32)
    sim_ids = np.array(ds["sim_id"], dtype=np.int32)
    return tiles, theta, sim_ids


class TileDataset:
    """Normalized tiles + theta stored as tensors."""

    def __init__(self, tiles, theta, tile_mean, tile_std, theta_mean, theta_std):
        # Normalize in numpy, then convert to tensors once
        tiles = (tiles - tile_mean[None, :, None, None]) / tile_std[None, :, None, None]
        theta = (theta - theta_mean) / theta_std
        self.x = torch.tensor(tiles, dtype=torch.float32)
        self.y = torch.tensor(theta, dtype=torch.float32)
        self.tile_mean = tile_mean
        self.tile_std = tile_std
        self.theta_mean = theta_mean
        self.theta_std = theta_std

    def __len__(self):
        return len(self.x)

    def tensors(self):
        """Return (tiles, theta) as float32 tensors."""
        return self.x, self.y


class TileDataModule(L.LightningDataModule):
    """Lightning DataModule for tile images from HuggingFace dataset.

    3-way split (80/10/10%), shuffled. Same logic as SpectraDataModule.
    Per-channel normalization stats computed from train split only.
    """

    def __init__(self, parquet_dir, batch_size=64, seed=42):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.batch_size = batch_size
        self.seed = seed
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._train_dl = None
        self._val_dl = None

    def setup(self, stage=None):
        tiles, theta, _sim_ids = load_tiles_parquet(self.parquet_dir)

        # Shuffle and split 80/10/10 — mirrors SpectraDataModule exactly
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(theta))
        rng.shuffle(idx)

        tiles = tiles[idx]
        theta = theta[idx]

        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n_train] = True
        val_mask = np.zeros(n, dtype=bool)
        val_mask[n_train:n_train + n_val] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[n_train + n_val:] = True

        # Per-channel normalization stats from train split
        tile_mean = tiles[train_mask].mean(axis=(0, 2, 3))  # (4,)
        tile_std = tiles[train_mask].std(axis=(0, 2, 3))     # (4,)
        tile_std = np.where(tile_std == 0, 1.0, tile_std)

        theta_mean = theta[train_mask].mean(axis=0)
        theta_std = theta[train_mask].std(axis=0)
        theta_std = np.where(theta_std == 0, 1.0, theta_std)

        norm = (tile_mean, tile_std, theta_mean, theta_std)

        self.train_ds = TileDataset(tiles[train_mask], theta[train_mask], *norm)
        self.val_ds = TileDataset(tiles[val_mask], theta[val_mask], *norm)
        self.test_ds = TileDataset(tiles[test_mask], theta[test_mask], *norm)

        # Pre-build DataLoaders once (data is in-memory tensors, no workers needed)
        self._train_dl = DataLoader(
            TensorDataset(self.train_ds.x, self.train_ds.y),
            batch_size=self.batch_size, shuffle=True, pin_memory=True,
        )
        self._val_dl = DataLoader(
            TensorDataset(self.val_ds.x, self.val_ds.y),
            batch_size=self.batch_size, pin_memory=True,
        )

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl
