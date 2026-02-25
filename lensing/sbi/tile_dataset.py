"""Parquet tile loading, normalization, and 3-way split for field-level SBI."""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset

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


class CompressedTileDataset(Dataset):
    """Wraps tiles + frozen compressor for on-the-fly augmented compression.

    Each __getitem__ call optionally augments the tile, runs it through
    the frozen compressor, and returns (compressed_summary, theta).
    """

    def __init__(self, tile_ds, compressor, augment=True):
        self.x = tile_ds.x          # (N, 4, H, W) normalized tiles
        self.y = tile_ds.y          # (N, 2) normalized theta
        self.compressor = compressor
        self.augment = augment

    def __len__(self):
        return len(self.x)

    @staticmethod
    def _augment_tile(x):
        """Random augmentation on a single tile: rotations, flips, circular rolls.

        Parameters
        ----------
        x : Tensor, shape (C, H, W)
        """
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])
        H, W = x.shape[1], x.shape[2]
        x = torch.roll(x, shifts=torch.randint(0, H, (1,)).item(), dims=1)
        x = torch.roll(x, shifts=torch.randint(0, W, (1,)).item(), dims=2)
        return x

    def __getitem__(self, idx):
        tile = self.x[idx]   # (C, H, W)
        theta = self.y[idx]  # (2,)

        if self.augment:
            tile = self._augment_tile(tile)

        with torch.no_grad():
            params = list(self.compressor.parameters())
            device = params[0].device if params else torch.device("cpu")
            summary = self.compressor(tile.unsqueeze(0).to(device)).squeeze(0).cpu()

        return summary, theta


class TileDataModule(L.LightningDataModule):
    """Lightning DataModule for tile images from HuggingFace dataset.

    Splits tiles into compressor-train (70%), NPE-train/compressor-val (25%),
    and test (5%). Optionally holds out a fiducial simulation for FoM evaluation.
    Per-channel normalization stats computed from compressor-train split only.
    """

    def __init__(self, parquet_dir, batch_size=64, seed=42, fiducial_sim_id=109):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.batch_size = batch_size
        self.seed = seed
        self.fiducial_sim_id = fiducial_sim_id
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.fiducial_ds = None
        self._train_dl = None
        self._val_dl = None

    def setup(self, stage=None):
        tiles, theta, sim_ids = load_tiles_parquet(self.parquet_dir)

        # Separate fiducial tiles
        if self.fiducial_sim_id is not None:
            fid_mask = sim_ids == self.fiducial_sim_id
            fid_tiles, fid_theta = tiles[fid_mask], theta[fid_mask]
            tiles, theta = tiles[~fid_mask], theta[~fid_mask]
        else:
            fid_tiles, fid_theta = None, None

        # Shuffle and split 70/25/5
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(theta))
        rng.shuffle(idx)
        tiles, theta = tiles[idx], theta[idx]

        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.25 * n)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n_train] = True
        val_mask = np.zeros(n, dtype=bool)
        val_mask[n_train:n_train + n_val] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[n_train + n_val:] = True

        # Per-channel normalization stats from compressor-train split
        tile_mean = tiles[train_mask].mean(axis=(0, 2, 3))
        tile_std = tiles[train_mask].std(axis=(0, 2, 3))
        tile_std = np.where(tile_std == 0, 1.0, tile_std)

        theta_mean = theta[train_mask].mean(axis=0)
        theta_std = theta[train_mask].std(axis=0)
        theta_std = np.where(theta_std == 0, 1.0, theta_std)

        norm = (tile_mean, tile_std, theta_mean, theta_std)

        self.train_ds = TileDataset(tiles[train_mask], theta[train_mask], *norm)
        self.val_ds = TileDataset(tiles[val_mask], theta[val_mask], *norm)
        self.test_ds = TileDataset(tiles[test_mask], theta[test_mask], *norm)

        if fid_tiles is not None and len(fid_tiles) > 0:
            self.fiducial_ds = TileDataset(fid_tiles, fid_theta, *norm)
        else:
            self.fiducial_ds = None

        # Pre-build DataLoaders
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
