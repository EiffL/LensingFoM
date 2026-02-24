"""Parquet loading, normalization, and 3-way split for SBI pipeline."""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

# 10 unique (i,j) pairs for 4 tomographic bins, matching lensing.spectra.SPEC_PAIRS
CL_COLUMNS = [f"cl_{i}_{j}" for i in range(4) for j in range(i, 4)]


def load_spectra_parquet(path):
    """Load spectra parquet into arrays using HuggingFace datasets.

    Parameters
    ----------
    path : str or Path
        Path to spectra_lmax{lmax}.parquet.

    Returns
    -------
    spectra : np.ndarray, shape (N, 200)
        Stacked power spectra (10 pairs x 20 bins).
    theta : np.ndarray, shape (N, 2)
        Cosmological parameters [Omega_m, S8].
    sim_ids : np.ndarray, shape (N,)
        Simulation IDs for splitting.
    """
    ds = load_dataset("parquet", data_files=str(path), split="train")
    ds = ds.with_format("numpy")
    spectra = np.column_stack([ds[col] for col in CL_COLUMNS]).astype(np.float64)
    theta = np.column_stack([ds["Omega_m"], ds["S8"]]).astype(np.float64)
    return spectra, theta


class SpectraDataModule(L.LightningDataModule):
    """Lightning DataModule for spectra parquet files.

    3-way split by sim_id (60/20/20%), shuffled:
    - Train: 60% sims — compressor training
    - Val/NPE: 20% sims — compressor validation + NPE training
    - Test: 20% sims — final FoM evaluation

    Normalization stats computed from train split only.
    """

    def __init__(self, parquet_path, batch_size=256, seed=42):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.seed = seed
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        spectra, theta = load_spectra_parquet(self.parquet_path)

        # Split by sim_id: shuffle sims, then assign 60/20/20
        rng = np.random.default_rng(self.seed)
        unique_ids = np.arange(len(theta))
        rng.shuffle(unique_ids)

        spectra = spectra[unique_ids]
        theta = theta[unique_ids]

        n = len(unique_ids)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n_train] = True
        val_mask = np.zeros(n, dtype=bool)
        val_mask[n_train:n_train + n_val] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[n_train + n_val:] = True

        # Normalization stats from train split only
        spectra_mean = spectra[train_mask].mean(axis=0)
        spectra_std = spectra[train_mask].std(axis=0)
        spectra_std = np.where(spectra_std == 0, 1.0, spectra_std)

        theta_mean = theta[train_mask].mean(axis=0)
        theta_std = theta[train_mask].std(axis=0)
        theta_std = np.where(theta_std == 0, 1.0, theta_std)

        norm = (spectra_mean, spectra_std, theta_mean, theta_std)

        self.train_ds = SpectraDataset(spectra[train_mask], theta[train_mask], *norm)
        self.val_ds = SpectraDataset(spectra[val_mask], theta[val_mask], *norm)
        self.test_ds = SpectraDataset(spectra[test_mask], theta[test_mask], *norm)

    def train_dataloader(self):
        x, y = self.train_ds.tensors()
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        x, y = self.val_ds.tensors()
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, num_workers=4, persistent_workers=True)


class SpectraDataset:
    """Normalized spectra + theta, with raw theta kept for calibration."""

    def __init__(self, spectra, theta, spectra_mean, spectra_std, theta_mean, theta_std):
        self.spectra = (spectra - spectra_mean) / spectra_std
        self.theta = (theta - theta_mean) / theta_std
        self.spectra_mean = spectra_mean
        self.spectra_std = spectra_std
        self.theta_mean = theta_mean
        self.theta_std = theta_std

    def __len__(self):
        return len(self.spectra)

    def tensors(self):
        """Return (spectra, theta) as float32 tensors."""
        return (
            torch.tensor(self.spectra, dtype=torch.float32),
            torch.tensor(self.theta, dtype=torch.float32),
        )
