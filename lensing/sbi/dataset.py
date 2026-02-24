"""Parquet loading, normalization, and 3-way split for SBI pipeline."""

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

# 10 unique (i,j) pairs for 4 tomographic bins, matching lensing.spectra.SPEC_PAIRS
CL_COLUMNS = [f"cl_{i}_{j}" for i in range(4) for j in range(i, 4)]


def load_spectra_parquet(path):
    """Load spectra parquet into arrays.

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
    table = pq.read_table(path)
    df = table.to_pandas()

    # Stack 10 cl columns (each length-20 list) into (N, 200)
    spectra = np.column_stack([np.stack(df[col].values) for col in CL_COLUMNS])
    theta = df[["Omega_m", "S8"]].values.astype(np.float64)
    sim_ids = df["sim_id"].values

    return spectra, theta, sim_ids


def split_by_sim_id(sim_ids, seed=42):
    """Split rows into train/npe/test by sim_id (60/20/20%).

    Uses a fixed seed so the same split is used across all lmax values.

    Parameters
    ----------
    sim_ids : np.ndarray, shape (N,)
        Simulation ID per row.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_mask, npe_mask, test_mask : np.ndarray of bool
        Boolean masks over rows.
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(sim_ids)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(0.6 * n)
    n_npe = int(0.2 * n)

    train_ids = set(unique_ids[:n_train])
    npe_ids = set(unique_ids[n_train : n_train + n_npe])
    test_ids = set(unique_ids[n_train + n_npe :])

    train_mask = np.array([s in train_ids for s in sim_ids])
    npe_mask = np.array([s in npe_ids for s in sim_ids])
    test_mask = np.array([s in test_ids for s in sim_ids])

    return train_mask, npe_mask, test_mask


class SpectraDataset:
    """Normalized spectra + theta dataset.

    Parameters
    ----------
    spectra : np.ndarray, shape (N, 200)
    theta : np.ndarray, shape (N, 2)
    spectra_mean, spectra_std : np.ndarray, shape (200,)
    theta_mean, theta_std : np.ndarray, shape (2,)
    """

    def __init__(self, spectra, theta, spectra_mean, spectra_std, theta_mean, theta_std):
        self.spectra = (spectra - spectra_mean) / spectra_std
        self.theta = (theta - theta_mean) / theta_std
        self.theta_raw = theta.copy()
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


class SpectraDataModule(L.LightningDataModule):
    """Lightning DataModule for spectra parquet files.

    Clean 3-way split with no data leakage:
    - Compressor train/val: 80/20 sub-split of the 60% train sims
    - NPE: 20% sims (untouched until NPE training)
    - Test: 20% sims (untouched until FoM evaluation)

    Parameters
    ----------
    parquet_path : str or Path
        Path to spectra_lmax{lmax}.parquet.
    batch_size : int
        Batch size for dataloaders.
    compressor_val_frac : float
        Fraction of train sims held out for compressor early stopping.
    """

    def __init__(self, parquet_path, batch_size=256, compressor_val_frac=0.2):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.compressor_val_frac = compressor_val_frac
        self.comp_train_ds = None
        self.comp_val_ds = None
        self.npe_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        spectra, theta, sim_ids = load_spectra_parquet(self.parquet_path)
        train_mask, npe_mask, test_mask = split_by_sim_id(sim_ids)

        # Sub-split train sims into compressor-train / compressor-val
        train_sim_ids = np.unique(sim_ids[train_mask])
        rng = np.random.default_rng(123)  # different seed from the main split
        rng.shuffle(train_sim_ids)
        n_comp_val = max(1, int(self.compressor_val_frac * len(train_sim_ids)))
        comp_val_ids = set(train_sim_ids[:n_comp_val])
        comp_train_mask = train_mask & np.array([s not in comp_val_ids for s in sim_ids])
        comp_val_mask = train_mask & np.array([s in comp_val_ids for s in sim_ids])

        # Compute normalization stats from compressor-train split only
        # Use relative threshold: clamp std to at least 1e-6 * |mean| to avoid
        # zeroing out features with small but physically meaningful variance
        spectra_mean = spectra[comp_train_mask].mean(axis=0)
        spectra_std = spectra[comp_train_mask].std(axis=0)
        min_std = np.maximum(np.abs(spectra_mean) * 1e-6, 1e-30)
        spectra_std = np.maximum(spectra_std, min_std)

        theta_mean = theta[comp_train_mask].mean(axis=0)
        theta_std = theta[comp_train_mask].std(axis=0)
        theta_std = np.where(theta_std == 0, 1.0, theta_std)

        norm = (spectra_mean, spectra_std, theta_mean, theta_std)

        self.comp_train_ds = SpectraDataset(
            spectra[comp_train_mask], theta[comp_train_mask], *norm,
        )
        self.comp_val_ds = SpectraDataset(
            spectra[comp_val_mask], theta[comp_val_mask], *norm,
        )
        self.npe_ds = SpectraDataset(
            spectra[npe_mask], theta[npe_mask], *norm,
        )
        self.test_ds = SpectraDataset(
            spectra[test_mask], theta[test_mask], *norm,
        )

    def train_dataloader(self):
        x, y = self.comp_train_ds.tensors()
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        x, y = self.comp_val_ds.tensors()
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size)
