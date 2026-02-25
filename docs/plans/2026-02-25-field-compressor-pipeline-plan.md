# Field Compressor Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the field compressor pipeline into two decoupled stages (compressor training, NPE training) with fiducial cosmology holdout (sim 109) for robust FoM evaluation.

**Architecture:** `TileDataModule` gains fiducial sim filtering and a 70/25/5 split. `load_tiles_parquet` returns `sim_id` arrays. Compressor training saves checkpoint + normalization stats. A new `train_field_npe.py` script loads the frozen compressor, wraps tiles in a `CompressedTileDataset` that augments on the fly, trains `GaussianNPE`, and evaluates FoM on the 12 fiducial tiles.

**Tech Stack:** PyTorch, Lightning, Modal, wandb, HuggingFace datasets, existing `lensing.sbi` modules.

---

### Task 1: Update `load_tiles_parquet` to return `sim_id`

**Files:**
- Modify: `lensing/sbi/tile_dataset.py:13-33`

**Step 1: Write the failing test**

Create `tests/test_tile_dataset.py`:

```python
"""Tests for tile dataset loading and splitting."""
import numpy as np
import pytest


def test_load_tiles_parquet_returns_sim_ids(tmp_path):
    """load_tiles_parquet should return (tiles, theta, sim_ids)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Create a minimal parquet shard with 2 sims, 2 tiles each
    nside = 4
    records = []
    for sim_id in [1, 2]:
        for tile_idx in range(2):
            records.append({
                "kappa": np.random.randn(4, nside, nside).tolist(),
                "sim_id": sim_id,
                "orientation_id": 0,
                "tile_id": tile_idx,
                "noise_level": "des_y3",
                "Omega_m": 0.3 + sim_id * 0.01,
                "sigma_8": 0.8,
                "S8": 0.8,
            })

    table = pa.table({k: [r[k] for r in records] for k in records[0]})
    pq.write_table(table, tmp_path / "shard_000.parquet")

    from lensing.sbi.tile_dataset import load_tiles_parquet
    tiles, theta, sim_ids = load_tiles_parquet(tmp_path)

    assert tiles.shape == (4, 4, nside, nside)
    assert theta.shape == (4, 2)
    assert sim_ids.shape == (4,)
    assert set(sim_ids) == {1, 2}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tile_dataset.py::test_load_tiles_parquet_returns_sim_ids -v`
Expected: FAIL — `load_tiles_parquet` returns 2 values, not 3.

**Step 3: Write minimal implementation**

In `lensing/sbi/tile_dataset.py`, change `load_tiles_parquet` to:

```python
def load_tiles_parquet(parquet_dir):
    """Load tile images + cosmo params + sim IDs from parquet shards.

    Returns
    -------
    tiles : np.ndarray, shape (N, 4, nside, nside)
    theta : np.ndarray, shape (N, 2)  — [Omega_m, S8]
    sim_ids : np.ndarray, shape (N,)  — simulation ID per tile
    """
    from pathlib import Path
    shards = sorted(Path(parquet_dir).glob("shard_*.parquet"))
    ds = load_dataset("parquet", data_files=[str(s) for s in shards], split="train")
    ds = ds.with_format("numpy")
    tiles = np.array(ds["kappa"], dtype=np.float32)
    theta = np.column_stack([ds["Omega_m"], ds["S8"]]).astype(np.float32)
    sim_ids = np.array(ds["sim_id"], dtype=np.int32)
    return tiles, theta, sim_ids
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tile_dataset.py::test_load_tiles_parquet_returns_sim_ids -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lensing/sbi/tile_dataset.py tests/test_tile_dataset.py
git commit -m "feat: load_tiles_parquet now returns sim_ids"
```

---

### Task 2: Update `TileDataModule` with fiducial holdout and 70/25/5 split

**Files:**
- Modify: `lensing/sbi/tile_dataset.py:58-127`
- Modify: `tests/test_tile_dataset.py` (add tests)

**Step 1: Write the failing tests**

Append to `tests/test_tile_dataset.py`:

```python
FIDUCIAL_SIM_ID = 109


def _make_parquet_shards(tmp_path, n_sims=10, tiles_per_sim=2, nside=4):
    """Helper: create parquet shards with known sim_ids."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for sim_id in range(1, n_sims + 1):
        for tile_idx in range(tiles_per_sim):
            records.append({
                "kappa": np.random.randn(4, nside, nside).tolist(),
                "sim_id": sim_id,
                "orientation_id": 0,
                "tile_id": tile_idx,
                "noise_level": "des_y3",
                "Omega_m": 0.3 + sim_id * 0.001,
                "sigma_8": 0.8,
                "S8": 0.8,
            })

    table = pa.table({k: [r[k] for r in records] for k in records[0]})
    pq.write_table(table, tmp_path / "shard_000.parquet")
    return tmp_path


def test_tile_data_module_fiducial_holdout(tmp_path):
    """Fiducial sim tiles should be in fiducial_ds, not in any split."""
    from lensing.sbi.tile_dataset import TileDataModule

    # Include sim 109 in our test data
    _make_parquet_shards(tmp_path, n_sims=200, tiles_per_sim=2, nside=4)

    dm = TileDataModule(tmp_path, batch_size=8, fiducial_sim_id=FIDUCIAL_SIM_ID)
    dm.setup()

    assert dm.fiducial_ds is not None
    assert len(dm.fiducial_ds) == 2  # 2 tiles for sim 109

    # Total should be n_sims * tiles_per_sim
    total = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds) + len(dm.fiducial_ds)
    assert total == 200 * 2


def test_tile_data_module_split_ratios(tmp_path):
    """70/25/5 split of non-fiducial tiles."""
    from lensing.sbi.tile_dataset import TileDataModule

    _make_parquet_shards(tmp_path, n_sims=200, tiles_per_sim=2, nside=4)

    dm = TileDataModule(tmp_path, batch_size=8, fiducial_sim_id=FIDUCIAL_SIM_ID)
    dm.setup()

    n_non_fid = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
    train_frac = len(dm.train_ds) / n_non_fid
    val_frac = len(dm.val_ds) / n_non_fid

    assert 0.68 < train_frac < 0.72  # ~70%
    assert 0.23 < val_frac < 0.27    # ~25%


def test_tile_data_module_no_fiducial(tmp_path):
    """With fiducial_sim_id=None, all tiles go into train/val/test."""
    from lensing.sbi.tile_dataset import TileDataModule

    _make_parquet_shards(tmp_path, n_sims=50, tiles_per_sim=2, nside=4)

    dm = TileDataModule(tmp_path, batch_size=8, fiducial_sim_id=None)
    dm.setup()

    assert dm.fiducial_ds is None
    total = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
    assert total == 50 * 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tile_dataset.py::test_tile_data_module_fiducial_holdout tests/test_tile_dataset.py::test_tile_data_module_split_ratios tests/test_tile_dataset.py::test_tile_data_module_no_fiducial -v`
Expected: FAIL — `TileDataModule.__init__` doesn't accept `fiducial_sim_id`, no `fiducial_ds` attribute.

**Step 3: Write implementation**

Rewrite `TileDataModule` in `lensing/sbi/tile_dataset.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tile_dataset.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lensing/sbi/tile_dataset.py tests/test_tile_dataset.py
git commit -m "feat: TileDataModule with fiducial holdout and 70/25/5 split"
```

---

### Task 3: Update `train_field_compressor.py` — compressor only, save norm stats

**Files:**
- Modify: `scripts/train_field_compressor.py`

Strip out all NPE/FoM code. Save normalization stats alongside checkpoint.

**Step 1: Rewrite the script**

The script should:
1. Load data via `TileDataModule(parquet_dir, batch_size, fiducial_sim_id=109)`
2. Train `FieldLevelCompressor` with same hyperparams
3. After training, compute MSE on train/val/test splits
4. Save `norm_stats.json` with `tile_mean`, `tile_std`, `theta_mean`, `theta_std` (as lists)
5. Save `result.json` with MSE metrics only (no FoM)
6. Do NOT train NPE or compute FoM

```python
"""Train field-level compressor on convergence map tiles with MSE loss.

Usage:
    modal run scripts/train_field_compressor.py
    modal run scripts/train_field_compressor.py --lmax 200 --noise-level des_y3
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow", "datasets",
        "torch", "torchvision", "lightning", "wandb",
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-train-field-compressor", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="H100",
    timeout=7200,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_compressor(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 500,
    max_lr: float = 0.008,
    backbone: str = "efficientnet_v2_s",
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 10,
):
    """Train field-level MSE compressor, save checkpoint and norm stats."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import torch
    import wandb
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from lensing.sbi.field_compressor import FieldLevelCompressor
    from lensing.sbi.tile_dataset import LMAX_TO_NSIDE, TileDataModule

    t0 = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # --- Load data ---
    nside = LMAX_TO_NSIDE[lmax]
    parquet_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"
    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"{parquet_dir} not found. Run pipeline.py --stage 3 first."
        )

    dm = TileDataModule(parquet_dir, batch_size=batch_size, fiducial_sim_id=109)
    dm.setup()

    n_train = len(dm.train_ds)
    n_val = len(dm.val_ds)
    n_test = len(dm.test_ds)
    n_fid = len(dm.fiducial_ds) if dm.fiducial_ds else 0

    print(f"Loaded {n_train + n_val + n_test + n_fid} tiles from {parquet_dir}")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test / {n_fid} fiducial")
    print(f"Tile shape: (4, {nside}, {nside})")

    # --- Build model ---
    tag = f"field_mse_lmax{lmax}_{noise_level}_{backbone}"
    run_dir = Path(RESULTS_DIR) / "field_compressor_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model = FieldLevelCompressor(
        nside=nside,
        n_bins=4,
        theta_dim=2,
        backbone=backbone,
        max_lr=max_lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        decay_rate=decay_rate,
        decay_every_epochs=decay_every_epochs,
    )

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, nside=nside,
            n_train=n_train, n_val=n_val, n_test=n_test,
            backbone=backbone, loss="mse",
            max_lr=max_lr, batch_size=batch_size, max_epochs=max_epochs,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            decay_rate=decay_rate, decay_every_epochs=decay_every_epochs,
        ),
        save_dir=str(run_dir),
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=str(run_dir), filename="best",
        monitor="val_loss", mode="min",
    )

    # --- Train ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        accelerator="auto",
        precision="bf16-mixed",
    )
    trainer.fit(model, dm)

    # --- Evaluate MSE on all splits ---
    best_path = ckpt_callback.best_model_path
    if best_path:
        model = FieldLevelCompressor.load_from_checkpoint(
            best_path, weights_only=False,
        )
    model = model.cpu().eval()

    splits = {}
    for split_name, ds in [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]:
        x, theta = ds.tensors()
        with torch.no_grad():
            pred = model(x)
            mse = torch.nn.functional.mse_loss(pred, theta)
        splits[split_name] = float(mse)
        print(f"{split_name:5s} MSE: {float(mse):.6f}")

    wandb_logger.experiment.summary.update({
        f"{s}/mse": v for s, v in splits.items()
    })

    # --- Save normalization stats ---
    norm_stats = {
        "tile_mean": dm.train_ds.tile_mean.tolist(),
        "tile_std": dm.train_ds.tile_std.tolist(),
        "theta_mean": dm.train_ds.theta_mean.tolist(),
        "theta_std": dm.train_ds.theta_std.tolist(),
    }
    with open(run_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # --- Save result ---
    result = dict(
        lmax=lmax, noise_level=noise_level, nside=nside,
        n_train=n_train, n_val=n_val, n_test=n_test, n_fiducial=n_fid,
        backbone=backbone,
        max_lr=max_lr, batch_size=batch_size, weight_decay=weight_decay,
        train_mse=splits["train"],
        val_mse=splits["val"],
        test_mse=splits["test"],
        epochs_trained=trainer.current_epoch,
        elapsed_s=time.time() - t0,
    )
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    wandb.finish()

    print(f"Total time: {time.time()-t0:.0f}s")
    return result


@app.local_entrypoint()
def main(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 500,
    max_lr: float = 0.008,
    backbone: str = "efficientnet_v2_s",
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 10,
):
    result = train_compressor.remote(
        lmax=lmax, noise_level=noise_level,
        max_epochs=max_epochs, max_lr=max_lr,
        backbone=backbone,
        batch_size=batch_size, weight_decay=weight_decay,
        warmup_steps=warmup_steps, decay_rate=decay_rate,
        decay_every_epochs=decay_every_epochs,
    )
    print(f"\nTrain MSE: {result['train_mse']:.6f}  Val MSE: {result['val_mse']:.6f}  Test MSE: {result['test_mse']:.6f}")
    print(f"Trained {result['epochs_trained']} epochs in {result['elapsed_s']:.0f}s")
```

**Step 2: Verify it parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_field_compressor.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add scripts/train_field_compressor.py
git commit -m "refactor: decouple compressor training from NPE, save norm_stats"
```

---

### Task 4: Create `CompressedTileDataset` with on-the-fly augmentation

**Files:**
- Modify: `lensing/sbi/tile_dataset.py` (add class at end)
- Modify: `tests/test_tile_dataset.py` (add tests)

**Step 1: Write the failing test**

Append to `tests/test_tile_dataset.py`:

```python
def test_compressed_tile_dataset_augmentation():
    """Same tile should produce different compressed outputs across calls."""
    import torch
    from lensing.sbi.tile_dataset import TileDataset, CompressedTileDataset

    # Minimal mock compressor: just sum channels (deterministic given input)
    class MockCompressor(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3))  # (B, 4) -> (B, 4)

    tiles = np.random.randn(10, 4, 8, 8).astype(np.float32)
    theta = np.random.randn(10, 2).astype(np.float32)
    norm = (np.zeros(4), np.ones(4), np.zeros(2), np.ones(2))

    tile_ds = TileDataset(tiles, theta, *norm)
    compressor = MockCompressor()
    cds = CompressedTileDataset(tile_ds, compressor, augment=True)

    assert len(cds) == 10

    # Get same index twice — should differ due to augmentation
    s1, t1 = cds[0]
    s2, t2 = cds[0]

    # Theta should be identical
    assert torch.allclose(t1, t2)
    # Summaries will usually differ (augmentation is random)
    # Run many times — at least one should differ
    any_different = False
    for _ in range(20):
        s_new, _ = cds[0]
        if not torch.allclose(s1, s_new):
            any_different = True
            break
    assert any_different, "Augmentation should produce varying outputs"


def test_compressed_tile_dataset_no_augmentation():
    """Without augmentation, same tile should give same compressed output."""
    import torch
    from lensing.sbi.tile_dataset import TileDataset, CompressedTileDataset

    class MockCompressor(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3))

    tiles = np.random.randn(10, 4, 8, 8).astype(np.float32)
    theta = np.random.randn(10, 2).astype(np.float32)
    norm = (np.zeros(4), np.ones(4), np.zeros(2), np.ones(2))

    tile_ds = TileDataset(tiles, theta, *norm)
    compressor = MockCompressor()
    cds = CompressedTileDataset(tile_ds, compressor, augment=False)

    s1, _ = cds[0]
    s2, _ = cds[0]
    assert torch.allclose(s1, s2)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tile_dataset.py::test_compressed_tile_dataset_augmentation tests/test_tile_dataset.py::test_compressed_tile_dataset_no_augmentation -v`
Expected: FAIL — `CompressedTileDataset` not defined.

**Step 3: Write implementation**

Add to `lensing/sbi/tile_dataset.py`, after the `TileDataset` class. Import `torch.utils.data.Dataset` at the top.

```python
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CompressedTileDataset(Dataset):
    """Wraps tiles + frozen compressor for on-the-fly augmented compression.

    Each __getitem__ call augments the tile (if enabled), runs it through
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
        """Random augmentation: rotations, flips, circular rolls."""
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
        tile = self.x[idx]  # (4, H, W)
        theta = self.y[idx]  # (2,)

        if self.augment:
            tile = self._augment_tile(tile)

        with torch.no_grad():
            summary = self.compressor(tile.unsqueeze(0)).squeeze(0)  # (summary_dim,)

        return summary, theta
```

Note: the `_augment_tile` method operates on a single tile `(C, H, W)`, not a batch, so dims are shifted by 1 compared to `FieldLevelCompressor._augment`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tile_dataset.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lensing/sbi/tile_dataset.py tests/test_tile_dataset.py
git commit -m "feat: CompressedTileDataset with on-the-fly augmentation"
```

---

### Task 5: Create `scripts/train_field_npe.py` — NPE training + fiducial FoM

**Files:**
- Create: `scripts/train_field_npe.py`

**Step 1: Write the script**

```python
"""Train Gaussian NPE on frozen compressor outputs, evaluate FoM on fiducial sim.

Usage:
    modal run scripts/train_field_npe.py
    modal run scripts/train_field_npe.py --lmax 200 --noise-level des_y3
    modal run scripts/train_field_npe.py --compressor-tag field_mse_lmax200_des_y3_efficientnet_v2_s
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow", "datasets",
        "torch", "torchvision", "lightning", "wandb",
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-train-field-npe", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="A10G",
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_npe_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    backbone: str = "efficientnet_v2_s",
    npe_max_epochs: int = 500,
    npe_patience: int = 30,
    compressor_tag: str = "",
):
    """Load frozen compressor, train NPE, evaluate FoM on fiducial."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import numpy as np
    import torch
    import wandb
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader

    from lensing.sbi.field_compressor import FieldLevelCompressor
    from lensing.sbi.npe import GaussianNPE, compute_fom
    from lensing.sbi.tile_dataset import (
        LMAX_TO_NSIDE,
        CompressedTileDataset,
        TileDataModule,
    )

    t0 = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # --- Resolve compressor checkpoint ---
    if not compressor_tag:
        compressor_tag = f"field_mse_lmax{lmax}_{noise_level}_{backbone}"
    compressor_dir = Path(RESULTS_DIR) / "field_compressor_runs" / compressor_tag
    ckpt_path = compressor_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run train_field_compressor.py first."
        )

    # --- Load frozen compressor ---
    compressor = FieldLevelCompressor.load_from_checkpoint(
        str(ckpt_path), weights_only=False,
    )
    compressor = compressor.eval()
    compressor.requires_grad_(False)
    if torch.cuda.is_available():
        compressor = compressor.cuda()
    print(f"Loaded compressor from {ckpt_path}")

    # --- Load data (same split as compressor training) ---
    nside = LMAX_TO_NSIDE[lmax]
    parquet_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"

    dm = TileDataModule(parquet_dir, batch_size=128, fiducial_sim_id=109)
    dm.setup()

    print(f"Train: {len(dm.train_ds)}, Val: {len(dm.val_ds)}, "
          f"Test: {len(dm.test_ds)}, Fiducial: {len(dm.fiducial_ds)}")

    # --- Build compressed datasets ---
    npe_train_ds = CompressedTileDataset(dm.val_ds, compressor, augment=True)
    npe_val_ds = CompressedTileDataset(dm.test_ds, compressor, augment=False)

    train_dl = DataLoader(npe_train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_dl = DataLoader(npe_val_ds, batch_size=256, num_workers=0)

    # --- Train NPE ---
    npe_tag = f"npe_{compressor_tag}"
    run_dir = Path(RESULTS_DIR) / "field_npe_runs" / npe_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    npe = GaussianNPE(input_dim=2, theta_dim=2)

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=npe_tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, backbone=backbone,
            compressor_tag=compressor_tag,
            npe_max_epochs=npe_max_epochs, npe_patience=npe_patience,
        ),
        save_dir=str(run_dir),
    )

    trainer = L.Trainer(
        max_epochs=npe_max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=npe_patience, mode="min")],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
    )
    trainer.fit(npe, train_dl, val_dl)
    npe = npe.cpu().eval()

    # --- FoM on fiducial tiles ---
    fiducial_ds = CompressedTileDataset(dm.fiducial_ds, compressor.cpu(), augment=False)
    fid_summaries = torch.stack([fiducial_ds[i][0] for i in range(len(fiducial_ds))])
    fid_summaries_np = fid_summaries.numpy()

    fid_foms = npe.predict_fom(fid_summaries_np)

    # Un-normalize to physical units
    theta_std_prod = float(np.prod(dm.train_ds.theta_std))
    fid_foms_phys = fid_foms / theta_std_prod

    fom_median = float(np.median(fid_foms_phys))
    fom_std = float(np.std(fid_foms_phys))
    print(f"\nFiducial FoM: {fom_median:.1f} +/- {fom_std:.1f}")
    print(f"  All 12: {[f'{f:.1f}' for f in fid_foms_phys]}")

    # --- FoM on test set (sanity check) ---
    test_ds_compressed = CompressedTileDataset(dm.test_ds, compressor.cpu(), augment=False)
    test_summaries = torch.stack([test_ds_compressed[i][0] for i in range(len(test_ds_compressed))])
    test_fom_median, test_fom_lo, test_fom_hi, _ = compute_fom(npe, test_summaries.numpy())
    test_fom_median /= theta_std_prod
    test_fom_lo /= theta_std_prod
    test_fom_hi /= theta_std_prod
    print(f"Test FoM: {test_fom_median:.1f} [{test_fom_lo:.1f}, {test_fom_hi:.1f}]")

    # --- Save results ---
    wandb_logger.experiment.summary.update({
        "fom_fiducial_median": fom_median,
        "fom_fiducial_std": fom_std,
        "fom_test_median": test_fom_median,
    })

    result = dict(
        lmax=lmax, noise_level=noise_level, backbone=backbone,
        compressor_tag=compressor_tag,
        fom_fiducial_median=fom_median,
        fom_fiducial_std=fom_std,
        fom_fiducial_all=[float(f) for f in fid_foms_phys],
        fom_test_median=test_fom_median,
        fom_test_lo=test_fom_lo,
        fom_test_hi=test_fom_hi,
        npe_epochs=trainer.current_epoch,
        elapsed_s=time.time() - t0,
    )
    with open(run_dir / "fom_result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    wandb.finish()

    print(f"Total time: {time.time()-t0:.0f}s")
    return result


@app.local_entrypoint()
def main(
    lmax: int = 200,
    noise_level: str = "des_y3",
    backbone: str = "efficientnet_v2_s",
    npe_max_epochs: int = 500,
    npe_patience: int = 30,
    compressor_tag: str = "",
):
    result = train_npe_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level, backbone=backbone,
        npe_max_epochs=npe_max_epochs, npe_patience=npe_patience,
        compressor_tag=compressor_tag,
    )
    print(f"\nFiducial FoM: {result['fom_fiducial_median']:.1f} +/- {result['fom_fiducial_std']:.1f}")
    print(f"Test FoM: {result['fom_test_median']:.1f} [{result['fom_test_lo']:.1f}, {result['fom_test_hi']:.1f}]")
    print(f"NPE trained {result['npe_epochs']} epochs in {result['elapsed_s']:.0f}s")
```

**Step 2: Verify it parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_field_npe.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add scripts/train_field_npe.py
git commit -m "feat: add train_field_npe.py for decoupled NPE training + fiducial FoM"
```

---

### Task 6: Update CLAUDE.md with new pipeline structure

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the Phase 2c section and running instructions**

In the "What's next" section, update Phase 2c to reflect the two-stage design:

```markdown
- **Phase 2c: SBI with Neural Compression** — Two decoupled stages:
  1. Train EfficientNet MSE compressor on tiles (`scripts/train_field_compressor.py`)
  2. Train Gaussian NPE on frozen compressor outputs with on-the-fly augmentation (`scripts/train_field_npe.py`)
  FoM evaluated on held-out fiducial cosmology (sim 109, 12 tiles).
```

Add to the Modal pipeline section:

```markdown
# Field compressor pipeline (Phase 2c)
modal run scripts/train_field_compressor.py --lmax 200 --noise-level des_y3
modal run scripts/train_field_npe.py --lmax 200 --noise-level des_y3
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with decoupled compressor/NPE pipeline"
```
