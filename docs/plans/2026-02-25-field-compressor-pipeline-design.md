# Field Compressor Pipeline: Decoupled Training with Fiducial FoM

## Goal

Produce a robust FoM estimate for (Omega_m, S8) from neural field-level compression of convergence map tiles, at a given lmax. The pipeline should:

1. Hold out a fiducial cosmology (sim 109, closest to Planck) for FoM evaluation only
2. Decouple compressor training from NPE training so each can be iterated independently
3. Support on-the-fly data augmentation during NPE training (frozen compressor)
4. Report FoM on the 12 fiducial tiles with error bars from tile-to-tile scatter

## Data splitting

### Fiducial holdout

Sim 109 (Omega_m=0.3007, sigma_8=0.8047, S8=0.8055) is the Gower Street simulation closest to Planck fiducial. All 12 of its tiles are held out from all training. They are used only for final FoM evaluation.

### Tile-level split of remaining 790 sims

The remaining 790 sims produce 9480 tiles total (12 tiles per sim). These are split at the tile level (cosmology leakage between splits is acceptable):

| Split | Fraction | ~Tiles | Purpose |
|-------|----------|--------|---------|
| Compressor train | 70% | 6636 | Train EfficientNet MSE regressor |
| NPE train / compressor val | 25% | 2370 | Train Gaussian NPE; also monitor compressor val_loss |
| Test | 5% | 474 | Sanity check of compressor+NPE pipeline |

The split is deterministic (seed=42), shuffled at the tile level.

### Why this split ratio

- The compressor is a large CNN that benefits from maximal training data.
- The NPE is a small 2-layer MLP (64 hidden units) that learns a Gaussian posterior from 2D compressed summaries. 2370 tiles is ample.
- 5% test is a sanity check only; the real evaluation is on the fiducial holdout.

## Architecture: two decoupled stages

### Stage 1: Train compressor (`scripts/train_field_compressor.py`)

**Input**: Parquet tiles for a given lmax/noise_level.

**Model**: `FieldLevelCompressor` (EfficientNet backbone + linear head, MSE loss predicting normalized theta).

**Training details** (unchanged from current):
- Data augmentation: random 90-degree rotations, flips, circular rolls
- Theta noise injection (std=0.005)
- LR schedule: linear warmup then step decay
- Monitor `val_loss` on the NPE split
- bf16 mixed precision

**Outputs saved to Modal volume** (`/results/field_compressor_runs/{tag}/`):
- `best.ckpt` — best checkpoint by val_loss
- `norm_stats.json` — tile_mean, tile_std, theta_mean, theta_std (from compressor train split)
- `result.json` — training metrics (train_mse, val_mse, epochs, elapsed time)

No NPE training, no FoM computation. This script's only job is to produce a good compressor.

### Stage 2: Train NPE and evaluate FoM (`scripts/train_field_npe.py`, new)

**Input**: A trained compressor checkpoint + same parquet tiles.

**Process**:
1. Load compressor from checkpoint, freeze it (eval mode, no grad)
2. Load tiles, apply same normalization (from saved norm_stats.json)
3. Build a `CompressedTileDataset` that applies augmentation + compression on the fly:
   - `__getitem__` returns `(compressed_summary, theta)` where the tile is augmented before compression
   - This means the NPE sees different compressed representations each epoch
4. Train `GaussianNPE` on NPE split with early stopping on test split val_loss
5. Evaluate FoM on fiducial sim 109:
   - Compress all 12 fiducial tiles (no augmentation, deterministic)
   - `npe.predict_fom(compressed_fiducial)` gives 12 FoM values
   - Report: median FoM, std across tiles, min/max
   - Un-normalize to physical units by dividing by `prod(theta_std)`

**Outputs saved to Modal volume** (`/results/field_npe_runs/{tag}/`):
- `npe_best.ckpt` — best NPE checkpoint
- `fom_result.json`:
  ```json
  {
    "lmax": 200,
    "noise_level": "des_y3",
    "fom_fiducial_median": 450.5,
    "fom_fiducial_std": 12.3,
    "fom_fiducial_all": [440.2, 455.1, ...],
    "fom_test_median": 460.8,
    "fom_test_lo_16": 450.0,
    "fom_test_hi_84": 470.5,
    "compressor_checkpoint": "field_mse_lmax200_des_y3_efficientnet_v2_s/best.ckpt"
  }
  ```

### Why on-the-fly compression matters

Pre-computing compressed summaries fixes the representation: the NPE sees the same 2370 points every epoch. With on-the-fly augmentation through the frozen compressor, each tile produces a different 2D summary each time due to random rotations/flips/rolls. This effectively multiplies the NPE training set, making the posterior estimate more robust.

## Data loading changes

### `TileDataModule` modifications

The current `TileDataModule` needs to change:

1. **Filter out fiducial sim** before splitting. This requires the parquet data to contain `sim_id` (it does — the parquet schema includes `sim_id`, `orientation_id`, `tile_id`).
2. **Split 70/25/5** instead of 80/10/10.
3. **Expose split indices** so the NPE script can load the same splits.

The simplest approach: `TileDataModule` accepts a `fiducial_sim_id` parameter. On `setup()`:
- Separate fiducial tiles by sim_id
- Split remaining tiles 70/25/5
- Store fiducial tiles as `self.fiducial_ds`

### New `CompressedTileDataModule` for NPE

A new DataModule for the NPE stage that:
- Takes a frozen compressor model and tile data
- Returns `(compressed_summary, theta)` pairs
- Applies augmentation before compression in `__getitem__`
- Uses the NPE split for training, test split for validation

## FoM reporting

The primary FoM number is the **median over the 12 fiducial tiles**. The tile-to-tile scatter (std or range) captures variance from:
- Different sky orientations (3 rotations)
- Different equatorial tile positions (4 tiles per rotation)
- Cosmic variance within a single cosmology

This is more meaningful than bootstrap error bars on a test set with mixed cosmologies, because the question is: "how well can we constrain parameters at a known fiducial point?"

## Scope

- Single lmax at a time (sweep comes later)
- No comparison to power spectrum FoM (comes later)
- DES Y3 noise level as default
