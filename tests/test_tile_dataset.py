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
