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
