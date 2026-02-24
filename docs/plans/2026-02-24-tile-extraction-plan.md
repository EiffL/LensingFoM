# Tile Extraction + HuggingFace Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract lmax-filtered equatorial tiles from all 791 Gower Street sims (3 orientations x 4 tiles = 12 per sim), and publish as a HuggingFace dataset sharded by lmax.

**Architecture:** Extend `lensing/tiles.py` with rotation + downsampling logic. Add a Stage 2 Modal function in `pipeline.py` that reads existing `kappa_maps.npz` from the volume and writes compact tile npz files. Then a local script builds the HuggingFace dataset from the volume.

**Tech Stack:** healpy (SHT, rotation), modal (cloud compute), huggingface datasets

---

### Task 1: Extend `lensing/tiles.py` with rotation and tile extraction pipeline

**Files:**
- Modify: `lensing/tiles.py`
- Create: `tests/test_tiles.py`

**Context:** The existing `tiles.py` has `healpix_to_tile`, `healpix_to_all_tiles`, and `filter_map_lmax`. We need to add the rotation logic and a high-level function that produces all 12 equatorial tiles for a given kappa map at a given lmax.

**Step 1: Write the tests**

```python
# tests/test_tiles.py
"""Tests for tile extraction pipeline."""
import healpy as hp
import numpy as np
import pytest

from lensing.tiles import (
    EQUATORIAL_TILES,
    LMAX_TO_NSIDE,
    ORIENTATIONS,
    extract_tiles_for_lmax,
    filter_and_rotate,
    healpix_to_tile,
)


def test_lmax_to_nside_mapping():
    """Each lmax maps to a specific downsampled nside."""
    assert LMAX_TO_NSIDE[200] == 128
    assert LMAX_TO_NSIDE[400] == 256
    assert LMAX_TO_NSIDE[600] == 256
    assert LMAX_TO_NSIDE[800] == 512
    assert LMAX_TO_NSIDE[1000] == 512


def test_orientations():
    """Three orientations defined with ZYZ Euler angles in degrees."""
    assert len(ORIENTATIONS) == 3
    assert ORIENTATIONS[0] == (0, 0, 0)  # identity


def test_filter_and_rotate_identity():
    """Identity rotation should just filter (up to SHT round-trip noise)."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    result = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    nside_out = LMAX_TO_NSIDE.get(lmax_cut, nside)

    # For lmax not in LMAX_TO_NSIDE, output nside = input nside
    assert hp.npix2nside(len(result)) == nside_out or hp.npix2nside(len(result)) == nside


def test_filter_and_rotate_changes_map():
    """Non-identity rotation should produce a different map."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    m_id = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    m_rot = filter_and_rotate(m, lmax_cut, euler_angles=(0, 90, 0))

    assert not np.allclose(m_id, m_rot)


def test_filter_and_rotate_preserves_power():
    """Rotation should preserve the total power (C_ell is rotationally invariant)."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    m_id = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    m_rot = filter_and_rotate(m, lmax_cut, euler_angles=(0, 90, 0))

    cl_id = hp.anafast(m_id, lmax=lmax_cut)
    cl_rot = hp.anafast(m_rot, lmax=lmax_cut)

    np.testing.assert_allclose(cl_id, cl_rot, rtol=0.05)


def test_extract_tiles_for_lmax_shape():
    """extract_tiles_for_lmax returns (12, nside_down, nside_down) array."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    # Need lmax < 3*nside for map2alm to work
    tiles = extract_tiles_for_lmax(m, lmax)

    nside_down = LMAX_TO_NSIDE[lmax]
    assert tiles.shape == (12, nside_down, nside_down)


def test_extract_tiles_for_lmax_orientations():
    """Tiles from different orientations should differ."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    tiles = extract_tiles_for_lmax(m, lmax)

    # Tile 0 (orientation 0, equatorial tile 4) vs tile 4 (orientation 1, equatorial tile 4)
    assert not np.allclose(tiles[0], tiles[4])
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_tiles.py -v`
Expected: FAIL — `LMAX_TO_NSIDE`, `ORIENTATIONS`, `filter_and_rotate`, `extract_tiles_for_lmax` not defined.

**Step 3: Implement the new functions in `lensing/tiles.py`**

Replace the full file with:

```python
"""Tile extraction from HEALPix maps with harmonic filtering and rotation."""

import healpy as hp
import numpy as np


# lmax -> downsampled nside mapping (Nyquist: nside >= lmax/2, round up to power of 2)
LMAX_TO_NSIDE = {
    200: 128,
    400: 256,
    600: 256,
    800: 512,
    1000: 512,
}

# Three orientations as ZYZ Euler angles in degrees.
# Each maps a different set of 4 base tiles into equatorial positions.
ORIENTATIONS = [
    (0, 0, 0),       # identity — equatorial tiles 4-7 are already equatorial
    (0, 90, 0),      # 90 deg about y — rotates polar caps to equator
    (90, 0, 0),      # 90 deg about x — third independent orientation
]

# Equatorial tile indices (least distorted when projected to 2D)
EQUATORIAL_TILES = [4, 5, 6, 7]


def healpix_to_tile(hpx_map, tile_id):
    """Extract a single square 2D image from a HEALPix map for one base tile.

    In NESTED ordering, each of the 12 base tiles maps to an nside x nside
    square grid via the Z-order (Morton) curve.

    Parameters
    ----------
    hpx_map : np.ndarray
        HEALPix map in RING ordering.
    tile_id : int
        Base tile index (0-11).

    Returns
    -------
    tile : np.ndarray, shape (nside, nside)
    """
    nside = hp.npix2nside(len(hpx_map))

    # Pixel indices belonging to this face in NESTED ordering
    start = tile_id * nside * nside
    end = (tile_id + 1) * nside * nside
    pix_nest = np.arange(start, end)

    x, y, _ = hp.pix2xyf(nside, pix_nest, nest=True)

    # Convert those NESTED pixel indices to RING to index into the map
    pix_ring = hp.nest2ring(nside, pix_nest)

    tile = np.empty((nside, nside))
    tile[x, y] = hpx_map[pix_ring]
    return tile


def filter_and_rotate(hpx_map, lmax_cut, euler_angles=(0, 0, 0)):
    """Filter a HEALPix map to lmax_cut, optionally rotate, and downsample.

    All maps go through the same harmonic-space processing:
    map -> alm(lmax) -> rotate_alm -> alm2map(nside_down)

    Parameters
    ----------
    hpx_map : np.ndarray
        RING-ordered HEALPix map at nside=1024 (or any nside).
    lmax_cut : int
        Maximum multipole to retain.
    euler_angles : tuple of 3 floats
        ZYZ Euler angles in degrees for the rotation.

    Returns
    -------
    filtered_map : np.ndarray
        RING-ordered map at nside_down resolution.
    """
    nside_down = LMAX_TO_NSIDE.get(lmax_cut, hp.npix2nside(len(hpx_map)))

    alm = hp.map2alm(hpx_map, lmax=lmax_cut)

    if euler_angles != (0, 0, 0):
        rot = hp.Rotator(rot=list(euler_angles), deg=True, eulertype="ZYZ")
        alm = rot.rotate_alm(alm, lmax=lmax_cut)

    return hp.alm2map(alm, nside_down)


def extract_tiles_for_lmax(hpx_map, lmax_cut):
    """Extract all 12 equatorial tiles (3 orientations x 4 tiles) for one lmax.

    Parameters
    ----------
    hpx_map : np.ndarray
        Full-sky RING-ordered HEALPix map.
    lmax_cut : int
        Maximum multipole.

    Returns
    -------
    tiles : np.ndarray, shape (12, nside_down, nside_down)
        Tiles indexed as [orientation * 4 + tile_idx], where tile_idx
        runs over the 4 equatorial tiles.
    """
    nside_down = LMAX_TO_NSIDE[lmax_cut]
    tiles = np.empty((12, nside_down, nside_down), dtype=np.float32)

    for ori_idx, euler_angles in enumerate(ORIENTATIONS):
        filtered = filter_and_rotate(hpx_map, lmax_cut, euler_angles)
        for tile_idx, tile_id in enumerate(EQUATORIAL_TILES):
            tiles[ori_idx * 4 + tile_idx] = healpix_to_tile(filtered, tile_id)

    return tiles
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_tiles.py -v`
Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add lensing/tiles.py tests/test_tiles.py
git commit -m "feat: add rotation + downsampling to tile extraction pipeline"
```

---

### Task 2: Add Stage 2 Modal function for tile extraction

**Files:**
- Modify: `pipeline.py`

**Context:** The existing `pipeline.py` has a Stage 1 function `process_simulation` that produces `kappa_maps.npz`. We add a Stage 2 function `extract_tiles` that reads those and writes tile files. This runs as a second `.map()` pass over all sim IDs.

**Step 1: Write the Stage 2 function in `pipeline.py`**

Add after the existing `process_simulation` function:

```python
LMAX_VALUES = [200, 400, 600, 800, 1000]


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=1800,
    memory=4096,
)
def extract_tiles(sim_id: int) -> dict:
    import numpy as np

    from lensing.tiles import LMAX_TO_NSIDE, extract_tiles_for_lmax

    sim_tag = f"sim{sim_id:05d}"
    sim_dir = Path(RESULTS_DIR) / sim_tag
    npz_path = sim_dir / "kappa_maps.npz"

    # Check source exists
    if not npz_path.exists():
        print(f"{sim_tag}: no kappa_maps.npz, skipping")
        return {"sim_id": sim_id, "status": "no_source"}

    # Check if already done (all lmax files exist)
    tile_paths = [sim_dir / f"tiles_lmax{lmax}.npz" for lmax in LMAX_VALUES]
    if all(p.exists() for p in tile_paths):
        print(f"{sim_tag}: tiles already exist, skipping")
        return {"sim_id": sim_id, "status": "skipped"}

    # Load full-sky kappa maps
    data = np.load(npz_path)
    kappa_maps = [data[f"kappa_bin{i}"] for i in range(4)]

    for lmax in LMAX_VALUES:
        tile_path = sim_dir / f"tiles_lmax{lmax}.npz"
        if tile_path.exists():
            continue

        nside_down = LMAX_TO_NSIDE[lmax]
        # Shape: (12, 4, nside_down, nside_down)
        all_tiles = np.stack(
            [extract_tiles_for_lmax(kappa_maps[b], lmax) for b in range(4)],
            axis=1,
        )
        np.savez_compressed(tile_path, tiles=all_tiles)
        print(f"{sim_tag}: saved {tile_path.name} shape={all_tiles.shape}")

    vol.commit()
    return {"sim_id": sim_id, "status": "ok"}
```

Update the `main` entrypoint to support both stages:

```python
@app.local_entrypoint()
def main(sim_id: int = None, stage: int = 1):
    if stage == 1:
        if sim_id is not None:
            result = process_simulation.remote(sim_id)
            print(result)
        else:
            sim_ids = list(range(1, 792))
            results = list(process_simulation.map(sim_ids, return_exceptions=True))
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [
                (sim_ids[i], r)
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            print(f"{len(successes)} succeeded, {len(failures)} failed")
            for sid, err in failures:
                print(f"  sim{sid:05d}: {err}")
    elif stage == 2:
        if sim_id is not None:
            result = extract_tiles.remote(sim_id)
            print(result)
        else:
            sim_ids = list(range(1, 792))
            results = list(extract_tiles.map(sim_ids, return_exceptions=True))
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [
                (sim_ids[i], r)
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            print(f"Stage 2: {len(successes)} succeeded, {len(failures)} failed")
            for sid, err in failures:
                print(f"  sim{sid:05d}: {err}")
```

**Step 2: Test locally with a single sim**

Run: `modal run pipeline.py --stage 2 --sim-id 1`
Expected: Produces `tiles_lmax200.npz` through `tiles_lmax1000.npz` under `sim00001/` on the volume.

**Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: add Stage 2 tile extraction to Modal pipeline"
```

---

### Task 3: Run Stage 2 on all completed sims

**Step 1: Launch the full Stage 2 run**

Run: `modal run pipeline.py --stage 2`

This maps over all 791 sim IDs. Sims without `kappa_maps.npz` are skipped. Each worker takes ~2-5 min per sim (dominated by SHT at nside=1024).

**Step 2: Verify completion**

```python
import modal
vol = modal.Volume.from_name("lensing-results")
# Count sims with all tile files
complete = 0
for sim_id in range(1, 792):
    sim_tag = f"sim{sim_id:05d}"
    try:
        entries = [e.path for e in vol.listdir(f"/{sim_tag}")]
        if all(f"/{sim_tag}/tiles_lmax{lmax}.npz" in entries for lmax in [200,400,600,800,1000]):
            complete += 1
    except:
        pass
print(f"{complete} sims fully tiled")
```

---

### Task 4: Build HuggingFace dataset

**Files:**
- Create: `scripts/build_hf_dataset.py`

**Context:** Read all tile npz files from the Modal volume, combine with cosmological parameters from `gower_street_runs.csv`, and push to HuggingFace Hub. Shard by lmax so users can load individual resolutions.

**Step 1: Write the dataset builder script**

```python
# scripts/build_hf_dataset.py
"""Build a HuggingFace dataset from extracted tiles on the Modal volume."""

import csv
import io
import tempfile
from pathlib import Path

import datasets
import modal
import numpy as np

LMAX_VALUES = [200, 400, 600, 800, 1000]
LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
N_TILES = 12  # 3 orientations x 4 equatorial tiles
N_BINS = 4


def load_cosmo_params(csv_path):
    """Load all cosmological parameters from gower_street_runs.csv."""
    params = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # category row
        next(reader)  # column names
        for row in reader:
            sim_id = int(row[0])
            omega_m = float(row[3])
            sigma_8 = float(row[4])
            params[sim_id] = {
                "Omega_m": omega_m,
                "sigma_8": sigma_8,
                "S8": sigma_8 * (omega_m / 0.3) ** 0.5,
                "w": float(row[5]),
                "Omega_bh2": float(row[6]),
                "h": float(row[7]),
                "n_s": float(row[8]),
                "m_nu": float(row[9]),
                "Omega_b": float(row[10]),
            }
    return params


def main():
    vol = modal.Volume.from_name("lensing-results")
    cosmo_params = load_cosmo_params("gower_street_runs.csv")

    for lmax in LMAX_VALUES:
        nside = LMAX_TO_NSIDE[lmax]
        print(f"\n=== Building dataset for lmax={lmax} (nside={nside}) ===")

        records = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for sim_id in range(1, 792):
                sim_tag = f"sim{sim_id:05d}"
                remote_path = f"/{sim_tag}/tiles_lmax{lmax}.npz"

                try:
                    # Download tile file from volume
                    local_path = Path(tmpdir) / f"{sim_tag}_lmax{lmax}.npz"
                    with open(local_path, "wb") as f:
                        for chunk in vol.read_file(remote_path):
                            f.write(chunk)
                except Exception:
                    continue

                data = np.load(local_path)
                tiles = data["tiles"]  # shape (12, 4, nside, nside)

                if sim_id not in cosmo_params:
                    continue

                cp = cosmo_params[sim_id]

                for tile_idx in range(N_TILES):
                    orientation_id = tile_idx // 4
                    equatorial_tile = tile_idx % 4

                    records.append({
                        "tiles": tiles[tile_idx],  # shape (4, nside, nside)
                        "sim_id": sim_id,
                        "lmax": lmax,
                        "orientation_id": orientation_id,
                        "tile_id": equatorial_tile,
                        "Omega_m": cp["Omega_m"],
                        "sigma_8": cp["sigma_8"],
                        "S8": cp["S8"],
                        "w": cp["w"],
                        "h": cp["h"],
                        "n_s": cp["n_s"],
                        "Omega_b": cp["Omega_b"],
                        "m_nu": cp["m_nu"],
                    })

                if sim_id % 100 == 0:
                    print(f"  Processed {sim_id}/791 sims, {len(records)} tiles so far")

        print(f"  Total tiles: {len(records)}")

        # Build HuggingFace dataset
        ds = datasets.Dataset.from_list(records)
        ds.push_to_hub(
            "EiffL/lensing-tiles",
            config_name=f"lmax_{lmax}",
            commit_message=f"Add lmax={lmax} tiles ({nside}x{nside})",
        )
        print(f"  Pushed lmax={lmax} to Hub")


if __name__ == "__main__":
    main()
```

**Step 2: Test with a single lmax on a few sims first**

Modify the script temporarily to only process `lmax=200` and `range(1, 10)` to validate the format before the full push.

**Step 3: Run the full dataset build**

Run: `PYTHONPATH=. .venv/bin/python scripts/build_hf_dataset.py`

**Step 4: Verify the dataset loads correctly**

```python
from datasets import load_dataset
ds = load_dataset("EiffL/lensing-tiles", name="lmax_600")
print(ds)
print(ds[0].keys())
print(ds[0]["tiles"].shape)  # should be (4, 256, 256)
print(ds[0]["Omega_m"], ds[0]["S8"])
```

**Step 5: Commit**

```bash
git add scripts/build_hf_dataset.py
git commit -m "feat: add HuggingFace dataset builder for lensing tiles"
```

---

### Task 5: Add `lenstools` and `datasets` to requirements

**Files:**
- Modify: `requirements.txt`

**Step 1: Add dependencies**

Add to `requirements.txt`:
```
lenstools
datasets
```

**Step 2: Install and verify**

Run: `.venv/bin/pip install lenstools datasets`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add lenstools and datasets to requirements"
```

---

## Execution order

Tasks 1 and 5 can run in parallel. Task 2 depends on Task 1. Task 3 depends on Task 2 (and on Stage 1 being complete for enough sims). Task 4 depends on Task 3.

```
Task 1 (tiles.py) ──→ Task 2 (pipeline.py) ──→ Task 3 (run Stage 2) ──→ Task 4 (HF dataset)
Task 5 (requirements) ─────────────────────────────────────────────────/
```
