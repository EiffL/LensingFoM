# Shape Noise Injection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add survey-realistic shape noise (DES Y3 and LSST Y10) to convergence maps before harmonic filtering, and generate HuggingFace dataset components for each (lmax, noise_level) combination.

**Architecture:** Shape noise is added as Gaussian pixel noise to the full-sky nside=1024 convergence map before harmonic filtering and rotation. Noise variance per pixel per bin is `sigma_e^2 / (2 * n_gal * A_pix)`. The same noisy map is reused across all lmax cuts (the harmonic filter selects different scales from the same realization). Three noise levels: noiseless, des_y3, lsst_y10.

**Tech Stack:** numpy, healpy, Modal (pipeline), HuggingFace datasets

**Design doc:** `docs/plans/2026-02-24-shape-noise-design.md`

---

### Task 1: Add noise config and `add_shape_noise` function to `lensing/tiles.py`

**Files:**
- Modify: `lensing/tiles.py:1-26` (add constants after existing constants)
- Modify: `lensing/tiles.py:60-61` (add new function between `healpix_to_tile` and `filter_and_rotate`)
- Test: `tests/test_tiles.py`

**Step 1: Write the failing tests**

Add to `tests/test_tiles.py`:

```python
from lensing.tiles import NOISE_CONFIGS, add_shape_noise


def test_noise_configs_keys():
    """NOISE_CONFIGS has entries for des_y3 and lsst_y10, not noiseless."""
    assert "des_y3" in NOISE_CONFIGS
    assert "lsst_y10" in NOISE_CONFIGS
    assert "noiseless" not in NOISE_CONFIGS
    for key, cfg in NOISE_CONFIGS.items():
        assert len(cfg["n_eff_arcmin2"]) == 4
        assert len(cfg["sigma_e"]) == 4


def test_add_shape_noise_changes_map():
    """Adding shape noise should change the map."""
    nside = 64
    m = np.zeros(hp.nside2npix(nside))
    rng = np.random.default_rng(42)
    noisy = add_shape_noise(m, n_eff_arcmin2=1.5, sigma_e=0.26, rng=rng)
    assert noisy.shape == m.shape
    assert not np.allclose(noisy, m)


def test_add_shape_noise_variance():
    """Noise variance should match the analytic prediction."""
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    n_eff_arcmin2 = 1.5
    sigma_e = 0.26
    rng = np.random.default_rng(42)
    noisy = add_shape_noise(m, n_eff_arcmin2=n_eff_arcmin2, sigma_e=sigma_e, rng=rng)

    # Expected variance
    arcmin2_per_sr = (180 * 60 / np.pi) ** 2
    n_gal_sr = n_eff_arcmin2 * arcmin2_per_sr
    a_pix = 4 * np.pi / npix
    expected_var = sigma_e**2 / (2 * n_gal_sr * a_pix)

    measured_var = np.var(noisy)
    np.testing.assert_allclose(measured_var, expected_var, rtol=0.05)


def test_add_shape_noise_deterministic():
    """Same RNG seed should produce same noise."""
    nside = 64
    m = np.zeros(hp.nside2npix(nside))
    noisy1 = add_shape_noise(m, 1.5, 0.26, rng=np.random.default_rng(42))
    noisy2 = add_shape_noise(m, 1.5, 0.26, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(noisy1, noisy2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tiles.py -v -k "noise"`
Expected: FAIL with ImportError (NOISE_CONFIGS, add_shape_noise not defined)

**Step 3: Implement NOISE_CONFIGS and add_shape_noise**

Add after line 25 in `lensing/tiles.py` (after `EQUATORIAL_TILES`):

```python
# Survey noise configurations: per-bin n_eff (arcmin^-2) and sigma_e
# DES Y3: Amon et al. 2022, Table 1, arXiv:2105.13543
# LSST Y10: DESC SRD, arXiv:1809.01669 (27 arcmin^-2 total / 4 bins)
NOISE_CONFIGS = {
    "des_y3": {
        "n_eff_arcmin2": [1.476, 1.479, 1.484, 1.461],
        "sigma_e": [0.243, 0.262, 0.259, 0.301],
    },
    "lsst_y10": {
        "n_eff_arcmin2": [6.75, 6.75, 6.75, 6.75],
        "sigma_e": [0.26, 0.26, 0.26, 0.26],
    },
}

# Conversion factor: 1 steradian = (180*60/pi)^2 arcmin^2
_ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2


def add_shape_noise(hpx_map, n_eff_arcmin2, sigma_e, rng):
    """Add Gaussian shape noise to a HEALPix convergence map.

    Parameters
    ----------
    hpx_map : np.ndarray
        RING-ordered HEALPix map.
    n_eff_arcmin2 : float
        Effective galaxy number density in arcmin^-2.
    sigma_e : float
        Per-component intrinsic ellipticity dispersion.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    noisy_map : np.ndarray
        Map with added Gaussian noise.
    """
    npix = len(hpx_map)
    a_pix = 4.0 * np.pi / npix  # pixel area in steradians
    n_gal_sr = n_eff_arcmin2 * _ARCMIN2_PER_SR  # convert to sr^-1
    sigma_pix = sigma_e / np.sqrt(2.0 * n_gal_sr * a_pix)
    return hpx_map + rng.normal(0.0, sigma_pix, size=npix)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tiles.py -v -k "noise"`
Expected: all 4 noise tests PASS

**Step 5: Commit**

```bash
git add lensing/tiles.py tests/test_tiles.py
git commit -m "feat: add shape noise injection to tiles module

Add NOISE_CONFIGS with DES Y3 and LSST Y10 survey specs and
add_shape_noise() function for pixel-level Gaussian noise."
```

---

### Task 2: Modify `extract_tiles_for_lmax` to support noise injection

**Files:**
- Modify: `lensing/tiles.py:93-117` (the `extract_tiles_for_lmax` function)
- Test: `tests/test_tiles.py`

**Step 1: Write the failing test**

Add to `tests/test_tiles.py`:

```python
def test_extract_tiles_for_lmax_with_noise():
    """extract_tiles_for_lmax with noise should produce different tiles than noiseless."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    tiles_clean = extract_tiles_for_lmax(m, lmax)
    tiles_noisy = extract_tiles_for_lmax(
        m, lmax, noise_level="des_y3", bin_index=0, rng=np.random.default_rng(99)
    )

    assert tiles_clean.shape == tiles_noisy.shape
    assert not np.allclose(tiles_clean, tiles_noisy)


def test_extract_tiles_for_lmax_noiseless_unchanged():
    """Default (no noise args) should produce identical output to current behavior."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    tiles_default = extract_tiles_for_lmax(m, lmax)
    tiles_explicit = extract_tiles_for_lmax(m, lmax, noise_level="noiseless")

    np.testing.assert_array_equal(tiles_default, tiles_explicit)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tiles.py::test_extract_tiles_for_lmax_with_noise -v`
Expected: FAIL with TypeError (unexpected keyword argument)

**Step 3: Modify extract_tiles_for_lmax**

Replace the function signature and body in `lensing/tiles.py`:

```python
def extract_tiles_for_lmax(hpx_map, lmax_cut, noise_level="noiseless", bin_index=0, rng=None):
    """Extract all 12 equatorial tiles (3 orientations x 4 tiles) for one lmax.

    Parameters
    ----------
    hpx_map : np.ndarray
        Full-sky RING-ordered HEALPix map.
    lmax_cut : int
        Maximum multipole.
    noise_level : str
        Noise level: "noiseless", "des_y3", or "lsst_y10".
    bin_index : int
        Tomographic bin index (0-3), used to look up per-bin noise params.
    rng : np.random.Generator or None
        Random generator for noise. Required when noise_level != "noiseless".

    Returns
    -------
    tiles : np.ndarray, shape (12, nside_down, nside_down)
        Tiles indexed as [orientation * 4 + tile_idx], where tile_idx
        runs over the 4 equatorial tiles.
    """
    nside_down = LMAX_TO_NSIDE[lmax_cut]
    tiles = np.empty((12, nside_down, nside_down), dtype=np.float32)

    # Add noise to the full-sky map before filtering
    if noise_level != "noiseless":
        cfg = NOISE_CONFIGS[noise_level]
        hpx_map = add_shape_noise(
            hpx_map,
            n_eff_arcmin2=cfg["n_eff_arcmin2"][bin_index],
            sigma_e=cfg["sigma_e"][bin_index],
            rng=rng,
        )

    for ori_idx, euler_angles in enumerate(ORIENTATIONS):
        filtered = filter_and_rotate(hpx_map, lmax_cut, euler_angles)
        for tile_idx, tile_id in enumerate(EQUATORIAL_TILES):
            tiles[ori_idx * 4 + tile_idx] = healpix_to_tile(filtered, tile_id)

    return tiles
```

**Step 4: Run all tile tests to verify they pass**

Run: `pytest tests/test_tiles.py -v`
Expected: all tests PASS (including old tests — they still work because default is "noiseless")

**Step 5: Commit**

```bash
git add lensing/tiles.py tests/test_tiles.py
git commit -m "feat: add noise_level parameter to extract_tiles_for_lmax

Injects shape noise into the full-sky map before harmonic filtering
when noise_level is 'des_y3' or 'lsst_y10'."
```

---

### Task 3: Update Modal pipeline `extract_tiles` to loop over noise levels

**Files:**
- Modify: `pipeline.py:32-33` (add NOISE_LEVELS constant)
- Modify: `pipeline.py:128-172` (rewrite `extract_tiles` function)

**Step 1: Update pipeline.py**

Add after line 32 (after `LMAX_VALUES`):

```python
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]
NOISE_LEVEL_INDEX = {name: i for i, name in enumerate(NOISE_LEVELS)}
```

Replace the `extract_tiles` function:

```python
@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=2048,
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

    # Check if already done (all lmax x noise_level files exist)
    all_tile_paths = [
        sim_dir / f"tiles_lmax{lmax}_noise_{nl}.npz"
        for lmax in LMAX_VALUES
        for nl in NOISE_LEVELS
    ]
    if all(p.exists() for p in all_tile_paths):
        print(f"{sim_tag}: all tiles already exist, skipping")
        return {"sim_id": sim_id, "status": "skipped"}

    # Load full-sky kappa maps
    data = np.load(npz_path)
    kappa_maps = [data[f"kappa_bin{i}"] for i in range(4)]

    for noise_level in NOISE_LEVELS:
        noise_idx = NOISE_LEVEL_INDEX[noise_level]
        # Deterministic seed per (sim_id, noise_level)
        rng = np.random.default_rng(sim_id * 1000 + noise_idx)
        # Pre-generate one noise realization per bin (reused across lmax)
        if noise_level != "noiseless":
            from lensing.tiles import NOISE_CONFIGS, add_shape_noise
            cfg = NOISE_CONFIGS[noise_level]
            noisy_maps = [
                add_shape_noise(
                    kappa_maps[b],
                    n_eff_arcmin2=cfg["n_eff_arcmin2"][b],
                    sigma_e=cfg["sigma_e"][b],
                    rng=rng,
                )
                for b in range(4)
            ]
        else:
            noisy_maps = kappa_maps

        for lmax in LMAX_VALUES:
            tile_path = sim_dir / f"tiles_lmax{lmax}_noise_{noise_level}.npz"
            if tile_path.exists():
                continue

            nside_down = LMAX_TO_NSIDE[lmax]
            # Shape: (12, 4, nside_down, nside_down)
            all_tiles = np.stack(
                [extract_tiles_for_lmax(noisy_maps[b], lmax) for b in range(4)],
                axis=1,
            )
            np.savez_compressed(tile_path, tiles=all_tiles)
            print(f"{sim_tag}: saved {tile_path.name} shape={all_tiles.shape}")

    vol.commit()
    return {"sim_id": sim_id, "status": "ok"}
```

Note: The pipeline now adds noise at the full-sky level _once_ per (sim, noise_level), then reuses the same noisy maps across all lmax values. This means `extract_tiles_for_lmax` is called with the default `noise_level="noiseless"` — the noise was already baked in.

**Step 2: Verify the pipeline module imports correctly**

Run: `python -c "import pipeline; print('OK')"`
Expected: OK (or Modal import warning, but no syntax errors)

**Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: extract tiles for all (lmax, noise_level) combinations

Pipeline Stage 2 now loops over noiseless/des_y3/lsst_y10 noise levels.
Each noise level gets a deterministic RNG per sim, with one noise
realization shared across all lmax values."
```

---

### Task 4: Update `build_hf_dataset.py` for noise levels

**Files:**
- Modify: `scripts/build_hf_dataset.py`

**Step 1: Update the script**

Key changes:
- Add `NOISE_LEVELS` constant
- Change `build_parquet_for_lmax` to `build_parquet_for_config` taking `(lmax, noise_level)` tuple
- Update file naming: `tiles_lmax{lmax}_noise_{noise_level}.npz`
- Output directory: `hf_dataset/lmax_{lmax}_{noise_level}/`
- Add `noise_level` column to each record
- Update `main()` to iterate over all (lmax, noise_level) pairs

Replace the full script:

```python
"""Build a HuggingFace dataset from extracted tiles on the Modal volume.

Runs on Modal for direct volume access. Builds parquet shards per
(lmax, noise_level), then pushes to HuggingFace Hub.

Usage:
    # Build parquet shards (all configs)
    modal run scripts/build_hf_dataset.py

    # Build for a specific config
    modal run scripts/build_hf_dataset.py --lmax 200 --noise-level des_y3

    # Push to HuggingFace Hub (requires HF_TOKEN secret on Modal)
    modal run scripts/build_hf_dataset.py --push
"""

import csv
from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]
LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]
N_TILES = 12  # 3 orientations x 4 equatorial tiles
SIMS_PER_SHARD = 50

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow", "datasets", "huggingface_hub")
    .add_local_file("gower_street_runs.csv", "/pipeline/gower_street_runs.csv")
)

app = modal.App("lensing-hf-dataset", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"
CSV_PATH = "/pipeline/gower_street_runs.csv"


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


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=8192,
)
def build_parquet_for_config(lmax: int, noise_level: str) -> dict:
    """Build parquet shards for one (lmax, noise_level) combination."""
    import numpy as np
    from datasets import Dataset

    cosmo_params = load_cosmo_params(CSV_PATH)
    nside = LMAX_TO_NSIDE[lmax]

    out_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    shard_idx = 0
    skipped = 0

    for shard_start in range(1, 792, SIMS_PER_SHARD):
        shard_end = min(shard_start + SIMS_PER_SHARD, 792)
        records = []

        for sim_id in range(shard_start, shard_end):
            sim_tag = f"sim{sim_id:05d}"
            npz_path = Path(RESULTS_DIR) / sim_tag / f"tiles_lmax{lmax}_noise_{noise_level}.npz"

            if not npz_path.exists():
                skipped += 1
                continue

            if sim_id not in cosmo_params:
                skipped += 1
                continue

            data = np.load(npz_path)
            tiles = data["tiles"]  # shape (12, 4, nside, nside)
            cp = cosmo_params[sim_id]

            for tile_idx in range(N_TILES):
                records.append({
                    "kappa": tiles[tile_idx].tolist(),  # (4, nside, nside)
                    "sim_id": sim_id,
                    "orientation_id": tile_idx // 4,
                    "tile_id": tile_idx % 4,
                    "noise_level": noise_level,
                    "Omega_m": cp["Omega_m"],
                    "sigma_8": cp["sigma_8"],
                    "S8": cp["S8"],
                    "w": cp["w"],
                    "h": cp["h"],
                    "n_s": cp["n_s"],
                    "Omega_b": cp["Omega_b"],
                    "m_nu": cp["m_nu"],
                })

        if records:
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            ds = Dataset.from_list(records)
            ds.to_parquet(str(shard_path))
            total_tiles += len(records)
            print(f"lmax={lmax}/{noise_level}: wrote {shard_path.name} ({len(records)} tiles, sims {shard_start}-{shard_end - 1})")
            del records, ds
        shard_idx += 1

    vol.commit()
    print(f"lmax={lmax}/{noise_level}: done — {total_tiles} tiles in {shard_idx} shards, {skipped} sims skipped")
    return {"lmax": lmax, "noise_level": noise_level, "total_tiles": total_tiles, "shards": shard_idx, "skipped": skipped}


@app.local_entrypoint()
def main(lmax: int = None, noise_level: str = None, push: bool = False):
    if push:
        print("Use scripts/push_hf_dataset.py for pushing to HuggingFace Hub.")
        return

    if lmax is not None and noise_level is not None:
        result = build_parquet_for_config.remote(lmax, noise_level)
        print(result)
    else:
        # Build all (lmax, noise_level) combinations
        configs = [(l, n) for l in LMAX_VALUES for n in NOISE_LEVELS]
        results = list(build_parquet_for_config.starmap(configs))
        for r in results:
            print(r)
        print(f"\nAll {len(configs)} configs built. Use scripts/push_hf_dataset.py to upload.")
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/build_hf_dataset.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add scripts/build_hf_dataset.py
git commit -m "feat: build HF dataset for all (lmax, noise_level) configs

Each (lmax, noise_level) pair becomes a separate HuggingFace component
with directory lmax_{lmax}_{noise_level}/. Records include noise_level column."
```

---

### Task 5: Update `push_hf_dataset.py` for noise levels

**Files:**
- Modify: `scripts/push_hf_dataset.py`

**Step 1: Update the script**

Key changes:
- Add `NOISE_LEVELS` constant
- Loop over `(lmax, noise_level)` pairs
- Update directory and repo paths
- Add `--noise-level` CLI filter

Replace the full script:

```python
"""Push parquet shards from the Modal volume to HuggingFace Hub.

Usage:
    modal run scripts/push_hf_dataset.py
    modal run scripts/push_hf_dataset.py --lmax 200 --noise-level des_y3

Requires a Modal secret named 'huggingface-secret' with key HF_TOKEN.
Create it with:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
"""

from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
)

app = modal.App("lensing-push-hf", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=4096,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_hub(
    repo_id: str = "EiffL/GowerStreetDESY3",
    lmax_filter: int = None,
    noise_filter: str = None,
) -> dict:
    """Push parquet shards to HuggingFace Hub."""
    import os
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    hf_dir = Path(RESULTS_DIR) / "hf_dataset"
    lmax_list = [lmax_filter] if lmax_filter else LMAX_VALUES
    noise_list = [noise_filter] if noise_filter else NOISE_LEVELS
    pushed = {}

    for lmax in lmax_list:
        for noise_level in noise_list:
            config_name = f"lmax_{lmax}_{noise_level}"
            config_dir = hf_dir / config_name
            if not config_dir.exists():
                print(f"{config_name}: no parquet shards found, skipping")
                continue

            parquet_files = sorted(config_dir.glob("shard_*.parquet"))
            if not parquet_files:
                print(f"{config_name}: no parquet files, skipping")
                continue

            for pf in parquet_files:
                api.upload_file(
                    path_or_fileobj=str(pf),
                    path_in_repo=f"data/{config_name}/{pf.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"{config_name}: uploaded {pf.name}")

            pushed[config_name] = len(parquet_files)
            print(f"{config_name}: pushed {len(parquet_files)} shards")

    return {"repo_id": repo_id, "pushed": pushed}


@app.local_entrypoint()
def main(lmax: int = None, noise_level: str = None):
    result = push_to_hub.remote(lmax_filter=lmax, noise_filter=noise_level)
    print(result)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/push_hf_dataset.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add scripts/push_hf_dataset.py
git commit -m "feat: push HF dataset with (lmax, noise_level) component structure

Components are named lmax_{lmax}_{noise_level} (e.g. lmax_200_des_y3).
Supports --lmax and --noise-level filters."
```

---

### Task 6: Update CLAUDE.md with noise level documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add to the "What's been done" section a note about noise support. Update the "Key Conventions" section. Add noise-related Modal commands.

In the "Key Conventions" section, add:
```
- 3 noise levels: "noiseless" (no noise), "des_y3" (DES Y3 shape noise), "lsst_y10" (LSST Y10 shape noise)
- Noise added at full-sky nside=1024 level before harmonic filtering
- HuggingFace dataset components: lmax_{lmax}_{noise_level} (15 total = 5 lmax x 3 noise)
```

In the Modal pipeline section, update Stage 2 description and add build/push commands:
```bash
# Stage 2: Tile extraction with noise levels (all sims)
modal run pipeline.py --stage 2

# Build HuggingFace dataset (all configs)
modal run scripts/build_hf_dataset.py

# Build for specific config
modal run scripts/build_hf_dataset.py --lmax 200 --noise-level des_y3

# Push to HuggingFace Hub
modal run scripts/push_hf_dataset.py
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with noise level documentation"
```

---

### Task 7: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: all tests PASS

**Step 2: Verify pipeline imports**

Run: `python -c "from lensing.tiles import NOISE_CONFIGS, add_shape_noise, extract_tiles_for_lmax; print('tiles OK')"`
Run: `python -c "import ast; ast.parse(open('pipeline.py').read()); print('pipeline OK')"`
Run: `python -c "import ast; ast.parse(open('scripts/build_hf_dataset.py').read()); print('build OK')"`
Run: `python -c "import ast; ast.parse(open('scripts/push_hf_dataset.py').read()); print('push OK')"`

Expected: all print OK
