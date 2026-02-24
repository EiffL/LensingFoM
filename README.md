# LensingFoM

Comparison of Figure of Merit (FoM) for weak lensing cosmological constraints: simulation-based inference (SBI) vs 2-point function analysis, as a function of maximum multipole ell_max.

## Phase 1: Born Raytracing Validation

The first phase processes [Gower Street](http://star.ucl.ac.uk/GowerStreetSims/) N-body simulations through a Born-approximation raytracing pipeline with DES Y3 source galaxy redshift distributions, and validates the output convergence power spectra against CAMB Halofit theory predictions.

### What it does

1. Loads particle lightcone shells from an N-body simulation (HEALPix nside=2048)
2. Converts particle counts to overdensity fields
3. Applies Born lensing weights using DES Y3 n(z) for 4 tomographic bins
4. Produces full-sky convergence maps at nside=1024
5. Measures auto- and cross-power spectra (10 total: 4 auto + 6 cross)
6. Compares against CAMB theory predictions

### Local quick start (single simulation)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh    # Downloads ~3 GB of simulation data
jupyter notebook notebooks/01_validate_raytracing.ipynb
```

### Processing all 791 simulations with Modal

The `pipeline.py` script runs the Born raytracing pipeline on all 791 Gower Street simulations in parallel using [Modal](https://modal.com). Each simulation is processed in its own container: download the tarball, extract, raytrace at nside=1024, and save the 4 tomographic convergence maps to a persistent Modal Volume.

#### Prerequisites

1. Install Modal and authenticate:

```bash
pip install modal
modal setup   # opens browser to log in
```

2. That's it. All dependencies (healpy, CAMB, astropy, etc.) are installed inside the Modal image automatically. The DES Y3 n(z) FITS file is downloaded at image build time so it's shared across all containers.

#### Run a single simulation (test)

```bash
modal run pipeline.py --sim-id 1
```

This takes ~10 minutes and is a good sanity check before launching the full run. Inspect the output on the volume:

```bash
modal volume ls lensing-results sim00001/
```

#### Run all 791 simulations

```bash
modal run pipeline.py
```

This processes all simulations in parallel (up to 20 concurrent containers, to avoid overwhelming the UCL download server). Each simulation takes ~10 minutes, so the full run completes in ~7 hours. The run is idempotent — simulations that already have output on the volume are skipped, so you can safely re-run after failures.

At the end, a summary is printed:

```
785 succeeded, 6 failed
  sim00123: <error message>
  ...
```

#### Output format

Results are stored on the `lensing-results` Modal Volume:

```
sim00001/
  kappa_maps.npz    # keys: kappa_bin0..kappa_bin3, each nside=1024 (12.6M pixels)
  metadata.json     # {"sim_id": 1, "nside": 1024, "cosmo_params": {...}, "n_shells_processed": 77}
sim00002/
  ...
```

#### Downloading results locally

```bash
# Single simulation
modal volume get lensing-results sim00001/kappa_maps.npz ./sim00001_kappa.npz

# All results (creates local sim*/ directories)
modal volume get lensing-results . ./results/
```

### Data

- **Gower Street simulations**: 791 N-body lightcone simulations with varied cosmologies (Omega_m, sigma_8, w, h, n_s). Each is ~3 GB compressed.
- **DES Y3 MagLim n(z)**: 4 tomographic source bins from the Dark Energy Survey Year 3 analysis

### Validation Results

The Born raytracing C_ell agrees with CAMB Halofit theory to within:
- Cosmic variance at low ell (< 100)
- ~1% at intermediate ell (~200–500)
- ~20% suppression at high ell (> 1000) due to resolution and non-linear effects beyond Halofit

## Phase 2a: Tile Extraction & HuggingFace Dataset

Full-sky convergence maps are filtered in harmonic space and projected onto flat tiles for downstream analysis (SBI with neural compression, or pseudo-Cl 2-point functions).

### How it works

1. **Harmonic filtering**: `map2alm(lmax)` → `rotate_alm(euler)` → `alm2map(nside_down)` ensures all tiles see identical harmonic-space processing
2. **Rotation augmentation**: 3 fixed orientations of the sphere (identity, 90° about y, 90° about z) provide data augmentation while keeping tiles in equatorial positions (minimal projection distortion)
3. **Tile extraction**: 4 equatorial HEALPix base tiles (faces 4–7, ~3400 deg² each) are extracted per orientation, giving 12 tiles per simulation per lmax

### Configurations

| lmax | Tile size | Angular scales | nside |
|------|-----------|---------------|-------|
| 200  | 128×128   | > 0.9°        | 128   |
| 400  | 256×256   | > 0.45°       | 256   |
| 600  | 256×256   | > 0.3°        | 256   |
| 800  | 512×512   | > 0.23°       | 512   |
| 1000 | 512×512   | > 0.18°       | 512   |

### Shape noise levels

Tiles are generated at 3 noise levels to study the impact of survey depth on constraining power:

| Noise level | Survey | n_eff (arcmin^-2/bin) | sigma_e |
|-------------|--------|----------------------|---------|
| `noiseless` | — | — | — |
| `des_y3` | DES Y3 | 1.46–1.48 | 0.24–0.30 |
| `lsst_y10` | LSST Y10 | 6.75 | 0.26 |

Shape noise is added as Gaussian pixel noise to the full-sky nside=1024 convergence map **before** harmonic filtering, so the noise is band-limited consistently with the signal. DES Y3 values from [Amon et al. 2022](https://arxiv.org/abs/2105.13543), LSST Y10 from [DESC SRD](https://arxiv.org/abs/1809.01669).

### Running tile extraction (Stage 2)

```bash
# Single simulation
modal run pipeline.py --stage 2 --sim-id 1

# All simulations (generates 15 tile files per sim: 5 lmax x 3 noise levels)
modal run pipeline.py --stage 2
```

### HuggingFace Dataset

The extracted tiles are published as a HuggingFace dataset at [`EiffL/GowerStreetDESY3`](https://huggingface.co/datasets/EiffL/GowerStreetDESY3).

```python
from datasets import load_dataset

# Load a specific (lmax, noise_level) configuration
ds = load_dataset("EiffL/GowerStreetDESY3", data_dir="data/lmax_600_des_y3")
sample = ds["train"][0]
kappa = sample["kappa"]           # (4, 256, 256) convergence map
omega_m = sample["Omega_m"]       # Matter density parameter
noise = sample["noise_level"]     # "des_y3"
```

Each sample includes the convergence map tile (4 tomographic bins), noise level, and all cosmological parameters (Omega_m, sigma_8, S8, w, h, n_s, Omega_b, m_nu). Components are named `lmax_{lmax}_{noise_level}` (15 total = 5 lmax x 3 noise levels).

#### Building and pushing the dataset

```bash
# Build parquet shards on Modal volume (all 15 configs)
modal run scripts/build_hf_dataset.py

# Build specific config
modal run scripts/build_hf_dataset.py --lmax 600 --noise-level des_y3

# Push to HuggingFace Hub (requires huggingface-secret on Modal)
modal run scripts/push_hf_dataset.py
```

### Spectra dataset

For 2-point function analysis, flat-sky auto/cross power spectra (10 per tile) are pre-computed from the tiles using 2D FFT with azimuthal binning.

```bash
# Build spectra for all (lmax, noise_level) configs
modal run scripts/build_spectra_dataset.py

# Single config
modal run scripts/build_spectra_dataset.py --lmax 600 --noise-level des_y3
```
