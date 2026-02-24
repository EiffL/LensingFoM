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

### Modal pipeline

The full pipeline runs on [Modal](https://modal.com) in 5 stages. All stages are idempotent — completed work is skipped on re-run.

| Stage | Name | What it does |
|-------|------|-------------|
| 1 | Born raytracing | Download sims, raytrace, save convergence maps |
| 2 | Tile extraction | Harmonic filter + noise + rotation → flat tiles |
| 3 | Build HF dataset | Convert tiles to parquet shards |
| 4 | Push HF dataset | Upload parquet shards to HuggingFace Hub |
| 5 | Build spectra | Compute auto/cross power spectra from tiles |

#### Prerequisites

```bash
pip install modal
modal setup   # opens browser to log in
```

All dependencies are installed inside the Modal image automatically. Stage 4 requires a Modal secret: `modal secret create huggingface-secret HF_TOKEN=hf_xxx`.

#### Run the full pipeline

```bash
modal run pipeline.py --stage all
```

This runs stages 1→2→3→4→5 sequentially. Each stage prints a summary when done. Safe to re-run at any time.

#### Run individual stages

```bash
modal run pipeline.py --stage 1   # Born raytracing only
modal run pipeline.py --stage 2   # Tile extraction only
modal run pipeline.py --stage 3   # Build HF parquet shards
modal run pipeline.py --stage 4   # Push to HuggingFace Hub
modal run pipeline.py --stage 5   # Build power spectra

# Single simulation (stages 1-2 only)
modal run pipeline.py --stage 1 --sim-id 1
modal run pipeline.py --stage 2 --sim-id 1
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

Tiles are generated at 3 noise levels to study the impact of survey depth on constraining power.

**Noise model.** Shape noise in convergence maps arises from the intrinsic ellipticity dispersion of source galaxies. For a HEALPix pixel at resolution nside, the noise standard deviation per pixel per tomographic bin is:

```
sigma_pix = sigma_e / sqrt(2 * n_eff * A_pix)
```

where `sigma_e` is the per-component intrinsic ellipticity dispersion, `n_eff` is the effective galaxy number density (in sr⁻¹), and `A_pix = 4pi / N_pix` is the pixel solid angle. The factor of 2 accounts for two ellipticity components. Noise is Gaussian and independent per pixel.

Shape noise is added to the full-sky nside=1024 convergence map **before** harmonic filtering, so the noise is band-limited consistently with the signal. For a given (sim_id, noise_level), the same noise realization is shared across all lmax cuts and orientations (the harmonic filter selects different scales from the same underlying noisy field). RNG seed: `sim_id * 1000 + noise_level_index`.

#### DES Y3 ([Amon et al. 2022](https://arxiv.org/abs/2105.13543), Table 1)

| Bin | n_eff (arcmin⁻²) | sigma_e |
|-----|-------------------|---------|
| 0   | 1.476             | 0.243   |
| 1   | 1.479             | 0.262   |
| 2   | 1.484             | 0.259   |
| 3   | 1.461             | 0.301   |

#### LSST Y10 ([DESC SRD](https://arxiv.org/abs/1809.01669))

| Bin | n_eff (arcmin⁻²) | sigma_e |
|-----|-------------------|---------|
| 0–3 | 6.75              | 0.26    |

Total n_eff = 27 arcmin⁻² split uniformly across 4 bins to match the DES tomographic structure.

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

Built by Stage 3 (`modal run pipeline.py --stage 3`), pushed by Stage 4 (`modal run pipeline.py --stage 4`).

### Spectra dataset

For 2-point function analysis, flat-sky auto/cross power spectra are pre-computed from the tiles (Stage 5). Each tile produces 10 binned C_ell (4 auto + 6 cross-spectra for the 4 tomographic bins) in 20 linear ell bins from the fundamental mode to lmax.

**Method.** 2D FFT on each flat tile, followed by azimuthal averaging in annular ell bins. Normalization: `C_ell = (pixel_size² / N²) * |FFT|²`, averaged over modes per bin.

**Output.** One parquet file per (lmax, noise_level) configuration (15 total). Each row is one tile and contains:
- `sim_id`, `orientation_id` (0–2), `tile_id` (0–3), `noise_level`
- `ell_eff`: effective multipole at bin centers (length 20)
- `cl_i_j`: binned power spectrum for bin pair (i, j) — 10 columns: `cl_0_0`, `cl_0_1`, `cl_0_2`, `cl_0_3`, `cl_1_1`, `cl_1_2`, `cl_1_3`, `cl_2_2`, `cl_2_3`, `cl_3_3`
- Cosmological parameters: `Omega_m`, `sigma_8`, `S8`, `w`, `h`, `n_s`, `Omega_b`, `m_nu`
