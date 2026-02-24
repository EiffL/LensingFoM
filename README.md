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

This processes all simulations in parallel (up to 100 concurrent containers). Each simulation takes ~5-10 minutes, so the full run completes in ~1-2 hours. The run is idempotent — simulations that already have output on the volume are skipped, so you can safely re-run after failures.

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
