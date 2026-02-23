# LensingFoM

Comparison of Figure of Merit (FoM) for weak lensing cosmological constraints: simulation-based inference (SBI) vs 2-point function analysis, as a function of maximum multipole ell_max.

## Phase 1: Born Raytracing Validation

The first phase processes [Gower Street](http://star.ucl.ac.uk/GowerStreetSims/) N-body simulations through a Born-approximation raytracing pipeline with DES Y3 source galaxy redshift distributions, and validates the output convergence power spectra against CAMB Halofit theory predictions.

### What it does

1. Loads particle lightcone shells from an N-body simulation (HEALPix nside=2048)
2. Converts particle counts to overdensity fields
3. Applies Born lensing weights using DES Y3 n(z) for 4 tomographic bins
4. Produces full-sky convergence maps at nside=512
5. Measures auto- and cross-power spectra (10 total: 4 auto + 6 cross)
6. Compares against CAMB theory predictions

### Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh    # Downloads ~3 GB of simulation data
jupyter notebook notebooks/01_validate_raytracing.ipynb
```

### Data

- **Gower Street sim00001**: ~100 lightcone shells from an N-body simulation with cosmology Omega_m=0.29, sigma_8=0.77, w=-1.01
- **DES Y3 MagLim n(z)**: 4 tomographic source bins from the Dark Energy Survey Year 3 analysis

### Validation Results

The Born raytracing C_ell agrees with CAMB Halofit theory to within:
- Cosmic variance at low ell (< 100)
- ~1% at intermediate ell (~200–500)
- ~20% suppression at high ell (> 1000) due to resolution and non-linear effects beyond Halofit
