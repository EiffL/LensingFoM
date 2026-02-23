# CLAUDE.md

## Project Overview

**LensingFoM** compares the Figure of Merit (FoM) of different weak lensing summary statistics — SBI (simulation-based inference) vs 2-point functions — as a function of ell_max.

## Current Phase

**Phase 1: Born Raytracing Validation.** We process Gower Street N-body simulations through a Born-approximation raytracing pipeline with DES Y3 n(z) source distributions, and validate the output convergence power spectra against CAMB theory predictions.

## Repository Structure

- `lensing/` — Core library
  - `io.py` — Load simulation data, DES Y3 n(z), shell info
  - `cosmology.py` — CAMB interface for theory C_ell predictions
  - `raytracing.py` — Born-approximation raytracing with n(z) weighting
  - `validation.py` — Power spectrum measurement and comparison plots
- `scripts/download_data.sh` — Download Gower Street sim00001 + DES Y3 FITS
- `notebooks/01_validate_raytracing.ipynb` — End-to-end validation
- `data/` — Simulation and observational data (not committed)
- `gower_street_runs.csv` — Cosmological parameters for all Gower Street sims

## Key Conventions

- HEALPix maps: RING ordering, nside=2048 (input), nside=512 (analysis)
- Distances in shell_info (z_values.txt): Mpc/h. Converted to Mpc internally for CAMB consistency.
- CAMB theory C_ell: raw C_ell (not l(l+1)C_ell/2pi). Use `raw_cl=True` with `get_source_cls_dict`.
- 4 tomographic bins from DES Y3 MagLim source n(z)
- 10 power spectra total: 4 auto + 6 cross

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
jupyter notebook notebooks/01_validate_raytracing.ipynb
```

## Data Sources

- Gower Street sim00001: `http://star.ucl.ac.uk/GowerStreetSims/simulations/sim00001.tar.gz`
- DES Y3 2pt FITS: `https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits`

## Dependencies

Uses `dorian-astro` as a utility library (cosmology distances, power spectrum measurement). The Born raytracing and n(z) weighting are implemented in-house in `lensing/raytracing.py`.
