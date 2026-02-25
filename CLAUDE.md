# CLAUDE.md

## Project Overview

**LensingFoM** compares the Figure of Merit (FoM) of different weak lensing summary statistics — SBI (simulation-based inference) vs 2-point functions — as a function of ell_max.

## Current Phase

**Phase 2a complete. Phase 1 and Phase 2a (tile extraction + HuggingFace dataset) are done.** Next steps: pseudo-Cl measurement with NaMaster (Phase 2b) and SBI with neural compression (Phase 2c).

### What's been done

- **Phase 1: Born Raytracing** — 791 Gower Street simulations processed through Born-approximation raytracing with DES Y3 MagLim n(z), producing 4 tomographic convergence maps at nside=1024 per sim. Validated against CAMB Halofit theory.
- **Phase 2a: Tile Extraction** — Full-sky maps filtered in harmonic space at 5 lmax values (200, 400, 600, 800, 1000), rotated for augmentation (3 orientations), and projected onto 4 equatorial HEALPix base tiles per orientation (12 tiles per sim per lmax). Supports 3 noise levels (noiseless, DES Y3, LSST Y10) — shape noise is added at full-sky nside=1024 before harmonic filtering. Published as HuggingFace dataset [`EiffL/GowerStreetDESY3`](https://huggingface.co/datasets/EiffL/GowerStreetDESY3).

### What's next

- **Phase 2b: Pseudo-Cl with NaMaster** — Measure pseudo-Cl power spectra on each tile using NaMaster for mode-coupling correction
- **Phase 2c: SBI with Neural Compression** — Two decoupled stages: (1) train EfficientNet MSE compressor on tiles, (2) train Gaussian NPE on frozen compressor outputs with on-the-fly augmentation. FoM evaluated on held-out fiducial cosmology (sim 109, 12 tiles).
- **Phase 2d: FoM Comparison** — Compare SBI vs 2pt FoM as a function of ell_max

## Repository Structure

- `lensing/` — Core library
  - `io.py` — Load simulation data, DES Y3 n(z), shell info
  - `cosmology.py` — CAMB interface for theory C_ell predictions
  - `raytracing.py` — Born-approximation raytracing with n(z) weighting
  - `validation.py` — Power spectrum measurement and comparison plots
  - `tiles.py` — Tile extraction from HEALPix maps with harmonic filtering, rotation, and shape noise injection
  - `spectra.py` — Flat-sky FFT power spectrum estimation from projected tiles
- `pipeline.py` — Modal pipeline with 5 stages: (1) Born raytracing, (2) tile extraction, (3) build HF parquet, (4) push to HF Hub, (5) build spectra
- `scripts/`
  - `download_data.sh` — Download Gower Street sim00001 + DES Y3 FITS
  - `healpix_to_tiles.py` — Standalone tile extraction demo
  - `lmax_filtering_demo.py` — Harmonic filtering demo
  - `train_field_compressor.py` — Modal: train EfficientNet compressor (stage 1)
  - `train_field_npe.py` — Modal: train Gaussian NPE on frozen compressor (stage 2)
- `notebooks/01_validate_raytracing.ipynb` — End-to-end validation
- `tests/test_tiles.py` — Tests for tile extraction
- `tests/test_tile_dataset.py` — Tests for tile dataset loading and splitting
- `data/` — Simulation and observational data (not committed)
- `gower_street_runs.csv` — Cosmological parameters for all 791 Gower Street sims

## Key Conventions

- HEALPix maps: RING ordering, nside=2048 (input), nside=1024 (raytracing output), nside matched to lmax for tiles
- Distances in shell_info (z_values.txt): Mpc/h. Converted to Mpc internally for CAMB consistency.
- CAMB theory C_ell: raw C_ell (not l(l+1)C_ell/2pi). Use `raw_cl=True` with `get_source_cls_dict`.
- 4 tomographic bins from DES Y3 MagLim source n(z)
- 10 power spectra total: 4 auto + 6 cross
- Tile extraction: 3 rotations x 4 equatorial tiles = 12 tiles per sim per lmax
- Fiducial cosmology: sim 109 (Omega_m=0.3007, sigma_8=0.8047) — held out from all training, used for FoM evaluation
- Data split (field compressor): 70% compressor train / 25% NPE train+compressor val / 5% test (tile-level split, fiducial excluded)
- lmax-to-nside mapping: {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
- Euler rotations in ZYZ convention (healpy default)
- 3 noise levels: "noiseless" (no noise), "des_y3" (DES Y3 shape noise), "lsst_y10" (LSST Y10 shape noise)
- Shape noise added at full-sky nside=1024 level before harmonic filtering; same realization reused across lmax values
- HuggingFace dataset components: lmax_{lmax}_{noise_level} (15 total = 5 lmax x 3 noise)

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
jupyter notebook notebooks/01_validate_raytracing.ipynb
```

### Modal pipeline

```bash
# Run full pipeline (all 5 stages sequentially, idempotent)
modal run pipeline.py --stage all

# Individual stages
modal run pipeline.py --stage 1   # Born raytracing
modal run pipeline.py --stage 2   # Tile extraction (3 noise x 5 lmax)
modal run pipeline.py --stage 3   # Build HF parquet shards
modal run pipeline.py --stage 4   # Push to HuggingFace Hub
modal run pipeline.py --stage 5   # Build power spectra dataset

# Single simulation (stages 1-2 only)
modal run pipeline.py --stage 1 --sim-id 1
modal run pipeline.py --stage 2 --sim-id 1

# Field compressor pipeline (Phase 2c)
modal run scripts/train_field_compressor.py --lmax 200 --noise-level des_y3
modal run scripts/train_field_npe.py --lmax 200 --noise-level des_y3
```

## Data Sources

- Gower Street sim00001: `http://star.ucl.ac.uk/GowerStreetSims/simulations/sim00001.tar.gz`
- DES Y3 2pt FITS: `https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits`
- HuggingFace dataset: `EiffL/GowerStreetDESY3`

## Dependencies

Uses `dorian-astro` as a utility library (cosmology distances, power spectrum measurement). The Born raytracing and n(z) weighting are implemented in-house in `lensing/raytracing.py`. Uses a fork of LensTools (`EiffL/LensTools`) for compatibility with modern Python.
