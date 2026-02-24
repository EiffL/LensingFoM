# FoM Pipeline Design: SBI vs 2pt as a function of lmax

## Goal

Produce a plot of Figure of Merit (FoM) for (Omega_m, S8) as a function of lmax, comparing two inference approaches:
1. **2pt function** — binned power spectra fed to SBI
2. **Neural compression** — EfficientNet compressor trained under VMIM, then posterior estimation with a normalizing flow

## Data: Gower Street simulations

- 791 N-body simulations, each with different cosmological parameters
- Phase 1 (complete): Born raytracing produces 4 full-sky convergence maps (nside=1024, 4 DES Y3 MagLim tomo bins) per sim
- Modal pipeline saves `kappa_maps.npz` per sim to a Modal volume

## Tile extraction strategy

### Base tiles
HEALPix has 12 base tiles. Equatorial tiles (4-7) have minimal projection distortion when flattened to a 2D grid. Polar tiles (0-3, 8-11) are significantly distorted near the poles.

### Fixed rotations for data augmentation
Three fixed orientations of the sphere, each yielding 4 equatorial tiles:
- **Orientation 0 (identity):** Euler angles (0, 0, 0)
- **Orientation 1 (90 deg about y):** Euler angles (0, pi/2, 0)
- **Orientation 2 (90 deg about x):** Euler angles (pi/2, 0, 0)

Total: 3 orientations x 4 equatorial tiles = 12 tiles per sim.
Dataset size: 791 sims x 12 tiles = 9,492 tiles.

### Harmonic filtering and consistency
All maps (rotated or not) go through the same harmonic-space processing:
```
alm = map2alm(kappa, lmax=lmax_cut)
alm_rot = rotate_alm(alm, euler_angles)  # identity for orientation 0
filtered_map = alm2map(alm_rot, nside=nside_down)
tile = extract_equatorial_tile(filtered_map, tile_id)
```
This ensures all tiles see identical harmonic-space artifacts (truncation, rotation interpolation), keeping the dataset consistent for SBI.

### lmax values and resolution
| lmax | nside_down | tile size |
|------|-----------|-----------|
| 200  | 128       | 128x128   |
| 400  | 256       | 256x256   |
| 600  | 256       | 256x256   |
| 800  | 512       | 512x512   |
| 1000 | 512       | 512x512   |

### Tile angular size
Each base tile covers 4*pi/12 sr ~ 3438 deg^2. Side length ~ 58.6 deg. Used as the `angle` parameter for lenstools power spectrum computation.

### Storage
Per sim: ~5 MB across all lmax values. Total: ~4 GB.

## Power spectra (2pt summary)

Computed using lenstools `ConvergenceMap`:
- `.powerSpectrum(l_edges)` for 4 auto-spectra
- `.cross(other_map, l_edges)` for 6 cross-spectra
- 10 spectra total per tile, binned into ~30-50 bandpowers each
- Summary vector: ~300-500 dimensions

## Neural compression (full-field summary)

Architecture adapted from s8ball (NeurIPS weak lensing challenge):
- EfficientNet backbone taking (4, H, W) input — all 4 tomo bins as channels
- No patch decomposition — process full tile directly
- Two-stage training:
  1. Train compressor + flow head under VMIM to learn ~10-20 dimensional summary
  2. Freeze compressor, train separate small normalizing flow for posterior on (Omega_m, S8)

## FoM computation

For each lmax and each branch:
- Run SBI (NPE with normalizing flow) on the summary statistics
- Get posterior samples for held-out test sims
- Compute 2x2 covariance of (Omega_m, S8)
- FoM = 1 / sqrt(det(cov))

Final deliverable: FoM vs lmax plot with two curves.

## Implementation phases

### Phase 2a — Modal tile extraction
- Extend `lensing/tiles.py` with rotation and downsampling logic
- New Modal function reads existing `kappa_maps.npz`, produces tile files
- Runs as a second pass over the volume (no re-download needed)

### Phase 2b — HuggingFace dataset
- Build a proper HuggingFace dataset from the extracted tiles
- Sharded by lmax: `load_dataset("username/lensing-tiles", name="lmax_600")`
- Each row: tile (4, H, W) float32 + cosmological parameters (Omega_m, sigma_8, S8, w, h, n_s, Omega_b, m_nu) + metadata (sim_id, lmax, orientation_id, tile_id)
- Standalone product useful beyond this project

### Phase 2c — Power spectrum computation
- Use lenstools to compute all 10 auto+cross spectra per tile
- Store alongside dataset or compute on-the-fly

### Phase 2d — SBI training (2pt branch)
- NPE with normalizing flow on binned power spectrum vectors
- Train/val split by sim_id (not by tile, to avoid data leakage)
- Extract FoM at each lmax

### Phase 2e — SBI training (neural branch)
- EfficientNet (4, H, W) -> compressor -> VMIM
- Freeze compressor -> posterior flow
- Extract FoM at each lmax

### Phase 2f — FoM comparison plot
- FoM vs lmax, two curves (2pt vs neural)
- Error bars from variation across test set
