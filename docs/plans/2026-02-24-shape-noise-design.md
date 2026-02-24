# Shape Noise Injection for Tile Extraction Pipeline

## Problem

The current tile extraction pipeline produces noiseless convergence tiles. To realistically compare SBI vs 2pt FoM as a function of ell_max, we need tiles that include survey-realistic shape noise. Different noise levels (DES Y3 vs LSST Y10) allow studying how constraining power scales with survey depth.

## Noise Model

Shape noise in convergence maps arises from the intrinsic ellipticity dispersion of source galaxies. For a HEALPix pixel at resolution nside, the noise variance per pixel per tomographic bin is:

```
sigma^2_pix,i = sigma_e,i^2 / (2 * n_gal,i * A_pix)
```

where:
- `sigma_e,i` is the per-component intrinsic ellipticity dispersion for bin i
- `n_gal,i` is the effective galaxy number density for bin i (in sr^-1)
- `A_pix = 4*pi / N_pix` is the pixel solid angle in steradians
- The factor of 2 accounts for two ellipticity components

The noise is Gaussian and independent per pixel.

## Survey Specifications

### DES Y3 (Amon et al. 2022, Table 1, arXiv:2105.13543)

| Bin | n_eff (arcmin^-2) | sigma_e |
|-----|-------------------|---------|
| 0   | 1.476             | 0.243   |
| 1   | 1.479             | 0.262   |
| 2   | 1.484             | 0.259   |
| 3   | 1.461             | 0.301   |

### LSST Y10 (DESC SRD, arXiv:1809.01669)

| Bin | n_eff (arcmin^-2) | sigma_e |
|-----|-------------------|---------|
| 0   | 6.75              | 0.26    |
| 1   | 6.75              | 0.26    |
| 2   | 6.75              | 0.26    |
| 3   | 6.75              | 0.26    |

Total n_eff = 27 arcmin^-2 split uniformly across 4 bins to match DES tomographic structure.

## Design

### Where noise is injected

Noise is added to the full-sky nside=1024 convergence map **before** harmonic filtering and rotation. This ensures:
1. The harmonic filter band-limits the noise consistently with the signal
2. All tiles from the same orientation share the same noise realization
3. The noise power spectrum is white before filtering (as expected for shape noise)

### Noise levels

Three noise levels, stored as string identifiers:
- `"noiseless"` — no noise added (reproduces current behavior)
- `"des_y3"` — DES Y3 per-bin specs from table above
- `"lsst_y10"` — LSST Y10 per-bin specs from table above

### Code changes

**`lensing/tiles.py`:**
- Add survey noise specs as module-level dicts: `NOISE_CONFIGS`
- Add `add_shape_noise(hpx_map, n_eff_arcmin2, sigma_e, rng)` function
- Modify `extract_tiles_for_lmax` to accept `noise_level` and `rng` parameters
- When noise_level is not "noiseless", add noise to the map before calling `filter_and_rotate`
- One noise realization per (sim, bin, noise_level) — same noise map used across all lmax cuts and orientations for a given bin

**`pipeline.py`:**
- `extract_tiles` loops over `NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]`
- Output: `tiles_lmax{lmax}_noise_{level}.npz` per (lmax, noise_level)
- Deterministic RNG seeding: `seed = sim_id * 1000 + noise_level_index`

**`scripts/build_hf_dataset.py`:**
- Loop over `(lmax, noise_level)` pairs
- HuggingFace components: `lmax_200_noiseless`, `lmax_200_des_y3`, `lmax_200_lsst_y10`, ...
- Data path: `data/lmax_{lmax}_{noise_level}/shard_*.parquet`
- Add `noise_level` column to each record

**`scripts/push_hf_dataset.py`:**
- Update to push all `(lmax, noise_level)` components

### RNG strategy

- For a given (sim_id, noise_level), create one RNG: `np.random.default_rng(sim_id * 1000 + noise_idx)`
- Generate one noise map per tomographic bin from this RNG
- The same noisy maps are used across all lmax values — this is correct because the lmax filter selects different scales from the same underlying noisy field
- Different orientations also share the noise because rotation is applied after noise addition

### Output structure

```
/results/sim00001/
  kappa_maps.npz                          # Stage 1 (unchanged)
  tiles_lmax200_noise_noiseless.npz       # (12, 4, 128, 128)
  tiles_lmax200_noise_des_y3.npz
  tiles_lmax200_noise_lsst_y10.npz
  tiles_lmax400_noise_noiseless.npz
  ...
```

HuggingFace dataset:
```
data/
  lmax_200_noiseless/shard_0000.parquet
  lmax_200_des_y3/shard_0000.parquet
  lmax_200_lsst_y10/shard_0000.parquet
  lmax_400_noiseless/shard_0000.parquet
  ...
```
