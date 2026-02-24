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
