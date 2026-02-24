"""Flat-sky power spectrum estimation from projected HEALPix tiles."""

import numpy as np

# Angular side length of one HEALPix base face: sqrt(4pi/12) rad ≈ 58.6 deg
TILE_SIDE_RAD = np.sqrt(np.pi / 3)

# 10 unique (i,j) pairs for 4 tomographic bins
SPEC_PAIRS = [(i, j) for i in range(4) for j in range(i, 4)]


def compute_all_spectra(kappa_tile, nside, lmax, n_bins=20):
    """Compute all 10 binned auto/cross power spectra for a 4-bin tile.

    Uses 2D FFT on the flat tile with azimuthal binning in ell-space.
    Normalization: C_ell = (pixel_area / map_area) * |FFT|^2, then
    averaged over modes in each annular bin.

    Parameters
    ----------
    kappa_tile : np.ndarray, shape (4, N, N)
        Convergence maps for 4 tomographic bins.
    nside : int
        Tile side length in pixels (= N).
    lmax : int
        Maximum multipole (sets upper edge of last bin).
    n_bins : int
        Number of linear ell bins.

    Returns
    -------
    ell_eff : np.ndarray, shape (n_bins,)
        Effective ell at bin centers.
    spectra : np.ndarray, shape (10, n_bins)
        Binned C_ell for the 10 pairs, ordered as SPEC_PAIRS.
    """
    N = nside
    pixel_size = TILE_SIDE_RAD / N

    # Ell grid from 2D FFT frequencies
    freq = np.fft.fftfreq(N, d=pixel_size)  # cycles / rad
    ell_x, ell_y = np.meshgrid(2 * np.pi * freq, 2 * np.pi * freq)
    ell_2d = np.sqrt(ell_x**2 + ell_y**2)

    # Linear bins from fundamental mode to lmax
    ell_min = 2 * np.pi / TILE_SIDE_RAD
    ell_edges = np.linspace(ell_min, lmax, n_bins + 1)
    ell_eff = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    # Pre-compute bin masks and mode counts
    bin_masks = []
    for b in range(n_bins):
        bin_masks.append((ell_2d >= ell_edges[b]) & (ell_2d < ell_edges[b + 1]))

    # Pre-compute FFTs for all 4 bins
    ffts = [np.fft.fft2(kappa_tile[i]) for i in range(4)]

    # Normalization: C_ell = pixel_size^2 / N^2 * |FFT|^2
    norm = pixel_size**2 / N**2

    spectra = np.zeros((10, n_bins), dtype=np.float64)
    for s, (i, j) in enumerate(SPEC_PAIRS):
        cross_2d = np.real(ffts[i] * np.conj(ffts[j])) * norm
        for b, mask in enumerate(bin_masks):
            n_modes = mask.sum()
            if n_modes > 0:
                spectra[s, b] = cross_2d[mask].mean()

    return ell_eff, spectra
