"""Tests for tile extraction pipeline."""
import healpy as hp
import numpy as np
import pytest

from lensing.tiles import (
    EQUATORIAL_TILES,
    LMAX_TO_NSIDE,
    ORIENTATIONS,
    extract_tiles_for_lmax,
    filter_and_rotate,
    healpix_to_tile,
)


def test_lmax_to_nside_mapping():
    """Each lmax maps to a specific downsampled nside."""
    assert LMAX_TO_NSIDE[200] == 128
    assert LMAX_TO_NSIDE[400] == 256
    assert LMAX_TO_NSIDE[600] == 256
    assert LMAX_TO_NSIDE[800] == 512
    assert LMAX_TO_NSIDE[1000] == 512


def test_orientations():
    """Three orientations defined with ZYZ Euler angles in degrees."""
    assert len(ORIENTATIONS) == 3
    assert ORIENTATIONS[0] == (0, 0, 0)  # identity


def test_filter_and_rotate_identity():
    """Identity rotation should just filter (up to SHT round-trip noise)."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    result = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    nside_out = LMAX_TO_NSIDE.get(lmax_cut, nside)

    # For lmax not in LMAX_TO_NSIDE, output nside = input nside
    assert hp.npix2nside(len(result)) == nside_out or hp.npix2nside(len(result)) == nside


def test_filter_and_rotate_changes_map():
    """Non-identity rotation should produce a different map."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    m_id = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    m_rot = filter_and_rotate(m, lmax_cut, euler_angles=(0, 90, 0))

    assert not np.allclose(m_id, m_rot)


def test_filter_and_rotate_preserves_power():
    """Rotation should preserve the total power (C_ell is rotationally invariant)."""
    nside = 32
    lmax_cut = 20
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    m_id = filter_and_rotate(m, lmax_cut, euler_angles=(0, 0, 0))
    m_rot = filter_and_rotate(m, lmax_cut, euler_angles=(0, 90, 0))

    cl_id = hp.anafast(m_id, lmax=lmax_cut)
    cl_rot = hp.anafast(m_rot, lmax=lmax_cut)

    np.testing.assert_allclose(cl_id, cl_rot, rtol=0.05)


def test_extract_tiles_for_lmax_shape():
    """extract_tiles_for_lmax returns (12, nside_down, nside_down) array."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    tiles = extract_tiles_for_lmax(m, lmax)

    nside_down = LMAX_TO_NSIDE[lmax]
    assert tiles.shape == (12, nside_down, nside_down)


def test_extract_tiles_for_lmax_orientations():
    """Tiles from different orientations should differ."""
    nside = 64
    lmax = 200
    np.random.seed(42)
    m = np.random.randn(hp.nside2npix(nside))

    tiles = extract_tiles_for_lmax(m, lmax)

    # Tile 0 (orientation 0, equatorial tile 4) vs tile 4 (orientation 1, equatorial tile 4)
    assert not np.allclose(tiles[0], tiles[4])
