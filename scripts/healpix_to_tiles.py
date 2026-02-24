"""Extract 12 square 2D images from a HEALPix map using base tile (face) decomposition.

In NESTED ordering, each of the 12 HEALPix base tiles maps to an nside x nside
square grid via the Z-order (Morton) curve. hp.pix2xyf gives the (x, y, face)
coordinates directly.

Usage:
    python scripts/healpix_to_tiles.py
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def healpix_to_tile(hpx_map, tile_id):
    """Extract a single square 2D image from a HEALPix map for one base tile.

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


if __name__ == "__main__":
    # Load the reference kappa maps — only bin 1
    kappa_all = np.load("data/kappa_maps.npy", mmap_mode="r")
    kappa = np.array(kappa_all[0])
    del kappa_all
    nside = hp.npix2nside(len(kappa))
    print(f"Loaded kappa map: nside={nside}, npix={len(kappa)}")

    tile = healpix_to_tile(kappa, tile_id=0)
    print(f"Tile 0: shape={tile.shape}, min={tile.min():.4f}, max={tile.max():.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(tile, origin="lower", cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    ax.set_title(f"Tile 0 (nside={nside}, bin 1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\kappa$")
    fig.savefig("data/tile0_preview.png", dpi=150, bbox_inches="tight")
    print("Saved to data/tile0_preview.png")
