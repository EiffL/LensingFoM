"""Illustrate the effect of lmax filtering on a single tile.

Produces a 1x5 panel showing tile 0, bin 4 (highest redshift),
at lmax = 100, 250, 500, 750, 1000.

Usage:
    python scripts/lmax_filtering_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

from lensing.tiles import filter_map_lmax, healpix_to_tile

LMAX_VALUES = [100, 250, 500, 750, 1000]
TILE_ID = 0
BIN = 3  # 0-indexed, bin 4 = highest redshift

kappa_all = np.load("data/kappa_maps.npy", mmap_mode="r")
kappa = np.array(kappa_all[BIN])
del kappa_all
print(f"Loaded bin {BIN+1} kappa map, std={kappa.std():.4f}")

fig, axes = plt.subplots(1, len(LMAX_VALUES), figsize=(4 * len(LMAX_VALUES), 4))

for ax, lmax in zip(axes, LMAX_VALUES):
    filtered = filter_map_lmax(kappa, lmax)
    tile = healpix_to_tile(filtered, TILE_ID)
    im = ax.imshow(tile, origin="lower", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    ax.set_title(rf"$\ell_{{\max}}={lmax}$", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\kappa$", pad=0.02)
fig.suptitle(f"Tile {TILE_ID}, Bin 4 — harmonic filtering", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig("data/lmax_filtering_demo.png", dpi=150, bbox_inches="tight")
print("Saved to data/lmax_filtering_demo.png")
