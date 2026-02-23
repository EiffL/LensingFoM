"""Power spectrum measurement and validation plots."""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def measure_all_cls(kappa_maps, lmax=None):
    """Measure auto- and cross-power spectra from convergence maps.

    Parameters
    ----------
    kappa_maps : list of np.ndarray
        4 HEALPix convergence maps.
    lmax : int or None
        Maximum multipole. Default: 3*nside - 1.

    Returns
    -------
    dict : keys are (i, j) tuples (0-indexed, i <= j),
           values are C_ell arrays.
    """
    n_bins = len(kappa_maps)
    if lmax is None:
        nside = hp.npix2nside(len(kappa_maps[0]))
        lmax = 3 * nside - 1

    # Compute all alm once
    alms = [hp.map2alm(m, lmax=lmax) for m in kappa_maps]

    measured_cls = {}
    for i in range(n_bins):
        for j in range(i, n_bins):
            cl = hp.alm2cl(alms[i], alms[j], lmax=lmax)
            measured_cls[(i, j)] = cl

    return measured_cls


def plot_validation(measured_cls, theory_cls, lmax=None, save_path=None):
    """Plot measured vs theory C_ell for all 10 tomographic spectra.

    Parameters
    ----------
    measured_cls : dict
        Measured C_ell from measure_all_cls.
    theory_cls : dict
        Theory C_ell from get_theory_cls.
    lmax : int or None
        Maximum ell for plotting.
    save_path : str or None
        If given, save figure to this path.
    """
    pairs = sorted(measured_cls.keys())
    n_pairs = len(pairs)

    # Determine layout
    if n_pairs <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        cl_meas = measured_cls[(i, j)]
        cl_theo = theory_cls[(i, j)]

        ell_meas = np.arange(len(cl_meas))
        ell_theo = np.arange(len(cl_theo))

        plot_lmax = lmax or min(len(cl_meas), len(cl_theo)) - 1
        sl_m = slice(2, plot_lmax + 1)
        sl_t = slice(2, plot_lmax + 1)

        ax.loglog(
            ell_meas[sl_m],
            ell_meas[sl_m] * (ell_meas[sl_m] + 1) * cl_meas[sl_m] / (2 * np.pi),
            alpha=0.7,
            label="Measured",
        )
        ax.loglog(
            ell_theo[sl_t],
            ell_theo[sl_t] * (ell_theo[sl_t] + 1) * cl_theo[sl_t] / (2 * np.pi),
            "k--",
            alpha=0.8,
            label="CAMB theory",
        )
        label = f"({i+1},{j+1})" if i != j else f"({i+1},{i+1})"
        ax.set_title(f"Bin {label}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$")
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Convergence Power Spectra: Measured vs Theory", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ratio(measured_cls, theory_cls, lmax=None, save_path=None):
    """Plot ratio of measured/theory C_ell with cosmic variance band.

    Parameters
    ----------
    measured_cls : dict
    theory_cls : dict
    lmax : int or None
    save_path : str or None
    """
    pairs = sorted(measured_cls.keys())
    n_pairs = len(pairs)

    if n_pairs <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        cl_meas = measured_cls[(i, j)]
        cl_theo = theory_cls[(i, j)]

        min_len = min(len(cl_meas), len(cl_theo))
        plot_lmax = lmax or min_len - 1
        plot_lmax = min(plot_lmax, min_len - 1)
        ell = np.arange(2, plot_lmax + 1)

        ratio = cl_meas[2 : plot_lmax + 1] / np.where(
            cl_theo[2 : plot_lmax + 1] > 0, cl_theo[2 : plot_lmax + 1], np.inf
        )

        # Cosmic variance: sqrt(2/(2l+1)) for auto-spectra
        if i == j:
            cv = np.sqrt(2.0 / (2 * ell + 1))
            ax.fill_between(ell, 1 - cv, 1 + cv, alpha=0.15, color="gray",
                            label="Cosmic variance")

        ax.semilogx(ell, ratio, alpha=0.6, lw=0.8)
        ax.axhline(1.0, color="k", ls="--", lw=0.5)
        ax.set_ylim(0.5, 1.5)
        label = f"({i+1},{j+1})" if i != j else f"({i+1},{i+1})"
        ax.set_title(f"Bin {label}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell^{\rm meas} / C_\ell^{\rm theory}$")
        if idx == 0 and i == j:
            ax.legend(fontsize=8)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Ratio: Measured / Theory", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence_maps(kappa_maps, save_path=None):
    """Plot Mollweide projections of the 4 convergence maps.

    Parameters
    ----------
    kappa_maps : list of np.ndarray
    save_path : str or None
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                              subplot_kw={"projection": "mollweide"})

    for i, (ax, kmap) in enumerate(zip(axes.flatten(), kappa_maps)):
        # Use healpy's mollview but into our axes
        hp.mollview(kmap, title=f"Bin {i+1}", sub=(2, 2, i + 1),
                     hold=True, fig=fig.number, min=-0.03, max=0.03)

    fig.suptitle("Convergence Maps (Born approximation)", y=0.98)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
