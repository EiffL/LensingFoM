"""Born-approximation raytracing with n(z) weighting."""

import healpy as hp
import numpy as np
from scipy.interpolate import interp1d

from .cosmology import get_camb_background
from .io import get_valid_shell_ids, load_shell_map


def compute_lensing_weights(shell_info, z_nz, nz_bins, cosmo_params):
    """Compute Born lensing weights W[n_bins, n_shells].

    W_k^a = (3/2) * (H_0/c)^2 * Omega_m * (1+z_k) * chi_k * g_a(chi_k) * Delta_chi_k

    where g_a(chi) = integral_{chi}^{inf} n_a(chi') * (chi' - chi) / chi' dchi'

    All distances in Mpc (CAMB convention, not Mpc/h).

    Parameters
    ----------
    shell_info : structured array
        From load_shell_info. Distances are in Mpc/h.
    z_nz : np.ndarray, shape (n_z,)
        Redshift grid of n(z).
    nz_bins : np.ndarray, shape (n_bins, n_z)
        Normalised n(z) for each tomographic bin.
    cosmo_params : dict

    Returns
    -------
    weights : np.ndarray, shape (n_bins, n_shells)
    """
    h = cosmo_params["h"]
    Omega_m = cosmo_params["Omega_m"]
    H0 = 100 * h  # km/s/Mpc
    c = 299792.458  # km/s

    # Get CAMB background for chi(z)
    bg = get_camb_background(cosmo_params)

    # Convert shell distances from Mpc/h to Mpc
    chi_mid = shell_info["chi_mid"] / h  # Mpc
    delta_chi = shell_info["delta_chi"] / h  # Mpc
    z_mid = shell_info["z_mid"]

    n_bins = nz_bins.shape[0]
    n_shells = len(shell_info)

    # Build chi(z) mapping from CAMB for the n(z) redshift grid
    chi_nz = bg.comoving_radial_distance(z_nz)  # Mpc

    # Compute lensing efficiency g_a(chi_k) for each shell and bin
    weights = np.zeros((n_bins, n_shells))
    prefactor = 1.5 * (H0 / c) ** 2 * Omega_m

    for a in range(n_bins):
        # n_a(z) as a function of chi: n_a(chi) = n_a(z) * dz/dchi
        # But in the integral g_a(chi) = int n_a(z(chi')) * (chi' - chi)/chi' * dz/dchi' dchi'
        # which simplifies to g_a(chi) = int n_a(z') * (chi(z') - chi)/chi(z') dz'
        # since n_a(z) dz = n_a(chi) dchi
        nz_a = nz_bins[a]

        for k in range(n_shells):
            chi_k = chi_mid[k]
            z_k = z_mid[k]

            # Lensing efficiency: integrate over source redshifts beyond chi_k
            mask = chi_nz > chi_k
            if not np.any(mask):
                continue

            integrand = nz_a * (chi_nz - chi_k) / np.where(chi_nz > 0, chi_nz, 1.0)
            integrand[~mask] = 0.0
            g_a_k = np.trapezoid(integrand, z_nz)

            weights[a, k] = prefactor * (1 + z_k) * chi_k * g_a_k * delta_chi[k]

    return weights


def born_raytrace(sim_dir, shell_info, weights, nside_out=512):
    """Perform Born raytracing to produce convergence maps.

    Parameters
    ----------
    sim_dir : str or Path
        Path to simulation directory.
    shell_info : structured array
        From load_shell_info.
    weights : np.ndarray, shape (n_bins, n_shells)
        Lensing weights from compute_lensing_weights.
    nside_out : int
        Output HEALPix resolution (default: 512).

    Returns
    -------
    kappa_maps : list of np.ndarray
        4 convergence maps at nside_out resolution.
    """
    n_bins = weights.shape[0]
    npix_out = hp.nside2npix(nside_out)
    kappa_maps = [np.zeros(npix_out) for _ in range(n_bins)]

    valid_ids = get_valid_shell_ids(sim_dir)

    # Build a lookup from shell_id to index in shell_info
    id_to_idx = {int(sid): idx for idx, sid in enumerate(shell_info["shell_id"])}

    for shell_id in valid_ids:
        if shell_id not in id_to_idx:
            continue
        idx = id_to_idx[shell_id]

        # Check if any bin has non-zero weight for this shell
        shell_weights = weights[:, idx]
        if np.all(shell_weights == 0):
            continue

        # Load particle count map and convert to overdensity
        particle_map = load_shell_map(sim_dir, shell_id).astype(np.float64)
        mean_count = np.mean(particle_map)
        if mean_count == 0:
            continue
        delta = particle_map / mean_count - 1.0

        # Downgrade resolution if needed
        nside_in = hp.npix2nside(len(delta))
        if nside_in != nside_out:
            delta = hp.ud_grade(delta, nside_out)

        # Accumulate weighted overdensity into convergence maps
        for a in range(n_bins):
            if shell_weights[a] != 0:
                kappa_maps[a] += shell_weights[a] * delta

        print(f"  Shell {shell_id:5d} (z={shell_info['z_mid'][idx]:.3f}): done")

    return kappa_maps
