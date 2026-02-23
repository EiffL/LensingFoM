"""CAMB interface for theory C_ell predictions."""

import camb
import numpy as np


def setup_camb(cosmo_params, z_nz, nz_bins, lmax=1500):
    """Set up CAMB with simulation cosmology and DES Y3 source windows.

    Parameters
    ----------
    cosmo_params : dict
        Keys: Omega_m, sigma_8, w, Omega_bh2, h, n_s, m_nu
    z_nz : np.ndarray, shape (n_z,)
        Redshift grid for n(z).
    nz_bins : np.ndarray, shape (n_bins, n_z)
        Normalised n(z) for each tomographic bin.
    lmax : int
        Maximum multipole.

    Returns
    -------
    camb.CAMBdata : results object with lensing power spectra.
    """
    h = cosmo_params["h"]
    ombh2 = cosmo_params["Omega_bh2"]
    omch2 = cosmo_params["Omega_m"] * h**2 - ombh2
    # Subtract neutrino contribution from omch2
    # For m_nu = 0.06 eV with 1 massive neutrino:
    # Omega_nu h^2 ~ m_nu / 93.14
    omnuh2 = cosmo_params["m_nu"] / 93.14
    omch2 -= omnuh2

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * h,
        ombh2=ombh2,
        omch2=omch2,
        mnu=cosmo_params["m_nu"],
        num_massive_neutrinos=1,
    )
    pars.InitPower.set_params(
        As=2e-9,  # placeholder, will calibrate to sigma_8
        ns=cosmo_params["n_s"],
    )
    pars.set_dark_energy(w=cosmo_params["w"])

    # First pass: get sigma_8 and rescale As
    pars.set_matter_power(redshifts=[0], kmax=10.0)
    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)
    sigma8_raw = results.get_sigma8_0()
    target_sigma8 = cosmo_params["sigma_8"]
    As_new = pars.InitPower.As * (target_sigma8 / sigma8_raw) ** 2

    # Second pass with correct As and source windows
    pars.InitPower.set_params(As=As_new, ns=cosmo_params["n_s"])

    # Set up lensing source windows for each tomographic bin
    source_windows = []
    for i in range(nz_bins.shape[0]):
        source_windows.append(
            camb.sources.SplinedSourceWindow(
                z=z_nz,
                W=nz_bins[i],
                source_type="lensing",
            )
        )

    pars.SourceWindows = source_windows
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.NonLinear = camb.model.NonLinear_both

    results = camb.get_results(pars)
    return results


def get_theory_cls(results, lmax=1500):
    """Extract theory C_ell^{kappa_a, kappa_b} from CAMB results.

    Parameters
    ----------
    results : camb.CAMBdata
    lmax : int

    Returns
    -------
    dict : keys are (i, j) tuples (0-indexed bin pairs, i <= j),
           values are C_ell arrays of length lmax+1.
    """
    # get_source_cls_dict with raw_cl=True returns C_ell (not l(l+1)C_ell/(2pi))
    # Keys are like 'W1xW1', 'W1xW2', etc. (1-indexed)
    cls_dict = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

    theory_cls = {}
    n_bins = len(results.Params.SourceWindows)
    for i in range(n_bins):
        for j in range(i, n_bins):
            key = f"W{i+1}xW{j+1}"
            theory_cls[(i, j)] = cls_dict[key]

    return theory_cls


def get_camb_background(cosmo_params):
    """Get CAMB background functions (chi(z), H(z)) for distance calculations.

    Returns
    -------
    results : camb.CAMBdata
        Use results.comoving_radial_distance(z) for chi in Mpc,
        results.hubble_parameter(z) for H(z) in km/s/Mpc.
    """
    h = cosmo_params["h"]
    ombh2 = cosmo_params["Omega_bh2"]
    omch2 = cosmo_params["Omega_m"] * h**2 - ombh2
    omnuh2 = cosmo_params["m_nu"] / 93.14
    omch2 -= omnuh2

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * h,
        ombh2=ombh2,
        omch2=omch2,
        mnu=cosmo_params["m_nu"],
        num_massive_neutrinos=1,
    )
    pars.set_dark_energy(w=cosmo_params["w"])
    pars.set_matter_power(redshifts=[0], kmax=10.0)
    results = camb.get_results(pars)
    return results
