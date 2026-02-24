#!/usr/bin/env python
"""Plot theoretical Fisher FoM vs ell_max for DES Y3 noise.

Uses the same ell binning as the measured Cl data (20 linear bins from
2*pi/TILE_SIDE_RAD to lmax) and the same 10 auto/cross power spectra
for 4 DES Y3 MagLim tomographic bins.

The Fisher matrix includes:
- Signal: C_ell^{ij}(Omega_m, S8) from CAMB
- Noise: N_ell^{ij} = delta_{ij} * sigma_e_i^2 / (2 * n_eff_i)
  with DES Y3 MagLim shape noise parameters
- Derivatives via finite differences
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lensing.spectra import TILE_SIDE_RAD, SPEC_PAIRS
from lensing.tiles import NOISE_CONFIGS
from lensing.cosmology import setup_camb, get_theory_cls
from lensing.io import load_des_y3_nz

# ── Constants ──────────────────────────────────────────────────────────
LMAX_VALUES = [200, 400, 600, 800, 1000]
N_ELL = 20
F_SKY = 1.0 / 12  # one HEALPix base tile
ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2

# Fiducial cosmology (full parameter set needed by CAMB)
FIDUCIAL = {
    "Omega_m": 0.3,
    "sigma_8": 0.8 / np.sqrt(0.3 / 0.3),  # sigma8 when S8=0.8 at Om=0.3
    "w": -1.0,
    "Omega_bh2": 0.0224,
    "h": 0.67,
    "n_s": 0.96,
    "m_nu": 0.06,
}

# S8 = sigma8 * sqrt(Omega_m / 0.3), so at fiducial: S8 = 0.8
OMEGA_M_FID = 0.3
S8_FID = 0.8

# Noise configs
DES_Y3 = NOISE_CONFIGS["des_y3"]
LSST_Y10 = NOISE_CONFIGS["lsst_y10"]
N_BINS = 4

# Pair ordering matching SPEC_PAIRS
PAIR_TO_IJ = list(SPEC_PAIRS)  # [(0,0), (0,1), ..., (3,3)]

NZ_PATH = "data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"

# Step sizes for finite differences
D_OMEGA_M = 0.005
D_S8 = 0.005


# ── Noise power spectrum ──────────────────────────────────────────────
def noise_cls(noise_config):
    """Noise power spectrum N_ell^{ij} for each of the 10 pairs (ell-independent)."""
    n_eff_sr = [n * ARCMIN2_PER_SR for n in noise_config["n_eff_arcmin2"]]
    sigma_e = noise_config["sigma_e"]
    n_ell_vec = np.zeros(10)
    for p, (i, j) in enumerate(PAIR_TO_IJ):
        if i == j:
            n_ell_vec[p] = sigma_e[i] ** 2 / (2.0 * n_eff_sr[i])
    return n_ell_vec


# ── Ell binning (matching lensing.spectra) ────────────────────────────
def get_ell_bins(lmax, n_bins=N_ELL):
    ell_min = 2 * np.pi / TILE_SIDE_RAD
    ell_edges = np.linspace(ell_min, lmax, n_bins + 1)
    ell_eff = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    return ell_edges, ell_eff


# ── CAMB C_ell at given (Omega_m, S8) ────────────────────────────────
def _get_cls_at(omega_m, s8, z_mid, nz_bins, lmax_camb=1100):
    """Compute theory C_ell dict at (Omega_m, S8), keeping other params fixed."""
    sigma8 = s8 / np.sqrt(omega_m / 0.3)
    params = FIDUCIAL.copy()
    params["Omega_m"] = omega_m
    params["sigma_8"] = sigma8
    results = setup_camb(params, z_mid, nz_bins, lmax=lmax_camb)
    return get_theory_cls(results, lmax=lmax_camb)


def _bin_cls(cl_dict, ell_eff):
    """Bin full-ell C_ell dict into (10, n_ell) at the given ell_eff values.

    Uses linear interpolation from integer ell to ell_eff.
    """
    n_ell = len(ell_eff)
    cls_binned = np.zeros((10, n_ell))
    for p, (i, j) in enumerate(PAIR_TO_IJ):
        cl_full = cl_dict[(i, j)]  # length lmax+1, indexed by integer ell
        # Interpolate to ell_eff
        ell_int = np.arange(len(cl_full))
        cls_binned[p] = np.interp(ell_eff, ell_int, cl_full)
    return cls_binned


def _to_matrix(cl_vec, n_ell):
    """Convert (10, n_ell) vectorized spectra to (n_ell, 4, 4) symmetric matrix."""
    mat = np.zeros((n_ell, N_BINS, N_BINS))
    for p, (i, j) in enumerate(PAIR_TO_IJ):
        mat[:, i, j] = cl_vec[p]
        if i != j:
            mat[:, j, i] = cl_vec[p]
    return mat


# ── CAMB precomputation ───────────────────────────────────────────────
def precompute_camb_cls(z_mid, nz_bins):
    """Compute CAMB C_ell at fiducial and perturbed points (reusable across noise configs)."""
    print("  Computing CAMB C_ell at fiducial...")
    cl_fid = _get_cls_at(OMEGA_M_FID, S8_FID, z_mid, nz_bins)

    print("  Computing CAMB C_ell at Omega_m + delta...")
    cl_om_plus = _get_cls_at(OMEGA_M_FID + D_OMEGA_M, S8_FID, z_mid, nz_bins)
    print("  Computing CAMB C_ell at Omega_m - delta...")
    cl_om_minus = _get_cls_at(OMEGA_M_FID - D_OMEGA_M, S8_FID, z_mid, nz_bins)

    print("  Computing CAMB C_ell at S8 + delta...")
    cl_s8_plus = _get_cls_at(OMEGA_M_FID, S8_FID + D_S8, z_mid, nz_bins)
    print("  Computing CAMB C_ell at S8 - delta...")
    cl_s8_minus = _get_cls_at(OMEGA_M_FID, S8_FID - D_S8, z_mid, nz_bins)

    return cl_fid, cl_om_plus, cl_om_minus, cl_s8_plus, cl_s8_minus


# ── Fisher FoM computation ────────────────────────────────────────────
def compute_fisher_fom(camb_cls, lmax_values, noise_config=None):
    """Compute Fisher FoM(Omega_m, S8) vs lmax using precomputed CAMB C_ell."""
    cl_fid, cl_om_plus, cl_om_minus, cl_s8_plus, cl_s8_minus = camb_cls

    # Noise vector (ell-independent, shape (10,))
    n_ell_noise = noise_cls(noise_config) if noise_config else np.zeros(10)

    results = []
    for lmax in lmax_values:
        ell_edges, ell_eff = get_ell_bins(lmax)
        n_ell = len(ell_eff)

        # Bin C_ell to ell_eff
        cl_fid_b = _bin_cls(cl_fid, ell_eff)           # (10, n_ell)
        cl_om_plus_b = _bin_cls(cl_om_plus, ell_eff)
        cl_om_minus_b = _bin_cls(cl_om_minus, ell_eff)
        cl_s8_plus_b = _bin_cls(cl_s8_plus, ell_eff)
        cl_s8_minus_b = _bin_cls(cl_s8_minus, ell_eff)

        # Central finite differences: dC/dtheta
        dcl_dom = (cl_om_plus_b - cl_om_minus_b) / (2 * D_OMEGA_M)
        dcl_ds8 = (cl_s8_plus_b - cl_s8_minus_b) / (2 * D_S8)

        # Total C_ell = signal + noise (for covariance)
        cl_total = cl_fid_b.copy()
        for p in range(10):
            cl_total[p] += n_ell_noise[p]

        # Build (n_ell, 4, 4) matrices
        C = _to_matrix(cl_total, n_ell)
        dC_dom = _to_matrix(dcl_dom, n_ell)
        dC_ds = _to_matrix(dcl_ds8, n_ell)

        # Number of modes per ell bin
        nu_bins = np.zeros(n_ell)
        for b in range(n_ell):
            nu_bins[b] = F_SKY * (ell_edges[b + 1] ** 2 - ell_edges[b] ** 2)

        # Fisher matrix: F_ab = sum_b nu_b/2 * Tr[C^-1 dC/da C^-1 dC/db]
        F = np.zeros((2, 2))
        for l_idx in range(n_ell):
            prefactor = nu_bins[l_idx] / 2.0
            try:
                C_inv = np.linalg.inv(C[l_idx])
            except np.linalg.LinAlgError:
                continue

            derivs = [dC_dom[l_idx], dC_ds[l_idx]]
            for a in range(2):
                for b in range(a, 2):
                    val = prefactor * np.trace(C_inv @ derivs[a] @ C_inv @ derivs[b])
                    F[a, b] += val
                    if a != b:
                        F[b, a] += val

        # FoM = sqrt(det(F))
        F_inv = np.linalg.inv(F)
        fom = 1.0 / np.sqrt(np.linalg.det(F_inv))
        sigma_om = np.sqrt(F_inv[0, 0])
        sigma_s8 = np.sqrt(F_inv[1, 1])

        results.append({
            "lmax": int(lmax),
            "fisher_fom": fom,
            "sigma_omega_m": sigma_om,
            "sigma_s8": sigma_s8,
        })
        print(f"  lmax={lmax:4d}: FoM={fom:10.1f}  "
              f"sigma(Om)={sigma_om:.4f}  sigma(S8)={sigma_s8:.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading DES Y3 n(z)...")
    z_mid, nz_bins = load_des_y3_nz(NZ_PATH)

    # Precompute CAMB C_ell (shared across all noise configs)
    print("\nPrecomputing CAMB theory spectra...")
    camb_cls = precompute_camb_cls(z_mid, nz_bins)

    # Noiseless Fisher
    print("\nNoiseless Fisher FoM:")
    res_noiseless = compute_fisher_fom(camb_cls, LMAX_VALUES, noise_config=None)

    # DES Y3 noise Fisher
    print("\nDES Y3 noise Fisher FoM:")
    res_des_y3 = compute_fisher_fom(camb_cls, LMAX_VALUES, noise_config=DES_Y3)

    # LSST Y10 noise Fisher
    print("\nLSST Y10 noise Fisher FoM:")
    res_lsst_y10 = compute_fisher_fom(camb_cls, LMAX_VALUES, noise_config=LSST_Y10)

    # ── Plot ──
    lmax_arr = np.array([r["lmax"] for r in res_noiseless])
    fom_noiseless = np.array([r["fisher_fom"] for r in res_noiseless])
    fom_des_y3 = np.array([r["fisher_fom"] for r in res_des_y3])
    fom_lsst_y10 = np.array([r["fisher_fom"] for r in res_lsst_y10])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lmax_arr, fom_noiseless, "o-", color="C0", label="Noiseless (cosmic variance only)")
    ax.plot(lmax_arr, fom_lsst_y10, "D-", color="C2", label="LSST Y10 shape noise")
    ax.plot(lmax_arr, fom_des_y3, "s-", color="C1", label="DES Y3 shape noise")
    ax.set_xlabel(r"$\ell_{\max}$", fontsize=13)
    ax.set_ylabel(r"Fisher FoM$(\Omega_m, S_8)$", fontsize=13)
    ax.set_title("Theoretical Fisher FoM vs $\\ell_{\\max}$\n"
                 "4 tomo bins, 10 spectra, $f_{\\rm sky}=1/12$, 20 linear $\\ell$-bins")
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(LMAX_VALUES)

    fig.tight_layout()
    fig.savefig("fisher_fom_vs_lmax.png", dpi=150)
    print(f"\nSaved: fisher_fom_vs_lmax.png")
