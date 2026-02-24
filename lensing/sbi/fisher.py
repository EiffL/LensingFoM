"""Fisher matrix FoM prediction using jax-cosmo.

Computes the theoretical Fisher information matrix for weak lensing power spectra
(Omega_m, S8) as a function of lmax, using autodiff for parameter derivatives.
"""

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

# DES Y3 MagLim effective number densities per arcmin^2 per tomographic bin
# and intrinsic ellipticity dispersion (not used for noiseless Fisher)
SIGMA_E = 0.26  # per component
N_EFF_ARCMIN2 = [1.47, 1.46, 1.24, 0.85]  # DES Y3 MagLim bins 1-4


def _make_cosmology(omega_m, s8, h=0.67, n_s=0.96, omega_b=0.05, w0=-1.0):
    """Create a jax-cosmo Cosmology from (Omega_m, S8) + fixed nuisance."""
    sigma8 = s8 / jnp.sqrt(omega_m / 0.3)
    return jc.Planck15(
        Omega_c=omega_m - omega_b,
        Omega_b=omega_b,
        h=h,
        n_s=n_s,
        sigma8=sigma8,
        w0=w0,
    )


def _make_probes(z_mid, nz_bins):
    """Build jax-cosmo weak lensing probes from DES Y3 n(z).

    Parameters
    ----------
    z_mid : array, shape (n_z,)
        Redshift bin midpoints.
    nz_bins : array, shape (4, n_z,)
        Normalized n(z) for each tomographic bin.

    Returns
    -------
    probes : list of jc.probes.WeakLensing
    """
    nzs = []
    for i in range(nz_bins.shape[0]):
        nz = jc.redshift.kde_nz(
            jnp.array(z_mid, dtype=jnp.float32),
            jnp.array(nz_bins[i], dtype=jnp.float32),
            bw=0.01,
        )
        nzs.append(nz)
    probes = [jc.probes.WeakLensing(nzs, sigma_e=0.0)]
    return probes


def _cl_matrix(cosmo, probes, ell):
    """Compute 4x4 C_ell matrix at given multipoles.

    Returns
    -------
    cls : array, shape (n_pairs, n_ell)
        Power spectra for all 10 unique (i,j) pairs.
    """
    return jc.angular_cl.angular_cl(cosmo, ell, probes)


def compute_fisher_fom(z_mid, nz_bins, lmax_values, f_sky=1.0 / 12,
                       omega_m_fid=0.3, s8_fid=0.8, n_ell=20):
    """Compute Fisher FoM(Omega_m, S8) for each lmax.

    Parameters
    ----------
    z_mid : np.ndarray, shape (n_z,)
        Redshift bin midpoints from DES Y3 n(z).
    nz_bins : np.ndarray, shape (4, n_z)
        Normalized n(z) per tomographic bin.
    lmax_values : list of int
        Maximum multipole values to evaluate.
    f_sky : float
        Sky fraction (default 1/12 for one HEALPix base tile).
    omega_m_fid : float
        Fiducial Omega_m.
    s8_fid : float
        Fiducial S8.
    n_ell : int
        Number of linearly-spaced ell bins.

    Returns
    -------
    results : list of dict
        One dict per lmax with keys: lmax, fisher_fom, sigma_omega_m, sigma_s8.
    """
    probes = _make_probes(z_mid, nz_bins)
    n_bins = nz_bins.shape[0]
    n_pairs = n_bins * (n_bins + 1) // 2

    # Build pair index mapping: (i,j) with i<=j to matrix indices
    pair_to_ij = []
    for i in range(n_bins):
        for j in range(i, n_bins):
            pair_to_ij.append((i, j))

    def _cls_for_params(omega_m, s8, ell):
        cosmo = _make_cosmology(omega_m, s8)
        return _cl_matrix(cosmo, probes, ell)

    # Autodiff: derivatives of C_ell w.r.t. (Omega_m, S8)
    dcl_domega_m = jax.jacfwd(_cls_for_params, argnums=0)
    dcl_ds8 = jax.jacfwd(_cls_for_params, argnums=1)

    results = []
    for lmax in lmax_values:
        ell = jnp.linspace(20, lmax, n_ell)

        # Fiducial C_ell: shape (n_pairs, n_ell)
        cl_fid = _cls_for_params(omega_m_fid, s8_fid, ell)

        # Derivatives: shape (n_pairs, n_ell)
        dcl_dom = dcl_domega_m(omega_m_fid, s8_fid, ell)
        dcl_ds = dcl_ds8(omega_m_fid, s8_fid, ell)

        # Reshape into (n_ell, n_bins, n_bins) covariance matrices
        def _to_matrix(cl_vec):
            """Convert (n_pairs, n_ell) to (n_ell, n_bins, n_bins)."""
            mat = jnp.zeros((len(ell), n_bins, n_bins))
            for p, (i, j) in enumerate(pair_to_ij):
                mat = mat.at[:, i, j].set(cl_vec[p])
                if i != j:
                    mat = mat.at[:, j, i].set(cl_vec[p])
            return mat

        C = _to_matrix(cl_fid)        # (n_ell, 4, 4)
        dC_dom = _to_matrix(dcl_dom)   # (n_ell, 4, 4)
        dC_ds = _to_matrix(dcl_ds)     # (n_ell, 4, 4)

        # Fisher matrix: F_ab = sum_ell (2*ell+1)*f_sky/2 * Tr[C^-1 dC/da C^-1 dC/db]
        F = jnp.zeros((2, 2))
        for l_idx in range(len(ell)):
            ell_val = ell[l_idx]
            prefactor = (2 * ell_val + 1) * f_sky / 2.0
            C_inv = jnp.linalg.inv(C[l_idx])

            derivs = [dC_dom[l_idx], dC_ds[l_idx]]
            for a in range(2):
                for b in range(a, 2):
                    val = prefactor * jnp.trace(C_inv @ derivs[a] @ C_inv @ derivs[b])
                    F = F.at[a, b].add(val)
                    if a != b:
                        F = F.at[b, a].add(val)

        # FoM = 1 / sqrt(det(F^-1)) = sqrt(det(F))
        F_inv = jnp.linalg.inv(F)
        fom = float(1.0 / jnp.sqrt(jnp.linalg.det(F_inv)))
        sigma_om = float(jnp.sqrt(F_inv[0, 0]))
        sigma_s8 = float(jnp.sqrt(F_inv[1, 1]))

        results.append({
            "lmax": int(lmax),
            "fisher_fom": fom,
            "sigma_omega_m": sigma_om,
            "sigma_s8": sigma_s8,
        })
        print(f"  lmax={lmax:4d}: Fisher FoM={fom:10.1f}  "
              f"sigma(Om)={sigma_om:.4f}  sigma(S8)={sigma_s8:.4f}")

    return results
