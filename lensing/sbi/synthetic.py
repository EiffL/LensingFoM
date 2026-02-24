"""Synthetic power spectra generator with cosmic variance noise.

Generates mock C_ell observations as:
    C_ell^obs = C_ell^theory(Omega_m, S8) + noise
where noise is drawn from the Gaussian covariance of the power spectrum estimator
(cosmic variance). This provides unlimited training data for validating the
VMIM compressor + FoM pipeline against Fisher theory.

Design: jax-cosmo is slow per call, so we JIT-compile the theory computation
and save generated datasets to disk (.npz). Training scripts load from disk.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from lensing.spectra import TILE_SIDE_RAD, SPEC_PAIRS

# Pair index mapping: position in the 10-element vector → (i, j) bin indices
_PAIR_TO_IJ = list(SPEC_PAIRS)  # [(0,0), (0,1), (0,2), (0,3), (1,1), ...]

# Cache probes across calls (they only depend on n(z) which is fixed)
_PROBES_CACHE = {}
_DEFAULT_NZ_PATH = "data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"


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
    """Build jax-cosmo weak lensing probes from DES Y3 n(z)."""
    nzs = []
    for i in range(nz_bins.shape[0]):
        nz = jc.redshift.kde_nz(
            jnp.array(z_mid, dtype=jnp.float32),
            jnp.array(nz_bins[i], dtype=jnp.float32),
            bw=0.01,
        )
        nzs.append(nz)
    return [jc.probes.WeakLensing(nzs, sigma_e=0.0)]


def _get_probes(nz_path=None):
    """Load DES Y3 n(z) and build probes, caching the result."""
    if "probes" not in _PROBES_CACHE:
        from lensing.io import load_des_y3_nz
        path = nz_path or _DEFAULT_NZ_PATH
        z_mid, nz_bins = load_des_y3_nz(path)
        _PROBES_CACHE["probes"] = _make_probes(z_mid, nz_bins)
        _PROBES_CACHE["z_mid"] = z_mid
        _PROBES_CACHE["nz_bins"] = nz_bins
    return _PROBES_CACHE["probes"]


def get_ell_bins(lmax, n_bins=20):
    """Compute ell bin edges and effective ell values matching lensing.spectra.

    Returns
    -------
    ell_edges : np.ndarray, shape (n_bins+1,)
    ell_eff : np.ndarray, shape (n_bins,)
    """
    ell_min = 2 * np.pi / TILE_SIDE_RAD
    ell_edges = np.linspace(ell_min, lmax, n_bins + 1)
    ell_eff = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    return ell_edges, ell_eff


def theory_cls(omega_m, s8, ell):
    """Compute theory C_ell for all 10 spectra at given multipoles.

    Parameters
    ----------
    omega_m : float
        Matter density parameter.
    s8 : float
        S8 = sigma8 * sqrt(Omega_m / 0.3).
    ell : array-like, shape (n_ell,)
        Multipole values.

    Returns
    -------
    cls : np.ndarray, shape (10, n_ell)
        Theory power spectra for all 10 unique (i,j) pairs.
    """
    probes = _get_probes()
    cosmo = _make_cosmology(omega_m, s8)
    ell_jax = jnp.array(ell, dtype=jnp.float32)
    cls = jc.angular_cl.angular_cl(cosmo, ell_jax, probes)
    return np.array(cls)


def _theory_cls_jit(probes, ell_jax):
    """Return a JIT-compiled function: (omega_m, s8) -> C_ell (10, n_ell)."""
    @jax.jit
    def _compute(omega_m, s8):
        cosmo = _make_cosmology(omega_m, s8)
        return jc.angular_cl.angular_cl(cosmo, ell_jax, probes)
    return _compute


def cosmic_variance_cov(cl_matrix_4x4, ell_eff, f_sky):
    """Compute block-diagonal covariance for vectorized C_ell.

    At each ell, the covariance of the 10 unique C_ell^{ij} entries is:
        Cov(C^{ab}, C^{cd}) = (C^{ac}*C^{bd} + C^{ad}*C^{bc}) / ((2*ell+1) * f_sky)

    Parameters
    ----------
    cl_matrix_4x4 : np.ndarray, shape (n_ell, 4, 4)
        Symmetric C_ell matrix at each multipole.
    ell_eff : np.ndarray, shape (n_ell,)
        Effective multipole values.
    f_sky : float
        Sky fraction.

    Returns
    -------
    cov_blocks : np.ndarray, shape (n_ell, 10, 10)
        Covariance matrix at each ell for the 10 unique pairs.
    """
    n_ell = len(ell_eff)
    n_pairs = 10
    cov_blocks = np.zeros((n_ell, n_pairs, n_pairs))

    for l_idx in range(n_ell):
        nu = (2 * ell_eff[l_idx] + 1) * f_sky
        C = cl_matrix_4x4[l_idx]  # (4, 4)

        for p1, (a, b) in enumerate(_PAIR_TO_IJ):
            for p2, (c, d) in enumerate(_PAIR_TO_IJ):
                cov_blocks[l_idx, p1, p2] = (
                    C[a, c] * C[b, d] + C[a, d] * C[b, c]
                ) / nu

    return cov_blocks


def _cls_vec_to_matrix(cls_vec):
    """Convert (10, n_ell) spectra to (n_ell, 4, 4) symmetric matrix."""
    n_ell = cls_vec.shape[1]
    mat = np.zeros((n_ell, 4, 4))
    for p, (i, j) in enumerate(_PAIR_TO_IJ):
        mat[:, i, j] = cls_vec[p]
        if i != j:
            mat[:, j, i] = cls_vec[p]
    return mat


def sample_mock_spectra(
    n_samples,
    lmax,
    f_sky=1.0 / 12,
    n_ell=20,
    omega_m_range=(0.13, 0.46),
    s8_range=(0.5, 1.0),
    seed=0,
    drop_first_bin=True,
):
    """Generate synthetic power spectra with cosmic variance noise.

    Uses JIT-compiled jax-cosmo for theory C_ell computation.

    Parameters
    ----------
    n_samples : int
        Number of mock observations to generate.
    lmax : int
        Maximum multipole.
    f_sky : float
        Sky fraction (default 1/12 for one HEALPix base tile).
    n_ell : int
        Number of ell bins (default 20).
    omega_m_range : tuple
        (min, max) for uniform Omega_m prior.
    s8_range : tuple
        (min, max) for uniform S8 prior.
    seed : int
        Random seed.
    drop_first_bin : bool
        If True, drop the first ell bin (too low for flat-sky). Default True.

    Returns
    -------
    spectra : np.ndarray, shape (n_samples, 10 * n_ell_out)
        Noisy mock power spectra, vectorized across pairs and ell bins.
    theta : np.ndarray, shape (n_samples, 2)
        Cosmological parameters [Omega_m, S8].
    ell_eff : np.ndarray, shape (n_ell_out,)
        Effective multipole values used.
    """
    rng = np.random.default_rng(seed)

    # Ell binning matching lensing.spectra
    _, ell_eff_full = get_ell_bins(lmax, n_bins=n_ell)

    if drop_first_bin:
        ell_eff = ell_eff_full[1:]
    else:
        ell_eff = ell_eff_full
    n_ell_out = len(ell_eff)

    # Sample cosmological parameters
    omega_m_vals = rng.uniform(*omega_m_range, size=n_samples)
    s8_vals = rng.uniform(*s8_range, size=n_samples)
    theta = np.column_stack([omega_m_vals, s8_vals])

    # JIT-compile theory computation
    probes = _get_probes()
    ell_jax = jnp.array(ell_eff, dtype=jnp.float32)
    compute_cl = _theory_cls_jit(probes, ell_jax)

    # Warm up JIT
    _ = compute_cl(jnp.float32(0.3), jnp.float32(0.8))
    print(f"  JIT compiled. Generating {n_samples} samples...")

    spectra = np.zeros((n_samples, 10 * n_ell_out))

    for i in range(n_samples):
        # Theory C_ell (JIT-compiled, fast after warmup)
        cls_theory = np.array(
            compute_cl(jnp.float32(omega_m_vals[i]), jnp.float32(s8_vals[i]))
        )  # (10, n_ell_out)

        # Build 4x4 C_ell matrix for covariance computation
        cl_matrix = _cls_vec_to_matrix(cls_theory)  # (n_ell_out, 4, 4)

        # Cosmic variance covariance at each ell
        cov_blocks = cosmic_variance_cov(cl_matrix, ell_eff, f_sky)  # (n_ell_out, 10, 10)

        # Sample noise independently at each ell
        cls_noisy = cls_theory.copy()
        for l_idx in range(n_ell_out):
            cov = cov_blocks[l_idx]
            # Regularize: ensure positive definite
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.min() < 0:
                cov += (abs(eigvals.min()) + 1e-30) * np.eye(10)
            noise = rng.multivariate_normal(np.zeros(10), cov)
            cls_noisy[:, l_idx] += noise

        # Vectorize: stack all 10 pairs across ell bins
        spectra[i] = cls_noisy.ravel()

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")

    return spectra, theta, ell_eff
