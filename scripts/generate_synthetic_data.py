"""Generate synthetic power spectra datasets using JAX on GPU via Modal.

Batched jax-cosmo computation with vmap for fast generation. Both the
theory C_ell and cosmic variance noise are fully vectorized — no per-sample
Python loops.

Usage:
    # Generate on Modal GPU (fast, ~1-2 min for 70k samples)
    modal run scripts/generate_synthetic_data.py

    # Custom size
    modal run scripts/generate_synthetic_data.py --n-samples 100000

    # Download result locally
    modal volume get lensing-results synthetic/synthetic_lmax1000_n70000.npz data/synthetic/

    # Local generation (slow, no GPU needed)
    python scripts/generate_synthetic_data.py --local
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libcfitsio-dev", "pkg-config")
    .run_commands(
        "pip install setuptools && pip install numpy scipy astropy 'jax[cuda12]' jax-cosmo"
    )
    .run_commands(
        "mkdir -p /pipeline && python -c \""
        "import urllib.request; "
        "urllib.request.urlretrieve("
        "'https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/"
        "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits', "
        "'/pipeline/des_y3_2pt.fits')\""
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-synthetic", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"
NZ_PATH = "/pipeline/des_y3_2pt.fits"


@app.function(
    image=image,
    volumes={RESULTS_DIR: vol},
    gpu="any",
    timeout=600,
    memory=16384,
)
def generate_on_gpu(
    n_samples: int = 70000,
    lmax: int = 1000,
    f_sky: float = 1.0 / 12,
    n_ell: int = 20,
    seed: int = 0,
    batch_size: int = 5000,
    noise_level: str = "noiseless",
):
    """Generate synthetic spectra with batched jax-cosmo on GPU.

    Fully vectorized: vmap for theory C_ell, then vectorized numpy for
    cosmic variance + shape noise (Cholesky decomposition + matmul).

    Parameters
    ----------
    noise_level : str
        "noiseless" (cosmic variance only), "des_y3", or "lsst_y10".
        Shape noise adds N_ell to auto-spectra in both the covariance
        and observed spectra (noise bias).
    """
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pathlib import Path

    # Work around jax-cosmo's pkg_resources dependency (missing on Modal GPU)
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import types, sys
        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.DistributionNotFound = Exception
        class _FakeDist:
            version = "0.1.0"
        pkg_resources.get_distribution = lambda *a, **kw: _FakeDist()
        sys.modules["pkg_resources"] = pkg_resources

    import jax_cosmo as jc

    from lensing.spectra import TILE_SIDE_RAD, SPEC_PAIRS
    from lensing.io import load_des_y3_nz

    t0 = time.time()
    PAIR_TO_IJ = list(SPEC_PAIRS)
    ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2
    n_pairs = 10

    # --- Noise power spectrum N_ell ---
    NOISE_CONFIGS = {
        "noiseless": None,
        "des_y3": {
            "n_eff_arcmin2": [1.476, 1.479, 1.484, 1.461],
            "sigma_e": [0.243, 0.262, 0.259, 0.301],
        },
        "lsst_y10": {
            "n_eff_arcmin2": [6.75, 6.75, 6.75, 6.75],
            "sigma_e": [0.26, 0.26, 0.26, 0.26],
        },
    }

    noise_cfg = NOISE_CONFIGS[noise_level]
    # N_ell vector (10,): non-zero only for auto-spectra (i==j)
    n_ell_noise = np.zeros(n_pairs, dtype=np.float64)
    if noise_cfg is not None:
        n_eff_sr = [n * ARCMIN2_PER_SR for n in noise_cfg["n_eff_arcmin2"]]
        sigma_e = noise_cfg["sigma_e"]
        for p, (i, j) in enumerate(PAIR_TO_IJ):
            if i == j:
                n_ell_noise[p] = sigma_e[i] ** 2 / (2.0 * n_eff_sr[i])
        print(f"Shape noise ({noise_level}): N_ell auto = {n_ell_noise[n_ell_noise > 0]}")

    # --- Ell bins ---
    ell_min = 2 * np.pi / TILE_SIDE_RAD
    ell_edges = np.linspace(ell_min, lmax, n_ell + 1)
    ell_eff_full = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    ell_eff = ell_eff_full[1:]  # drop first bin
    n_ell_out = len(ell_eff)
    ell_jax = jnp.array(ell_eff, dtype=jnp.float32)

    # --- Build probes ---
    z_mid, nz_bins = load_des_y3_nz(NZ_PATH)
    nzs = []
    for i in range(nz_bins.shape[0]):
        nz = jc.redshift.kde_nz(
            jnp.array(z_mid, dtype=jnp.float32),
            jnp.array(nz_bins[i], dtype=jnp.float32),
            bw=0.01,
        )
        nzs.append(nz)
    probes = [jc.probes.WeakLensing(nzs, sigma_e=0.0)]

    # --- JIT + vmap theory computation ---
    def _single_cl(omega_m, s8):
        omega_b = 0.05
        sigma8 = s8 / jnp.sqrt(omega_m / 0.3)
        cosmo = jc.Planck15(
            Omega_c=omega_m - omega_b, Omega_b=omega_b,
            h=0.67, n_s=0.96, sigma8=sigma8, w0=-1.0,
        )
        return jc.angular_cl.angular_cl(cosmo, ell_jax, probes)

    batched_cl = jax.jit(jax.vmap(_single_cl))

    # Warm up JIT
    _ = batched_cl(jnp.array([0.3]), jnp.array([0.8]))
    print(f"JIT compiled in {time.time() - t0:.1f}s")

    # --- Sample cosmological parameters ---
    rng = np.random.default_rng(seed)
    omega_m_vals = rng.uniform(0.13, 0.46, size=n_samples).astype(np.float32)
    s8_vals = rng.uniform(0.5, 1.0, size=n_samples).astype(np.float32)
    theta = np.column_stack([omega_m_vals, s8_vals])

    # --- Batched theory C_ell ---
    print(f"Computing theory C_ell for {n_samples} samples (batch_size={batch_size})...")
    t1 = time.time()
    all_cls = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        cls_batch = batched_cl(
            jnp.array(omega_m_vals[start:end]),
            jnp.array(s8_vals[start:end]),
        )
        all_cls.append(np.array(cls_batch))
        if end % 10000 == 0 or end == n_samples:
            print(f"  {end}/{n_samples}")

    cls_theory_all = np.concatenate(all_cls, axis=0)  # (n_samples, 10, n_ell_out)
    print(f"Theory done in {time.time() - t1:.1f}s")

    # --- Vectorized cosmic variance + shape noise ---
    # Covariance uses C_total = C_signal + N_ell:
    #   Cov(C^{ab}, C^{cd}) = (C_tot^{ac}*C_tot^{bd} + C_tot^{ad}*C_tot^{bc}) / nu
    # Observed spectra = C_signal + N_ell (noise bias) + noise realization
    print(f"Adding noise (vectorized, noise_level={noise_level})...")
    t2 = time.time()

    nu = (2 * ell_eff + 1) * f_sky  # (n_ell_out,)

    # N_ell matrix for 4x4 format (diagonal only)
    n_ell_mat = np.zeros((4, 4), dtype=np.float64)
    for p, (i, j) in enumerate(PAIR_TO_IJ):
        if i == j:
            n_ell_mat[i, j] = n_ell_noise[p]

    noise_chunk = 10000
    spectra = np.zeros((n_samples, n_pairs * n_ell_out), dtype=np.float32)

    for chunk_start in range(0, n_samples, noise_chunk):
        chunk_end = min(chunk_start + noise_chunk, n_samples)
        chunk_n = chunk_end - chunk_start
        cls_chunk = cls_theory_all[chunk_start:chunk_end]  # (chunk_n, 10, n_ell_out)

        # Build 4x4 C_total matrix: signal + noise
        cl_mat = np.zeros((chunk_n, n_ell_out, 4, 4), dtype=np.float64)
        for p, (a, b) in enumerate(PAIR_TO_IJ):
            cl_mat[:, :, a, b] = cls_chunk[:, p, :]
            if a != b:
                cl_mat[:, :, b, a] = cls_chunk[:, p, :]
        # Add shape noise N_ell to diagonal (auto-spectra)
        cl_mat += n_ell_mat[np.newaxis, np.newaxis, :, :]

        # Build 10x10 covariance using C_total
        cov = np.zeros((chunk_n, n_ell_out, n_pairs, n_pairs), dtype=np.float64)
        for p1, (a, b) in enumerate(PAIR_TO_IJ):
            for p2, (c, d) in enumerate(PAIR_TO_IJ):
                cov[:, :, p1, p2] = (
                    cl_mat[:, :, a, c] * cl_mat[:, :, b, d] +
                    cl_mat[:, :, a, d] * cl_mat[:, :, b, c]
                ) / nu[np.newaxis, :]

        # Cholesky + noise realization
        cov += 1e-35 * np.eye(n_pairs)[np.newaxis, np.newaxis, :, :]
        flat_shape = (chunk_n * n_ell_out, n_pairs, n_pairs)
        cov_flat = cov.reshape(flat_shape)
        L_flat = np.linalg.cholesky(cov_flat)
        L = L_flat.reshape(chunk_n, n_ell_out, n_pairs, n_pairs)

        z = rng.standard_normal((chunk_n, n_ell_out, n_pairs, 1))
        noise = (L @ z).squeeze(-1)  # (chunk_n, n_ell_out, 10)

        # Observed = signal + noise_bias + noise_realization
        cls_obs = cls_chunk.copy()
        # Add N_ell noise bias to auto-spectra
        cls_obs += n_ell_noise[np.newaxis, :, np.newaxis]
        # Add noise realization
        cls_obs += noise.transpose(0, 2, 1)
        spectra[chunk_start:chunk_end] = cls_obs.reshape(chunk_n, -1).astype(np.float32)

        if chunk_end % 10000 == 0 or chunk_end == n_samples:
            print(f"  Noise: {chunk_end}/{n_samples}")

    print(f"Noise done in {time.time() - t2:.1f}s")

    # --- Save ---
    out_dir = Path(RESULTS_DIR) / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{noise_level}" if noise_level != "noiseless" else ""
    out_path = out_dir / f"synthetic_lmax{lmax}_n{n_samples}{suffix}.npz"

    np.savez(
        out_path,
        spectra=spectra,
        theta=theta.astype(np.float32),
        ell_eff=ell_eff.astype(np.float32),
        lmax=lmax,
        f_sky=f_sky,
        n_samples=n_samples,
        seed=seed,
        noise_level=noise_level,
    )
    vol.commit()

    total = time.time() - t0
    print(f"\nSaved {out_path}")
    print(f"Shape: spectra={spectra.shape}, theta={theta.shape}")
    print(f"Total: {total:.1f}s ({total/60:.1f} min)")
    return str(out_path)


@app.local_entrypoint()
def main(
    n_samples: int = 70000,
    lmax: int = 1000,
    seed: int = 0,
    noise_level: str = "des_y3",
    local: bool = False,
):
    if local:
        raise NotImplementedError("Local generation not yet updated for noise_level. Use Modal.")
    else:
        result = generate_on_gpu.remote(
            n_samples=n_samples,
            lmax=lmax,
            seed=seed,
            noise_level=noise_level,
        )
        print(f"\nResult: {result}")
        suffix = f"_{noise_level}" if noise_level != "noiseless" else ""
        print(f"\nDownload with:")
        print(f"  modal volume get lensing-results synthetic/synthetic_lmax{lmax}_n{n_samples}{suffix}.npz data/synthetic/")
