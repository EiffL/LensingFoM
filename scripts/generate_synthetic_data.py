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
):
    """Generate synthetic spectra with batched jax-cosmo on GPU.

    Fully vectorized: vmap for theory C_ell, then vectorized numpy for
    cosmic variance noise (Cholesky decomposition + matmul, no per-sample loop).
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
    n_pairs = 10

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

    # --- Vectorized cosmic variance noise ---
    # For each sample, at each ell:
    #   Cov(C^{ab}, C^{cd}) = (C^{ac}*C^{bd} + C^{ad}*C^{bc}) / nu
    # where nu = (2*ell+1)*f_sky
    #
    # Strategy: build all covariance matrices, Cholesky decompose, multiply
    # by standard normal draws. Process in chunks to manage memory.
    print("Adding cosmic variance noise (vectorized)...")
    t2 = time.time()

    # Pre-build pair index arrays for vectorized covariance construction
    pair_a = np.array([a for a, b in PAIR_TO_IJ])  # (10,)
    pair_b = np.array([b for a, b in PAIR_TO_IJ])  # (10,)
    nu = (2 * ell_eff + 1) * f_sky  # (n_ell_out,)

    noise_chunk = 10000  # process this many samples at once
    spectra = np.zeros((n_samples, n_pairs * n_ell_out), dtype=np.float32)

    for chunk_start in range(0, n_samples, noise_chunk):
        chunk_end = min(chunk_start + noise_chunk, n_samples)
        chunk_n = chunk_end - chunk_start
        cls_chunk = cls_theory_all[chunk_start:chunk_end]  # (chunk_n, 10, n_ell_out)

        # Build 4x4 C_ell matrix: (chunk_n, n_ell_out, 4, 4)
        cl_mat = np.zeros((chunk_n, n_ell_out, 4, 4), dtype=np.float64)
        for p, (a, b) in enumerate(PAIR_TO_IJ):
            cl_mat[:, :, a, b] = cls_chunk[:, p, :]
            if a != b:
                cl_mat[:, :, b, a] = cls_chunk[:, p, :]

        # Build 10x10 covariance at each (sample, ell): (chunk_n, n_ell_out, 10, 10)
        # Cov[p1,p2] = (C[a,c]*C[b,d] + C[a,d]*C[b,c]) / nu
        # Use advanced indexing for full vectorization
        cov = np.zeros((chunk_n, n_ell_out, n_pairs, n_pairs), dtype=np.float64)
        for p1, (a, b) in enumerate(PAIR_TO_IJ):
            for p2, (c, d) in enumerate(PAIR_TO_IJ):
                cov[:, :, p1, p2] = (
                    cl_mat[:, :, a, c] * cl_mat[:, :, b, d] +
                    cl_mat[:, :, a, d] * cl_mat[:, :, b, c]
                ) / nu[np.newaxis, :]

        # Cholesky decompose at each (sample, ell)
        # Add small regularization for numerical stability
        cov += 1e-35 * np.eye(n_pairs)[np.newaxis, np.newaxis, :, :]
        # Shape: (chunk_n, n_ell_out, 10, 10) -> reshape for batch cholesky
        flat_shape = (chunk_n * n_ell_out, n_pairs, n_pairs)
        cov_flat = cov.reshape(flat_shape)
        L_flat = np.linalg.cholesky(cov_flat)  # (chunk_n*n_ell_out, 10, 10)
        L = L_flat.reshape(chunk_n, n_ell_out, n_pairs, n_pairs)

        # Draw standard normal and transform: noise = L @ z
        z = rng.standard_normal((chunk_n, n_ell_out, n_pairs, 1))
        noise = (L @ z).squeeze(-1)  # (chunk_n, n_ell_out, 10)

        # Add noise to theory and flatten to (chunk_n, 10*n_ell_out)
        cls_noisy = cls_chunk + noise.transpose(0, 2, 1)  # (chunk_n, 10, n_ell_out)
        spectra[chunk_start:chunk_end] = cls_noisy.reshape(chunk_n, -1).astype(np.float32)

        if chunk_end % 10000 == 0 or chunk_end == n_samples:
            print(f"  Noise: {chunk_end}/{n_samples}")

    print(f"Noise done in {time.time() - t2:.1f}s")

    # --- Save ---
    out_dir = Path(RESULTS_DIR) / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_lmax{lmax}_n{n_samples}.npz"

    np.savez(
        out_path,
        spectra=spectra,
        theta=theta.astype(np.float32),
        ell_eff=ell_eff.astype(np.float32),
        lmax=lmax,
        f_sky=f_sky,
        n_samples=n_samples,
        seed=seed,
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
    local: bool = False,
):
    if local:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lensing.sbi.synthetic import sample_mock_spectra
        import numpy as np
        import time

        out_dir = Path("data/synthetic")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synthetic_lmax{lmax}_n{n_samples}.npz"

        if out_path.exists():
            print(f"{out_path} already exists")
            return

        t0 = time.time()
        spectra, theta, ell_eff = sample_mock_spectra(n_samples, lmax, seed=seed)
        elapsed = time.time() - t0

        np.savez(
            out_path,
            spectra=spectra.astype(np.float32),
            theta=theta.astype(np.float32),
            ell_eff=ell_eff.astype(np.float32),
            lmax=lmax,
            f_sky=1.0/12,
            n_samples=n_samples,
            seed=seed,
        )
        print(f"Saved {out_path} in {elapsed:.1f}s")
    else:
        result = generate_on_gpu.remote(
            n_samples=n_samples,
            lmax=lmax,
            seed=seed,
        )
        print(f"\nResult: {result}")
        print(f"\nDownload with:")
        print(f"  modal volume get lensing-results synthetic/synthetic_lmax{lmax}_n{n_samples}.npz data/synthetic/")
