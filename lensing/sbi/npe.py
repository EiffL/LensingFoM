"""Neural Posterior Estimation with sbi package."""

import numpy as np
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform


def get_prior(theta_raw, padding=0.1):
    """Build a BoxUniform prior with padding around observed parameter range.

    Parameters
    ----------
    theta_raw : np.ndarray, shape (N, 2)
        Unnormalized [Omega_m, S8] values.
    padding : float
        Fractional padding around min/max (default 10%).

    Returns
    -------
    prior : BoxUniform
        Prior distribution for sbi.
    """
    lo = theta_raw.min(axis=0)
    hi = theta_raw.max(axis=0)
    extent = hi - lo
    low = torch.tensor(lo - padding * extent, dtype=torch.float32)
    high = torch.tensor(hi + padding * extent, dtype=torch.float32)
    return BoxUniform(low=low, high=high)


def train_npe(summaries, theta, prior):
    """Train NPE on compressed summaries and cosmological parameters.

    Parameters
    ----------
    summaries : np.ndarray, shape (N, 2)
        Compressed summary statistics from VMIM compressor.
    theta : np.ndarray, shape (N, 2)
        Cosmological parameters [Omega_m, S8].
    prior : BoxUniform
        Prior distribution.

    Returns
    -------
    posterior : sbi posterior object
        Trained posterior ready for sampling.
    """
    inference = NPE(prior=prior)
    theta_t = torch.tensor(theta, dtype=torch.float32)
    x_t = torch.tensor(summaries, dtype=torch.float32)
    inference.append_simulations(theta_t, x_t)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def compute_fom_single(posterior, obs, n_samples=10_000):
    """Compute FoM for a single observation.

    FoM = 1 / sqrt(det(Cov)) of posterior samples in (Omega_m, S8) space.

    Parameters
    ----------
    posterior : sbi posterior object
    obs : np.ndarray, shape (2,)
        Single compressed observation.
    n_samples : int
        Number of posterior samples to draw.

    Returns
    -------
    fom : float
        Figure of merit.
    """
    x_t = torch.tensor(obs, dtype=torch.float32)
    samples = posterior.sample((n_samples,), x=x_t).numpy()
    cov = np.cov(samples.T)
    det = np.linalg.det(cov)
    return 1.0 / np.sqrt(det) if det > 0 else 0.0


def compute_fom(posterior, test_summaries, n_samples=10_000, n_bootstrap=100, seed=42):
    """Compute FoM across all test observations with bootstrap error bars.

    Parameters
    ----------
    posterior : sbi posterior object
    test_summaries : np.ndarray, shape (N_test, 2)
        Compressed test observations.
    n_samples : int
        Posterior samples per observation.
    n_bootstrap : int
        Number of bootstrap resamples for error bars.
    seed : int
        Random seed for bootstrap.

    Returns
    -------
    median : float
        Median FoM across test observations.
    lo_16 : float
        16th percentile of bootstrap median distribution.
    hi_84 : float
        84th percentile of bootstrap median distribution.
    all_foms : np.ndarray
        FoM for each test observation.
    """
    all_foms = np.array([
        compute_fom_single(posterior, test_summaries[i], n_samples)
        for i in range(len(test_summaries))
    ])

    median = np.median(all_foms)

    # Bootstrap error bars on the median
    rng = np.random.default_rng(seed)
    boot_medians = np.array([
        np.median(rng.choice(all_foms, size=len(all_foms), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo_16 = np.percentile(boot_medians, 16)
    hi_84 = np.percentile(boot_medians, 84)

    return median, lo_16, hi_84, all_foms
