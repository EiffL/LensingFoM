"""Gaussian Neural Posterior Estimation (no external SBI dependency).

Learns p(theta | x) = N(mu(x), Sigma(x)) where the mean and covariance
are predicted by a small MLP conditioned on compressed summary statistics.
Since the posterior is Gaussian, FoM = 1/sqrt(det(Sigma)) comes directly
from the network output — no sampling needed.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


class GaussianNPE(L.LightningModule):
    """Gaussian neural posterior estimator.

    Predicts p(theta | x) = N(mu(x), Sigma(x)) for 2D theta = (Omega_m, S8).
    Sigma is parameterized via its Cholesky factor L so that Sigma = L L^T
    is always positive definite.

    Parameters
    ----------
    input_dim : int
        Dimension of compressed summaries (default 2).
    theta_dim : int
        Dimension of parameter space (default 2).
    lr : float
        Learning rate.
    """

    def __init__(self, input_dim=2, theta_dim=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.theta_dim = theta_dim

        # Number of Cholesky lower-triangle entries: d*(d+1)/2
        n_chol = theta_dim * (theta_dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, theta_dim + n_chol),  # mu (2) + L entries (3)
        )

    def forward(self, x):
        """Predict posterior mean and Cholesky factor.

        Returns
        -------
        mu : (B, 2)
        L : (B, 2, 2)  lower-triangular with positive diagonal
        """
        out = self.net(x)
        d = self.theta_dim
        mu = out[:, :d]
        chol_raw = out[:, d:]

        # Build lower-triangular Cholesky factor
        L = torch.zeros(x.shape[0], d, d, device=x.device)
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    # Positive diagonal via softplus
                    L[:, i, j] = nn.functional.softplus(chol_raw[:, idx]) + 1e-6
                else:
                    L[:, i, j] = chol_raw[:, idx]
                idx += 1

        return mu, L

    def _nll(self, x, theta):
        """Gaussian negative log-likelihood.

        -log N(theta; mu, L L^T) = 0.5 * ||L^{-1}(theta - mu)||^2
                                   + sum(log diag(L)) + const
        """
        mu, L = self.forward(x)
        diff = theta - mu  # (B, d)
        # Solve L z = diff for z, then NLL = 0.5*||z||^2 + log|det(L)|
        z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
        z = z.squeeze(-1)  # (B, d)
        log_det_L = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B,)
        nll = 0.5 * (z**2).sum(-1) + log_det_L  # (B,)
        return nll.mean()

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss = self._nll(x, theta)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss = self._nll(x, theta)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def predict_fom(self, x):
        """Compute FoM for each observation directly from predicted covariance.

        FoM = 1 / sqrt(det(Sigma)) = 1 / prod(diag(L))

        Parameters
        ----------
        x : np.ndarray, shape (N, input_dim)
            Compressed summary statistics.

        Returns
        -------
        foms : np.ndarray, shape (N,)
        """
        self.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        _, L = self.forward(x_t)
        # det(Sigma) = det(L L^T) = det(L)^2 = prod(diag(L))^2
        log_det_sigma = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        foms = torch.exp(-0.5 * log_det_sigma).cpu().numpy()
        return foms


def train_npe(summaries, theta, val_summaries=None, val_theta=None,
              max_epochs=200, patience=15):
    """Train Gaussian NPE on compressed summaries.

    Parameters
    ----------
    summaries : np.ndarray, shape (N, 2)
        Compressed training summaries.
    theta : np.ndarray, shape (N, 2)
        Training cosmological parameters [Omega_m, S8].
    val_summaries, val_theta : np.ndarray, optional
        Validation data for early stopping. If None, uses training data.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.

    Returns
    -------
    model : GaussianNPE
        Trained model.
    """
    x_train = torch.tensor(summaries, dtype=torch.float32)
    t_train = torch.tensor(theta, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(x_train, t_train), batch_size=256, shuffle=True)

    if val_summaries is not None:
        x_val = torch.tensor(val_summaries, dtype=torch.float32)
        t_val = torch.tensor(val_theta, dtype=torch.float32)
        val_dl = DataLoader(TensorDataset(x_val, t_val), batch_size=256)
    else:
        val_dl = train_dl

    model = GaussianNPE(input_dim=summaries.shape[1], theta_dim=theta.shape[1])

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dl, val_dl)
    return model


def compute_fom(npe_model, test_summaries, n_bootstrap=100, seed=42):
    """Compute FoM across test observations with bootstrap error bars.

    Parameters
    ----------
    npe_model : GaussianNPE
        Trained Gaussian NPE model.
    test_summaries : np.ndarray, shape (N_test, 2)
        Compressed test observations.
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
    all_foms = npe_model.predict_fom(test_summaries)

    median = float(np.median(all_foms))

    # Bootstrap error bars on the median
    rng = np.random.default_rng(seed)
    boot_medians = np.array([
        np.median(rng.choice(all_foms, size=len(all_foms), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo_16 = float(np.percentile(boot_medians, 16))
    hi_84 = float(np.percentile(boot_medians, 84))

    return median, lo_16, hi_84, all_foms
