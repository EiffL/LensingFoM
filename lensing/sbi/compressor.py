"""VMIM compressor: MLP that compresses spectra to low-dim summaries."""

import torch
import torch.nn as nn

import lightning as L


class VMIMCompressor(L.LightningModule):
    """Variational Mutual Information Maximization compressor.

    Compresses spectra into low-dim summaries by maximizing mutual
    information with cosmological parameters. Supports diagonal or
    full-covariance (Cholesky) Gaussian VMIM head with FoM logging.
    """

    def __init__(
        self,
        input_dim=200,
        summary_dim=2,
        theta_dim=2,
        hidden_dim=64,
        dropout=0.3,
        full_cov=False,
        lr=5e-4,
        weight_decay=1e-2,
        theta_std=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["theta_std"])
        if theta_std is not None:
            self.register_buffer("theta_std", torch.tensor(theta_std, dtype=torch.float32))
        else:
            self.theta_std = None

        # Compressor MLP with dropout
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, summary_dim),
        )

        # VMIM head: linear map from summary to posterior params
        if full_cov:
            n_chol = theta_dim * (theta_dim + 1) // 2
            head_out = theta_dim + n_chol
        else:
            head_out = theta_dim * 2

        self.vmim_head = nn.Linear(summary_dim, head_out)

    def forward(self, x):
        return self.compressor(x)

    def _predict_posterior(self, t):
        """From summary t, predict posterior mean and covariance factor.

        Returns
        -------
        mu : (B, theta_dim)
        L_or_sigma : (B, theta_dim, theta_dim) if full_cov else (B, theta_dim)
            Cholesky factor or diagonal sigma.
        """
        out = self.vmim_head(t)
        d = self.hparams.theta_dim

        if self.hparams.full_cov:
            mu = out[:, :d]
            chol_raw = out[:, d:]
            B = t.shape[0]
            L = torch.zeros(B, d, d, device=t.device)
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        L[:, i, j] = nn.functional.softplus(chol_raw[:, idx]) + 1e-4
                    else:
                        L[:, i, j] = chol_raw[:, idx]
                    idx += 1
            return mu, L
        else:
            mu = out[:, :d]
            sigma = nn.functional.softplus(out[:, d:]) + 1e-4
            return mu, sigma

    def _nll_and_fom(self, x, theta):
        """Compute Gaussian NLL and per-sample FoM."""
        t = self.compressor(x)
        mu, L_or_sigma = self._predict_posterior(t)

        if self.hparams.full_cov:
            L = L_or_sigma
            diff = theta - mu
            z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
            z = z.squeeze(-1)
            log_det_L = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            nll = 0.5 * (z ** 2).sum(-1) + log_det_L
            loss = nll.mean()
            # FoM = 1/sqrt(det(Sigma)) = 1/prod(diag(L))
            fom_norm = 1.0 / L.diagonal(dim1=-2, dim2=-1).prod(-1)
        else:
            sigma = L_or_sigma
            nll = sigma.log() + 0.5 * ((theta - mu) / sigma) ** 2
            loss = nll.mean()
            fom_norm = 1.0 / sigma.prod(dim=-1)

        # Un-normalize FoM to physical units
        if self.theta_std is not None:
            fom_phys = fom_norm / self.theta_std.prod()
        else:
            fom_phys = fom_norm

        return loss, fom_phys

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss, fom = self._nll_and_fom(x, theta)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_fom_median", fom.median(), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss, fom = self._nll_and_fom(x, theta)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_fom_median", fom.median(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    @torch.no_grad()
    def compress(self, spectra):
        """Compress normalized spectra to low-dim summaries.

        Parameters
        ----------
        spectra : np.ndarray, shape (N, input_dim)

        Returns
        -------
        summaries : np.ndarray, shape (N, summary_dim)
        """
        self.eval()
        x = torch.tensor(spectra, dtype=torch.float32, device=self.device)
        return self.compressor(x).cpu().numpy()

    @torch.no_grad()
    def predict_posterior(self, spectra):
        """Predict posterior mean and covariance from normalized spectra.

        Parameters
        ----------
        spectra : np.ndarray, shape (N, input_dim)

        Returns
        -------
        mu : np.ndarray, shape (N, theta_dim)
            Posterior mean (in normalized theta space).
        sigma : np.ndarray, shape (N, theta_dim)
            Posterior standard deviation (diagonal, in normalized space).
            For full_cov, returns diagonal of Sigma = L @ L^T.
        """
        self.eval()
        x = torch.tensor(spectra, dtype=torch.float32, device=self.device)
        t = self.compressor(x)
        mu, L_or_sigma = self._predict_posterior(t)

        if self.hparams.full_cov:
            # Sigma = L @ L^T, return diagonal
            Sigma = L_or_sigma @ L_or_sigma.transpose(-1, -2)
            sigma = Sigma.diagonal(dim1=-2, dim2=-1).sqrt()
        else:
            sigma = L_or_sigma

        return mu.cpu().numpy(), sigma.cpu().numpy()
