"""VMIM compressor: MLP that compresses spectra to low-dim summaries.

Two versions:
- VMIMCompressor: Original diagonal-Gaussian VMIM head.
- VMIMCompressorV2: Improved with LayerNorm, configurable architecture,
  and optional full-covariance (Cholesky) VMIM head with FoM logging.
"""

import numpy as np
import torch
import torch.nn as nn

import lightning as L


class VMIMCompressor(L.LightningModule):
    """Variational Mutual Information Maximization compressor (original).

    Compresses spectra into 2-dim summary statistics by maximizing mutual
    information with cosmological parameters via a diagonal Gaussian NLL loss.
    """

    def __init__(
        self,
        input_dim=200,
        summary_dim=2,
        lr=1e-3,
        weight_decay=1e-5,
        warmup_steps=100,
        total_steps=5000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, summary_dim),
        )

        self.vmim_head = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, summary_dim * 2),
        )

    def forward(self, x):
        return self.compressor(x)

    def _gaussian_nll(self, x, theta):
        t = self.compressor(x)
        out = self.vmim_head(t)
        d = self.hparams.summary_dim
        mu = out[:, :d]
        log_sigma = out[:, d:]
        nll = log_sigma + 0.5 * ((theta - mu) / (log_sigma.exp() + 1e-6)) ** 2
        return nll.mean()

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss = self._gaussian_nll(x, theta)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss = self._gaussian_nll(x, theta)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (step - self.hparams.warmup_steps) / max(
                1, self.hparams.total_steps - self.hparams.warmup_steps
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def compress(self, dataset):
        """Compress a full SpectraDataset, returning numpy arrays."""
        self.eval()
        x = torch.tensor(dataset.spectra, dtype=torch.float32, device=self.device)
        summaries = self.compressor(x).cpu().numpy()
        return summaries, dataset.theta_raw


class VMIMCompressorV2(L.LightningModule):
    """Improved VMIM compressor with LayerNorm and optional full-covariance head.

    Changes from V1:
    - Wider network (hidden_dim → 128 → 64 → summary_dim) with LayerNorm
    - Optional full-covariance VMIM head (Cholesky-parameterized)
    - FoM computed and logged at each step
    - Configurable summary_dim, hidden_dim, theta_dim

    Parameters
    ----------
    input_dim : int
        Dimension of input spectra.
    summary_dim : int
        Dimension of compressed summaries.
    theta_dim : int
        Dimension of parameter space (default 2 for Omega_m, S8).
    hidden_dim : int
        Width of first hidden layer.
    full_cov : bool
        If True, use Cholesky-parameterized full covariance in VMIM head.
        If False, use diagonal covariance (log_sigma).
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps for cosine annealing.
    theta_std : array-like, optional
        Standard deviation of theta used for un-normalizing FoM.
    """

    def __init__(
        self,
        input_dim=200,
        summary_dim=2,
        theta_dim=2,
        hidden_dim=256,
        full_cov=False,
        lr=5e-4,
        min_lr=1e-4,
        weight_decay=1e-4,
        warmup_steps=200,
        plateau_patience=10,
        plateau_factor=0.5,
        theta_std=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["theta_std"])
        if theta_std is not None:
            self.register_buffer("theta_std", torch.tensor(theta_std, dtype=torch.float32))
        else:
            self.theta_std = None

        # Compressor MLP with LayerNorm
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, summary_dim),
        )

        # VMIM head output size depends on covariance parameterization
        if full_cov:
            # mu (theta_dim) + Cholesky lower-triangle entries: theta_dim*(theta_dim+1)/2
            n_chol = theta_dim * (theta_dim + 1) // 2
            head_out = theta_dim + n_chol
        else:
            # mu (theta_dim) + log_sigma (theta_dim)
            head_out = theta_dim * 2

        self.vmim_head = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, head_out),
        )

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
                        L[:, i, j] = nn.functional.softplus(chol_raw[:, idx]) + 1e-6
                    else:
                        L[:, i, j] = chol_raw[:, idx]
                    idx += 1
            return mu, L
        else:
            mu = out[:, :d]
            log_sigma = out[:, d:]
            sigma = log_sigma.exp() + 1e-6
            return mu, sigma

    def _nll_and_fom(self, x, theta):
        """Compute Gaussian NLL and per-sample FoM."""
        t = self.compressor(x)
        mu, L_or_sigma = self._predict_posterior(t)
        d = self.hparams.theta_dim

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
            log_sigma = (sigma - 1e-6).clamp(min=1e-30).log()
            nll = log_sigma + 0.5 * ((theta - mu) / sigma) ** 2
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
        # Log current LR for schedule tracking
        opt = self.optimizers()
        if hasattr(opt, "param_groups"):
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss, fom = self._nll_and_fom(x, theta)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_fom_median", fom.median(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warmup via LambdaLR that saturates at 1.0 after warmup_steps,
        # combined with ReduceLROnPlateau for adaptive decay.
        # We use LambdaLR for warmup at step level, then let
        # ReduceLROnPlateau handle the rest at epoch level.
        #
        # Strategy: warmup is fast (a few hundred steps), so we run it as a
        # step-level scheduler. Once warmup is done (lr_lambda=1.0), the
        # plateau scheduler takes over by monitoring val_loss each epoch.
        self._warmup_done = False

        def warmup_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            return 1.0

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
            min_lr=self.hparams.min_lr,
        )

        return (
            [optimizer],
            [
                {"scheduler": warmup_sched, "interval": "step"},
                {"scheduler": plateau_sched, "interval": "epoch", "monitor": "val_fom_median"},
            ],
        )

    @torch.no_grad()
    def compress(self, dataset):
        """Compress a full SpectraDataset, returning numpy arrays."""
        self.eval()
        x = torch.tensor(dataset.spectra, dtype=torch.float32, device=self.device)
        summaries = self.compressor(x).cpu().numpy()
        return summaries, dataset.theta_raw

    @torch.no_grad()
    def compress_arrays(self, spectra):
        """Compress raw spectra arrays (already normalized).

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
