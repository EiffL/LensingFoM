"""VMIM compressor: MLP that compresses 200-dim spectra to 2-dim summaries."""

import numpy as np
import torch
import torch.nn as nn

import lightning as L


class VMIMCompressor(L.LightningModule):
    """Variational Mutual Information Maximization compressor.

    Compresses 200-dim power spectra into 2-dim summary statistics by
    maximizing mutual information with cosmological parameters (Omega_m, S8)
    via a Gaussian NLL loss.

    Parameters
    ----------
    input_dim : int
        Dimension of input spectra (default 200).
    summary_dim : int
        Dimension of compressed summaries (default 2).
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps for cosine annealing.
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

        # Compressor MLP: 200 → 128 → 64 → 2
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, summary_dim),
        )

        # VMIM head: 2 → 64 → 4 (mu_1, mu_2, log_sigma_1, log_sigma_2)
        self.vmim_head = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, summary_dim * 2),
        )

    def forward(self, x):
        """Compress spectra to summaries."""
        return self.compressor(x)

    def _gaussian_nll(self, x, theta):
        """Gaussian NLL loss averaged over batch and parameters."""
        t = self.compressor(x)
        out = self.vmim_head(t)

        d = self.hparams.summary_dim
        mu = out[:, :d]
        log_sigma = out[:, d:]

        # NLL = log(sigma) + 0.5 * ((theta - mu) / sigma)^2  (+ const)
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
        """Compress a full SpectraDataset, returning numpy arrays.

        Parameters
        ----------
        dataset : SpectraDataset
            Dataset with .spectra and .theta_raw attributes.

        Returns
        -------
        summaries : np.ndarray, shape (N, 2)
            Compressed summary statistics.
        theta_raw : np.ndarray, shape (N, 2)
            Unnormalized cosmological parameters.
        """
        self.eval()
        x = torch.tensor(dataset.spectra, dtype=torch.float32, device=self.device)
        summaries = self.compressor(x).cpu().numpy()
        return summaries, dataset.theta_raw
