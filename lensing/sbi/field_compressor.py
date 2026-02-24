"""Field-level VMIM compressor: EfficientNet backbone on convergence map tiles."""

import torch
import torch.nn as nn

import lightning as L

BACKBONE_REGISTRY = {
    "efficientnet_v2_s": ("efficientnet_v2_s", 1280),
    "efficientnet_b0": ("efficientnet_b0", 1280),
    "efficientnet_b2": ("efficientnet_b2", 1408),
}


def _build_backbone(name):
    """Lazy-import torchvision and build an EfficientNet feature extractor."""
    from torchvision.models import efficientnet_v2_s, efficientnet_b0, efficientnet_b2

    factory = {
        "efficientnet_v2_s": efficientnet_v2_s,
        "efficientnet_b0": efficientnet_b0,
        "efficientnet_b2": efficientnet_b2,
    }
    return factory[name](weights=None)


class FieldLevelCompressor(L.LightningModule):
    """Variational Mutual Information Maximization compressor for convergence map tiles.

    Uses an EfficientNet backbone for spatial feature extraction from 4-channel
    tomographic convergence maps, followed by the same VMIM head as VMIMCompressor.

    Parameters
    ----------
    nside : int
        Tile pixel size (nside x nside input images).
    n_bins : int
        Number of tomographic bins (input channels).
    summary_dim : int
        Dimensionality of the compressed summary.
    theta_dim : int
        Dimensionality of the cosmological parameter vector.
    backbone : str
        EfficientNet variant: "efficientnet_v2_s", "efficientnet_b0", or "efficientnet_b2".
    full_cov : bool
        If True, predict full Cholesky covariance; otherwise diagonal.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    theta_std : list or None
        Standard deviation of theta for un-normalizing FoM.
    """

    def __init__(
        self,
        nside=128,
        n_bins=4,
        summary_dim=2,
        theta_dim=2,
        backbone="efficientnet_v2_s",
        full_cov=False,
        lr=5e-4,
        weight_decay=1e-4,
        warmup_epochs=0,
        theta_noise_std=0.005,
        theta_std=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["theta_std"])
        if theta_std is not None:
            self.register_buffer("theta_std", torch.tensor(theta_std, dtype=torch.float32))
        else:
            self.theta_std = None

        # Build backbone
        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone {backbone!r}, choose from {list(BACKBONE_REGISTRY)}")
        _, backbone_dim = BACKBONE_REGISTRY[backbone]
        net = _build_backbone(backbone)

        # Replace first conv to accept n_bins channels instead of 3 RGB
        old_conv = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            n_bins,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.backbone = net.features  # drop classifier head

        # Pooling + compressor head
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.compressor_head = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.GELU(),
            nn.Linear(64, summary_dim),
        )

        # VMIM head — same structure as VMIMCompressor
        if full_cov:
            n_chol = theta_dim * (theta_dim + 1) // 2
            head_out = theta_dim + n_chol
        else:
            head_out = theta_dim * 2

        self.vmim_head = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, head_out),
        )

    def forward(self, x):
        """Compress tiles to summary vectors.

        Parameters
        ----------
        x : Tensor, shape (B, n_bins, nside, nside)

        Returns
        -------
        t : Tensor, shape (B, summary_dim)
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.compressor_head(pooled)

    def _predict_posterior(self, t):
        """From summary t, predict posterior mean and covariance factor.

        Returns
        -------
        mu : (B, theta_dim)
        L_or_sigma : (B, theta_dim, theta_dim) if full_cov else (B, theta_dim)
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
        t = self(x)
        mu, L_or_sigma = self._predict_posterior(t)

        if self.hparams.full_cov:
            L = L_or_sigma
            diff = theta - mu
            z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
            z = z.squeeze(-1)
            log_det_L = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            nll = 0.5 * (z ** 2).sum(-1) + log_det_L
            loss = nll.mean()
            fom_norm = 1.0 / L.diagonal(dim1=-2, dim2=-1).prod(-1)
        else:
            sigma = L_or_sigma
            nll = sigma.log() + 0.5 * ((theta - mu) / sigma) ** 2
            loss = nll.mean()
            fom_norm = 1.0 / sigma.prod(dim=-1)

        if self.theta_std is not None:
            fom_phys = fom_norm / self.theta_std.prod()
        else:
            fom_phys = fom_norm

        return loss, fom_phys

    def _augment(self, x):
        """Random augmentation for training: rotations, flips, and circular rolls.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)

        Returns
        -------
        x : Tensor, augmented
        """
        # Random 0/1/2/3 quarter-turns
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[2, 3])
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        # Random circular rolls along both spatial dims
        H, W = x.shape[2], x.shape[3]
        x = torch.roll(x, shifts=torch.randint(0, H, (1,)).item(), dims=2)
        x = torch.roll(x, shifts=torch.randint(0, W, (1,)).item(), dims=3)
        return x

    def training_step(self, batch, batch_idx):
        x, theta = batch
        x = self._augment(x)
        if self.hparams.theta_noise_std > 0:
            theta = theta + torch.randn_like(theta) * self.hparams.theta_noise_std
        loss, fom = self._nll_and_fom(x, theta)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_fom_median", fom.median(), prog_bar=False)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
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
    def compress(self, tiles):
        """Compress tile arrays to summaries.

        Parameters
        ----------
        tiles : np.ndarray, shape (N, n_bins, nside, nside)
            Already normalized tiles.

        Returns
        -------
        summaries : np.ndarray, shape (N, summary_dim)
        """
        self.eval()
        x = torch.tensor(tiles, dtype=torch.float32, device=self.device)
        return self(x).cpu().numpy()

    @torch.no_grad()
    def predict_posterior(self, tiles):
        """Predict posterior mean and std from normalized tiles.

        Parameters
        ----------
        tiles : np.ndarray, shape (N, n_bins, nside, nside)

        Returns
        -------
        mu : np.ndarray, shape (N, theta_dim)
        sigma : np.ndarray, shape (N, theta_dim)
        """
        self.eval()
        x = torch.tensor(tiles, dtype=torch.float32, device=self.device)
        t = self(x)
        mu, L_or_sigma = self._predict_posterior(t)

        if self.hparams.full_cov:
            Sigma = L_or_sigma @ L_or_sigma.transpose(-1, -2)
            sigma = Sigma.diagonal(dim1=-2, dim2=-1).sqrt()
        else:
            sigma = L_or_sigma

        return mu.cpu().numpy(), sigma.cpu().numpy()
