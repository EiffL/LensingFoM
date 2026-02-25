"""Field-level compressor: EfficientNet backbone on convergence map tiles."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """MSE regression compressor for convergence map tiles.

    Uses an EfficientNet backbone to predict cosmological parameters
    (Omega_m, S8) directly from 4-channel tomographic convergence maps.
    """

    def __init__(
        self,
        nside=128,
        n_bins=4,
        theta_dim=2,
        backbone="efficientnet_v2_s",
        max_lr=0.008,
        weight_decay=1e-5,
        warmup_steps=500,
        decay_rate=0.85,
        decay_every_epochs=10,
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

        # Pooling + regression head
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.GELU(),
            nn.Linear(128, theta_dim),
        )

    def forward(self, x):
        """Predict theta from tiles.

        Parameters
        ----------
        x : Tensor, shape (B, n_bins, nside, nside)

        Returns
        -------
        pred : Tensor, shape (B, theta_dim)
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.head(pooled)

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
        pred = self(x)
        loss = F.mse_loss(pred, theta)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        pred = self(x)
        loss = F.mse_loss(pred, theta)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = self.hparams.warmup_steps
        steps_per_epoch = total_steps // self.trainer.max_epochs
        step_size = self.hparams.decay_every_epochs * steps_per_epoch

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size,
            gamma=self.hparams.decay_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
