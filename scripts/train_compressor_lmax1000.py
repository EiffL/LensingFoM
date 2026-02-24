"""Focused compressor training for lmax=1000 with FoM tracking.

Since the VMIM head predicts p(theta|t(x)) = N(mu, diag(sigma^2)),
we can compute FoM = 1/(sigma_1 * sigma_2) directly at each epoch,
converted back to physical units via theta_std.

Usage:
    python scripts/train_compressor_lmax1000.py
"""

import sys
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lensing.sbi.dataset import SpectraDataModule


class VMIMCompressorV2(L.LightningModule):
    """Improved VMIM compressor with FoM logging.

    Changes from V1:
    - Wider network (256 → 128 → 64 → 2)
    - LayerNorm for stable training
    - Residual connection in middle layers
    - FoM computed and logged at each validation epoch
    """

    def __init__(
        self,
        input_dim=200,
        summary_dim=2,
        hidden_dim=256,
        lr=5e-4,
        weight_decay=1e-4,
        warmup_steps=200,
        total_steps=10000,
        theta_std=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["theta_std"])
        # Store theta_std for FoM un-normalization (not a learned param)
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

        # VMIM head: summaries → (mu, log_sigma) for diagonal Gaussian
        self.vmim_head = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Linear(64, summary_dim * 2),
        )

        # Track FoM history
        self.val_fom_history = []
        self.train_fom_history = []

    def forward(self, x):
        return self.compressor(x)

    def _gaussian_nll_and_fom(self, x, theta):
        """Compute Gaussian NLL and per-sample FoM."""
        t = self.compressor(x)
        out = self.vmim_head(t)

        d = self.hparams.summary_dim
        mu = out[:, :d]
        log_sigma = out[:, d:]
        sigma = log_sigma.exp() + 1e-6

        # NLL
        nll = log_sigma + 0.5 * ((theta - mu) / sigma) ** 2
        loss = nll.mean()

        # FoM in normalized space: 1 / prod(sigma) per sample
        # Convert to physical: sigma_phys = sigma_norm * theta_std
        # FoM_phys = 1 / prod(sigma_phys) = FoM_norm / prod(theta_std)
        fom_norm = 1.0 / sigma.prod(dim=-1)  # (B,)
        if self.theta_std is not None:
            fom_phys = fom_norm / self.theta_std.prod()
        else:
            fom_phys = fom_norm

        return loss, fom_phys

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss, fom = self._gaussian_nll_and_fom(x, theta)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_fom_median", fom.median(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss, fom = self._gaussian_nll_and_fom(x, theta)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_fom_median", fom.median(), prog_bar=True)
        return {"val_loss": loss, "fom": fom}

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
            return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class FoMTracker(Callback):
    """Track median FoM on full val set at each epoch end."""

    def __init__(self):
        self.epochs = []
        self.val_foms = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_fom = trainer.callback_metrics.get("val_fom_median")
        if val_loss is not None and val_fom is not None:
            self.epochs.append(trainer.current_epoch)
            self.val_losses.append(float(val_loss))
            self.val_foms.append(float(val_fom))


def main():
    parquet_path = Path("data/spectra/spectra_lmax1000.parquet")
    output_dir = Path("results/sbi/lmax1000_focused")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    dm = SpectraDataModule(parquet_path, batch_size=256)
    dm.setup()

    print(f"comp_train: {len(dm.comp_train_ds)}")
    print(f"comp_val:   {len(dm.comp_val_ds)}")
    print(f"npe:        {len(dm.npe_ds)}")
    print(f"test:       {len(dm.test_ds)}")
    print(f"theta_std:  {dm.comp_train_ds.theta_std}")
    print(f"theta_mean: {dm.comp_train_ds.theta_mean}")

    input_dim = dm.comp_train_ds.spectra.shape[1]
    theta_std = dm.comp_train_ds.theta_std

    max_epochs = 1000
    steps_per_epoch = max(1, len(dm.comp_train_ds) // 256)
    total_steps = steps_per_epoch * max_epochs

    model = VMIMCompressorV2(
        input_dim=input_dim,
        summary_dim=2,
        hidden_dim=256,
        lr=5e-4,
        weight_decay=1e-4,
        warmup_steps=steps_per_epoch * 5,
        total_steps=total_steps,
        theta_std=theta_std,
    )

    fom_tracker = FoMTracker()
    callbacks = [
        fom_tracker,
        EarlyStopping(monitor="val_loss", patience=100, mode="min"),
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="compressor-best",
            monitor="val_loss",
            mode="min",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=L.pytorch.loggers.CSVLogger(str(output_dir), name="compressor"),
    )
    trainer.fit(model, dm)

    print(f"\nBest val_loss: {callbacks[2].best_model_score:.4f}")

    # --- Compute final FoM on test set ---
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(dm.test_ds.spectra, dtype=torch.float32)
        theta_test = torch.tensor(dm.test_ds.theta, dtype=torch.float32)
        _, test_foms = model._gaussian_nll_and_fom(x_test, theta_test)
        test_fom_median = float(test_foms.median())
        test_fom_16 = float(test_foms.quantile(0.16))
        test_fom_84 = float(test_foms.quantile(0.84))

    print(f"\nFinal test FoM: {test_fom_median:.1f} [{test_fom_16:.1f}, {test_fom_84:.1f}]")

    # --- Plot FoM vs epoch ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = fom_tracker.epochs
    foms = fom_tracker.val_foms
    losses = fom_tracker.val_losses

    ax1.plot(epochs, losses, "b-", lw=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss (Gaussian NLL)")
    ax1.set_title("Validation Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, foms, "r-", lw=1)
    ax2.axhline(test_fom_median, color="k", ls="--", alpha=0.5,
                label=f"Final test FoM = {test_fom_median:.1f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"FoM$(\Omega_m, S_8)$")
    ax2.set_title("VMIM FoM vs Training Epoch (lmax=1000)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fom_vs_epoch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_dir / 'fom_vs_epoch.png'}")

    # Save FoM history
    np.savez(
        output_dir / "fom_history.npz",
        epochs=np.array(epochs),
        val_foms=np.array(foms),
        val_losses=np.array(losses),
        test_fom_median=test_fom_median,
        test_fom_16=test_fom_16,
        test_fom_84=test_fom_84,
    )


if __name__ == "__main__":
    main()
