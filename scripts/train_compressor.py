"""Train VMIM compressor on real power spectra from Gower Street simulations.

Usage:
    modal run scripts/train_compressor.py
    modal run scripts/train_compressor.py --lmax 400 --noise-level des_y3
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow", "datasets",
        "torch", "lightning", "wandb",
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-train-compressor", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="A100",
    timeout=3600,
    memory=16384,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 1000,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    summary_dim: int = 2,
    full_cov: bool = False,
    batch_size: int = 256,
):
    """Train VMIM compressor, log loss and FoM to wandb."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import torch
    import wandb
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from lensing.sbi.compressor import VMIMCompressor
    from lensing.sbi.dataset import SpectraDataModule

    t0 = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # --- Load data ---
    parquet_path = Path(RESULTS_DIR) / "spectra_dataset" / f"spectra_lmax{lmax}_noise_{noise_level}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run pipeline.py --stage 5 first."
        )

    dm = SpectraDataModule(parquet_path, batch_size=batch_size)
    dm.setup()

    n_train = len(dm.train_ds)
    n_val = len(dm.val_ds)
    n_test = len(dm.test_ds)
    input_dim = dm.train_ds.spectra.shape[1]

    print(f"Loaded {n_train + n_val + n_test} tiles from {parquet_path.name}")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    print(f"Input dim: {input_dim}")

    # --- Build model ---
    tag = f"vmim_lmax{lmax}_{noise_level}"
    if full_cov:
        tag += "_fullcov"
    run_dir = Path(RESULTS_DIR) / "compressor_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model = VMIMCompressor(
        input_dim=input_dim,
        summary_dim=summary_dim,
        theta_dim=2,
        hidden_dim=hidden_dim,
        full_cov=full_cov,
        lr=lr,
        weight_decay=1e-4,
        theta_std=dm.train_ds.theta_std,
    )

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, input_dim=input_dim,
            n_train=n_train, n_val=n_val, n_test=n_test,
            summary_dim=summary_dim, hidden_dim=hidden_dim, full_cov=full_cov,
            lr=lr, batch_size=batch_size, max_epochs=max_epochs,
        ),
        save_dir=str(run_dir),
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=str(run_dir), filename="best",
        monitor="val_loss", mode="min",
    )

    # --- Train ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        accelerator="auto",
    )
    trainer.fit(model, dm)

    # --- Evaluate on test set ---
    best_path = ckpt_callback.best_model_path
    if best_path:
        model = VMIMCompressor.load_from_checkpoint(
            best_path, theta_std=dm.train_ds.theta_std, weights_only=False,
        )
    model = model.cpu().eval()

    x_test, theta_test = dm.test_ds.tensors()
    with torch.no_grad():
        test_loss, test_foms = model._nll_and_fom(x_test, theta_test)
    test_fom = float(test_foms.median())
    print(f"\nTest loss: {float(test_loss):.4f}")
    print(f"Test FoM:  {test_fom:.1f}")

    wandb_logger.experiment.summary.update({
        "test/loss": float(test_loss),
        "test/fom": test_fom,
    })

    result = dict(
        lmax=lmax, noise_level=noise_level, input_dim=input_dim,
        n_train=n_train, n_val=n_val, n_test=n_test,
        summary_dim=summary_dim, hidden_dim=hidden_dim, full_cov=full_cov,
        lr=lr, batch_size=batch_size,
        test_loss=float(test_loss), test_fom=test_fom,
        epochs_trained=trainer.current_epoch,
        elapsed_s=time.time() - t0,
    )
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    wandb.finish()

    print(f"Total time: {time.time()-t0:.0f}s")
    return result


@app.local_entrypoint()
def main(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 1000,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    summary_dim: int = 2,
    full_cov: bool = False,
    batch_size: int = 256,
):
    result = train_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level,
        max_epochs=max_epochs, lr=lr, hidden_dim=hidden_dim,
        summary_dim=summary_dim, full_cov=full_cov,
        batch_size=batch_size,
    )
    print(f"\nTest FoM: {result['test_fom']:.1f}")
    print(f"Trained {result['epochs_trained']} epochs in {result['elapsed_s']:.0f}s")
