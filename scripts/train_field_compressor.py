"""Train field-level VMIM compressor on convergence map tiles.

Usage:
    modal run scripts/train_field_compressor.py
    modal run scripts/train_field_compressor.py --lmax 200 --noise-level des_y3
    modal run scripts/train_field_compressor.py --backbone efficientnet_b0 --batch-size 128
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow", "datasets",
        "torch", "torchvision", "lightning", "wandb",
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-train-field-compressor", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="A100-80GB",
    timeout=7200,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 200,
    lr: float = 5e-4,
    summary_dim: int = 2,
    backbone: str = "efficientnet_v2_s",
    full_cov: bool = False,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
):
    """Train field-level VMIM compressor, log loss and FoM to wandb."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import torch
    import wandb
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from lensing.sbi.field_compressor import FieldLevelCompressor
    from lensing.sbi.tile_dataset import LMAX_TO_NSIDE, TileDataModule

    t0 = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # --- Load data ---
    nside = LMAX_TO_NSIDE[lmax]
    parquet_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"
    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"{parquet_dir} not found. Run pipeline.py --stage 3 first."
        )

    dm = TileDataModule(parquet_dir, batch_size=batch_size)
    dm.setup()

    n_train = len(dm.train_ds)
    n_val = len(dm.val_ds)
    n_test = len(dm.test_ds)

    print(f"Loaded {n_train + n_val + n_test} tiles from {parquet_dir}")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    print(f"Tile shape: (4, {nside}, {nside})")

    # --- Build model ---
    tag = f"field_lmax{lmax}_{noise_level}_{backbone}_s{summary_dim}"
    run_dir = Path(RESULTS_DIR) / "field_compressor_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model = FieldLevelCompressor(
        nside=nside,
        n_bins=4,
        summary_dim=summary_dim,
        theta_dim=2,
        backbone=backbone,
        full_cov=full_cov,
        lr=lr,
        weight_decay=weight_decay,
        theta_std=dm.train_ds.theta_std,
    )

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, nside=nside,
            n_train=n_train, n_val=n_val, n_test=n_test,
            summary_dim=summary_dim, backbone=backbone, full_cov=full_cov,
            lr=lr, batch_size=batch_size, max_epochs=max_epochs,
            weight_decay=weight_decay,
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
        precision="bf16-mixed",
    )
    trainer.fit(model, dm)

    # --- Evaluate on all splits ---
    best_path = ckpt_callback.best_model_path
    if best_path:
        model = FieldLevelCompressor.load_from_checkpoint(
            best_path, theta_std=dm.train_ds.theta_std, weights_only=False,
        )
    model = model.cpu().eval()

    splits = {}
    for split_name, ds in [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]:
        x, theta = ds.tensors()
        with torch.no_grad():
            loss, foms = model._nll_and_fom(x, theta)
        splits[split_name] = {"loss": float(loss), "fom": float(foms.median())}
        print(f"{split_name:5s} loss: {float(loss):.4f}  FoM: {float(foms.median()):.1f}")

    wandb_logger.experiment.summary.update({
        f"{s}/loss": v["loss"] for s, v in splits.items()
    })
    wandb_logger.experiment.summary.update({
        f"{s}/fom": v["fom"] for s, v in splits.items()
    })

    result = dict(
        lmax=lmax, noise_level=noise_level, nside=nside,
        n_train=n_train, n_val=n_val, n_test=n_test,
        summary_dim=summary_dim, backbone=backbone, full_cov=full_cov,
        lr=lr, batch_size=batch_size, weight_decay=weight_decay,
        train_loss=splits["train"]["loss"], train_fom=splits["train"]["fom"],
        val_loss=splits["val"]["loss"], val_fom=splits["val"]["fom"],
        test_loss=splits["test"]["loss"], test_fom=splits["test"]["fom"],
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
    lr: float = 5e-4,
    summary_dim: int = 2,
    backbone: str = "efficientnet_v2_s",
    full_cov: bool = False,
    batch_size: int = 128,
    weight_decay: float = 1e-4,
):
    result = train_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level,
        max_epochs=max_epochs, lr=lr, summary_dim=summary_dim,
        backbone=backbone, full_cov=full_cov,
        batch_size=batch_size, weight_decay=weight_decay,
    )
    print(f"\nTrain FoM: {result['train_fom']:.1f}  Val FoM: {result['val_fom']:.1f}  Test FoM: {result['test_fom']:.1f}")
    print(f"Trained {result['epochs_trained']} epochs in {result['elapsed_s']:.0f}s")
