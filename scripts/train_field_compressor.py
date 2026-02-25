"""Train field-level compressor on convergence map tiles with MSE loss.

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
    gpu="H100",
    timeout=7200,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 500,
    max_lr: float = 0.008,
    backbone: str = "efficientnet_v2_s",
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 10,
):
    """Train field-level MSE compressor, log loss to wandb."""
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
    tag = f"field_mse_lmax{lmax}_{noise_level}_{backbone}"
    run_dir = Path(RESULTS_DIR) / "field_compressor_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model = FieldLevelCompressor(
        nside=nside,
        n_bins=4,
        theta_dim=2,
        backbone=backbone,
        max_lr=max_lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        decay_rate=decay_rate,
        decay_every_epochs=decay_every_epochs,
    )

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, nside=nside,
            n_train=n_train, n_val=n_val, n_test=n_test,
            backbone=backbone, loss="mse",
            max_lr=max_lr, batch_size=batch_size, max_epochs=max_epochs,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            decay_rate=decay_rate, decay_every_epochs=decay_every_epochs,
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
            best_path, weights_only=False,
        )
    model = model.cpu().eval()

    # --- MSE on all splits ---
    splits = {}
    compressed = {}
    for split_name, ds in [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]:
        x, theta = ds.tensors()
        with torch.no_grad():
            pred = model(x)
            mse = torch.nn.functional.mse_loss(pred, theta)
        splits[split_name] = {"mse": float(mse)}
        compressed[split_name] = (pred.numpy(), theta.numpy())
        print(f"{split_name:5s} MSE: {float(mse):.6f}")

    # --- Train Gaussian NPE on val set, evaluate FoM ---
    from lensing.sbi.npe import train_npe, compute_fom

    print("\nTraining Gaussian NPE on val compressed summaries...")
    val_pred, val_theta = compressed["val"]
    test_pred, test_theta = compressed["test"]
    train_pred, train_theta = compressed["train"]

    npe = train_npe(
        val_pred, val_theta,
        val_summaries=test_pred, val_theta=test_theta,
        max_epochs=500, patience=30,
    )
    npe = npe.cpu().eval()

    # Un-normalize FoM: NPE works in normalized theta space,
    # scale by 1/prod(theta_std) to get physical FoM
    theta_std_prod = float(dm.train_ds.theta_std.prod())

    for split_name, (pred, theta) in compressed.items():
        fom_median, fom_lo, fom_hi, fom_all = compute_fom(npe, pred)
        # Convert to physical units
        fom_median /= theta_std_prod
        fom_lo /= theta_std_prod
        fom_hi /= theta_std_prod
        splits[split_name]["fom"] = fom_median
        splits[split_name]["fom_lo"] = fom_lo
        splits[split_name]["fom_hi"] = fom_hi
        print(f"{split_name:5s} FoM: {fom_median:.1f} [{fom_lo:.1f}, {fom_hi:.1f}]")

    wandb_logger.experiment.summary.update({
        f"{s}/mse": v["mse"] for s, v in splits.items()
    })
    wandb_logger.experiment.summary.update({
        f"{s}/fom": v["fom"] for s, v in splits.items()
    })

    result = dict(
        lmax=lmax, noise_level=noise_level, nside=nside,
        n_train=n_train, n_val=n_val, n_test=n_test,
        backbone=backbone,
        max_lr=max_lr, batch_size=batch_size, weight_decay=weight_decay,
        train_mse=splits["train"]["mse"],
        val_mse=splits["val"]["mse"],
        test_mse=splits["test"]["mse"],
        train_fom=splits["train"]["fom"],
        val_fom=splits["val"]["fom"],
        test_fom=splits["test"]["fom"],
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
    max_epochs: int = 500,
    max_lr: float = 0.008,
    backbone: str = "efficientnet_v2_s",
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 10,
):
    result = train_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level,
        max_epochs=max_epochs, max_lr=max_lr,
        backbone=backbone,
        batch_size=batch_size, weight_decay=weight_decay,
        warmup_steps=warmup_steps, decay_rate=decay_rate,
        decay_every_epochs=decay_every_epochs,
    )
    print(f"\nTrain MSE: {result['train_mse']:.6f}  Val MSE: {result['val_mse']:.6f}  Test MSE: {result['test_mse']:.6f}")
    print(f"Train FoM: {result['train_fom']:.1f}  Val FoM: {result['val_fom']:.1f}  Test FoM: {result['test_fom']:.1f}")
    print(f"Trained {result['epochs_trained']} epochs in {result['elapsed_s']:.0f}s")
