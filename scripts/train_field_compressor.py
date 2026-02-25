"""Train field-level compressor on convergence map tiles with MSE loss.

Usage:
    modal run scripts/train_field_compressor.py
    modal run scripts/train_field_compressor.py --lmax 200 --noise-level des_y3
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
    timeout=72000,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_compressor(
    lmax: int = 200,
    noise_level: str = "des_y3",
    max_epochs: int = 500,
    max_lr: float = 0.008,
    backbone: str = "efficientnet_v2_s",
    batch_size: int = 128,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 20,
):
    """Train field-level MSE compressor, save checkpoint and norm stats."""
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

    dm = TileDataModule(parquet_dir, batch_size=batch_size, fiducial_sim_id=109)
    dm.setup()

    n_train = len(dm.train_ds)
    n_val = len(dm.val_ds)
    n_test = len(dm.test_ds)
    n_fid = len(dm.fiducial_ds) if dm.fiducial_ds else 0

    print(f"Loaded {n_train + n_val + n_test + n_fid} tiles from {parquet_dir}")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test / {n_fid} fiducial")
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

    # Deterministic wandb ID from tag so re-runs resume the same run
    import hashlib
    wandb_id = hashlib.md5(tag.encode()).hexdigest()[:8]

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=tag,
        id=wandb_id, resume="allow",
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

    best_callback = ModelCheckpoint(
        dirpath=str(run_dir), filename="best",
        monitor="val_loss", mode="min",
    )
    last_callback = ModelCheckpoint(
        dirpath=str(run_dir), filename="last",
        every_n_epochs=1,
    )

    # --- Train (resume from last checkpoint if available) ---
    resume_ckpt = run_dir / "last.ckpt"
    if resume_ckpt.exists():
        print(f"Resuming from {resume_ckpt}")
    else:
        resume_ckpt = None

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[best_callback, last_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        accelerator="auto",
        precision="bf16-mixed",
    )
    trainer.fit(model, dm, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    vol.commit()

    # --- Evaluate MSE on all splits ---
    best_path = best_callback.best_model_path
    if best_path:
        model = FieldLevelCompressor.load_from_checkpoint(
            best_path, weights_only=False,
        )
    model = model.cpu().eval()

    splits = {}
    for split_name, ds in [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]:
        x, theta = ds.tensors()
        with torch.no_grad():
            pred = model(x)
            mse = torch.nn.functional.mse_loss(pred, theta)
        splits[split_name] = float(mse)
        print(f"{split_name:5s} MSE: {float(mse):.6f}")

    wandb_logger.experiment.summary.update({
        f"{s}/mse": v for s, v in splits.items()
    })

    # --- Save normalization stats ---
    norm_stats = {
        "tile_mean": dm.train_ds.tile_mean.tolist(),
        "tile_std": dm.train_ds.tile_std.tolist(),
        "theta_mean": dm.train_ds.theta_mean.tolist(),
        "theta_std": dm.train_ds.theta_std.tolist(),
    }
    with open(run_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # --- Save result ---
    result = dict(
        lmax=lmax, noise_level=noise_level, nside=nside,
        n_train=n_train, n_val=n_val, n_test=n_test, n_fiducial=n_fid,
        backbone=backbone,
        max_lr=max_lr, batch_size=batch_size, weight_decay=weight_decay,
        train_mse=splits["train"],
        val_mse=splits["val"],
        test_mse=splits["test"],
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
    decay_every_epochs: int = 20,
):
    result = train_compressor.remote(
        lmax=lmax, noise_level=noise_level,
        max_epochs=max_epochs, max_lr=max_lr,
        backbone=backbone,
        batch_size=batch_size, weight_decay=weight_decay,
        warmup_steps=warmup_steps, decay_rate=decay_rate,
        decay_every_epochs=decay_every_epochs,
    )
    print(f"\nTrain MSE: {result['train_mse']:.6f}  Val MSE: {result['val_mse']:.6f}  Test MSE: {result['test_mse']:.6f}")
    print(f"Trained {result['epochs_trained']} epochs in {result['elapsed_s']:.0f}s")
