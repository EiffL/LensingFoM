"""Train Gaussian NPE on frozen compressor outputs, evaluate FoM on fiducial sim.

Usage:
    modal run scripts/train_field_npe.py
    modal run scripts/train_field_npe.py --lmax 200 --noise-level des_y3
    modal run scripts/train_field_npe.py --compressor-tag field_mse_lmax200_des_y3_efficientnet_v2_s
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

app = modal.App("lensing-train-field-npe", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="A10G",
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_npe_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    backbone: str = "efficientnet_v2_s",
    npe_max_epochs: int = 500,
    npe_patience: int = 30,
    compressor_tag: str = "",
):
    """Load frozen compressor, train NPE, evaluate FoM on fiducial."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import numpy as np
    import torch
    import wandb
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader

    from lensing.sbi.field_compressor import FieldLevelCompressor
    from lensing.sbi.npe import GaussianNPE, compute_fom
    from lensing.sbi.tile_dataset import (
        LMAX_TO_NSIDE,
        CompressedTileDataset,
        TileDataModule,
    )

    t0 = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # --- Resolve compressor checkpoint ---
    if not compressor_tag:
        compressor_tag = f"field_mse_lmax{lmax}_{noise_level}_{backbone}"
    compressor_dir = Path(RESULTS_DIR) / "field_compressor_runs" / compressor_tag
    ckpt_path = compressor_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run train_field_compressor.py first."
        )

    # --- Load frozen compressor ---
    compressor = FieldLevelCompressor.load_from_checkpoint(
        str(ckpt_path), weights_only=False,
    )
    compressor = compressor.eval()
    compressor.requires_grad_(False)
    if torch.cuda.is_available():
        compressor = compressor.cuda()
    print(f"Loaded compressor from {ckpt_path}")

    # --- Load data (same split as compressor training) ---
    nside = LMAX_TO_NSIDE[lmax]
    parquet_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"

    dm = TileDataModule(parquet_dir, batch_size=128, fiducial_sim_id=109)
    dm.setup()

    print(f"Train: {len(dm.train_ds)}, Val: {len(dm.val_ds)}, "
          f"Test: {len(dm.test_ds)}, Fiducial: {len(dm.fiducial_ds)}")

    # --- Build compressed datasets ---
    # NPE trains on val split (with augmentation), validates on test split (no augmentation)
    npe_train_ds = CompressedTileDataset(dm.val_ds, compressor, augment=True)
    npe_val_ds = CompressedTileDataset(dm.test_ds, compressor, augment=False)

    train_dl = DataLoader(npe_train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_dl = DataLoader(npe_val_ds, batch_size=256, num_workers=0)

    # --- Train NPE ---
    npe_tag = f"npe_{compressor_tag}"
    run_dir = Path(RESULTS_DIR) / "field_npe_runs" / npe_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    npe = GaussianNPE(input_dim=2, theta_dim=2)

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=npe_tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, backbone=backbone,
            compressor_tag=compressor_tag,
            npe_max_epochs=npe_max_epochs, npe_patience=npe_patience,
        ),
        save_dir=str(run_dir),
    )

    trainer = L.Trainer(
        max_epochs=npe_max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=npe_patience, mode="min")],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(npe, train_dl, val_dl)
    npe = npe.cpu().eval()

    # --- FoM on fiducial tiles ---
    fiducial_ds = CompressedTileDataset(dm.fiducial_ds, compressor.cpu(), augment=False)
    fid_summaries = torch.stack([fiducial_ds[i][0] for i in range(len(fiducial_ds))])
    fid_summaries_np = fid_summaries.numpy()

    fid_foms = npe.predict_fom(fid_summaries_np)

    # Un-normalize to physical units
    theta_std_prod = float(np.prod(dm.train_ds.theta_std))
    fid_foms_phys = fid_foms / theta_std_prod

    fom_median = float(np.median(fid_foms_phys))
    fom_std = float(np.std(fid_foms_phys))
    print(f"\nFiducial FoM: {fom_median:.1f} +/- {fom_std:.1f}")
    print(f"  All 12: {[f'{f:.1f}' for f in fid_foms_phys]}")

    # --- FoM on test set (sanity check) ---
    test_ds_compressed = CompressedTileDataset(dm.test_ds, compressor.cpu(), augment=False)
    test_summaries = torch.stack([test_ds_compressed[i][0] for i in range(len(test_ds_compressed))])
    test_fom_median, test_fom_lo, test_fom_hi, _ = compute_fom(npe, test_summaries.numpy())
    test_fom_median /= theta_std_prod
    test_fom_lo /= theta_std_prod
    test_fom_hi /= theta_std_prod
    print(f"Test FoM: {test_fom_median:.1f} [{test_fom_lo:.1f}, {test_fom_hi:.1f}]")

    # --- Save results ---
    wandb_logger.experiment.summary.update({
        "fom_fiducial_median": fom_median,
        "fom_fiducial_std": fom_std,
        "fom_test_median": test_fom_median,
    })

    result = dict(
        lmax=lmax, noise_level=noise_level, backbone=backbone,
        compressor_tag=compressor_tag,
        fom_fiducial_median=fom_median,
        fom_fiducial_std=fom_std,
        fom_fiducial_all=[float(f) for f in fid_foms_phys],
        fom_test_median=test_fom_median,
        fom_test_lo=test_fom_lo,
        fom_test_hi=test_fom_hi,
        npe_epochs=trainer.current_epoch,
        elapsed_s=time.time() - t0,
    )
    with open(run_dir / "fom_result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    wandb.finish()

    print(f"Total time: {time.time()-t0:.0f}s")
    return result


@app.local_entrypoint()
def main(
    lmax: int = 200,
    noise_level: str = "des_y3",
    backbone: str = "efficientnet_v2_s",
    npe_max_epochs: int = 500,
    npe_patience: int = 30,
    compressor_tag: str = "",
):
    result = train_npe_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level, backbone=backbone,
        npe_max_epochs=npe_max_epochs, npe_patience=npe_patience,
        compressor_tag=compressor_tag,
    )
    print(f"\nFiducial FoM: {result['fom_fiducial_median']:.1f} +/- {result['fom_fiducial_std']:.1f}")
    print(f"Test FoM: {result['fom_test_median']:.1f} [{result['fom_test_lo']:.1f}, {result['fom_test_hi']:.1f}]")
    print(f"NPE trained {result['npe_epochs']} epochs in {result['elapsed_s']:.0f}s")
