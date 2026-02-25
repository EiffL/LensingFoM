"""Train Gaussian NPE on frozen compressor outputs, evaluate FoM on fiducial sim.

Pre-compresses all tiles in batches through the frozen compressor (with
augmentation for training data), then trains the lightweight NPE on the
resulting 2D summaries. Much faster than per-sample compression.

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
        "matplotlib", "scipy",
    )
    .add_local_python_source("lensing")
)

app = modal.App("lensing-train-field-npe", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: vol},
    gpu="A100-40GB",
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_npe_and_evaluate(
    lmax: int = 200,
    noise_level: str = "des_y3",
    backbone: str = "efficientnet_v2_s",
    npe_lr: float = 2e-4,
    npe_max_steps: int = 20000,
    npe_patience: int = 100,
    n_augmentations: int = 10,
    compressor_tag: str = "",
):
    """Load frozen compressor, batch-compress tiles, train NPE, evaluate FoM."""
    import json
    import time
    from pathlib import Path

    import lightning as L
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import wandb
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
    from scipy import stats
    from torch.utils.data import DataLoader, TensorDataset

    from lensing.sbi.field_compressor import FieldLevelCompressor
    from lensing.sbi.npe import GaussianNPE, compute_fom
    from lensing.sbi.tile_dataset import TileDataModule, batch_compress

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
    parquet_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"

    dm = TileDataModule(parquet_dir, batch_size=128, fiducial_sim_id=109)
    dm.setup()

    print(f"Train: {len(dm.train_ds)}, Val: {len(dm.val_ds)}, "
          f"Test: {len(dm.test_ds)}, Fiducial: {len(dm.fiducial_ds)}")

    # Un-normalization factor for FoM
    theta_std_prod = float(np.prod(dm.train_ds.theta_std))

    # --- Batch-compress all datasets ---
    # NPE trains on val split (with augmentation), validates on test split
    t_compress = time.time()

    train_x, train_y = batch_compress(
        dm.val_ds, compressor, batch_size=256,
        augment=True, n_augmentations=n_augmentations,
    )
    val_x, val_y = batch_compress(
        dm.test_ds, compressor, batch_size=256,
        augment=False, n_augmentations=1,
    )

    print(f"Compressed {len(train_x)} train + {len(val_x)} val summaries "
          f"in {time.time()-t_compress:.1f}s")

    train_dl = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=256, shuffle=True, pin_memory=True,
    )
    val_dl = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=256, pin_memory=True,
    )

    # --- Train NPE ---
    npe_tag = f"npe_{compressor_tag}"
    run_dir = Path(RESULTS_DIR) / "field_npe_runs" / npe_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    npe = GaussianNPE(
        input_dim=train_x.shape[1], theta_dim=2, lr=npe_lr,
        decay_start=10000, decay_every=2000, decay_factor=0.5,
        theta_std_prod=theta_std_prod,
    )

    wandb_logger = WandbLogger(
        project="LensingFoM", entity="eiffl", name=npe_tag,
        config=dict(
            lmax=lmax, noise_level=noise_level, backbone=backbone,
            compressor_tag=compressor_tag,
            npe_lr=npe_lr, npe_max_steps=npe_max_steps,
            npe_patience=npe_patience,
            n_augmentations=n_augmentations,
            n_train_compressed=len(train_x),
            n_val_compressed=len(val_x),
        ),
        save_dir=str(run_dir),
    )

    trainer = L.Trainer(
        max_steps=npe_max_steps,
        callbacks=[EarlyStopping(monitor="val_loss", patience=npe_patience, mode="min")],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )
    trainer.fit(npe, train_dl, val_dl)
    npe = npe.cpu().eval()

    # --- FoM on fiducial tiles ---
    fid_x, _ = batch_compress(
        dm.fiducial_ds, compressor, batch_size=256,
        augment=False, n_augmentations=1,
    )
    fid_summaries_np = fid_x.numpy()

    fid_foms = npe.predict_fom(fid_summaries_np)
    fid_foms_phys = fid_foms / theta_std_prod

    fom_median = float(np.median(fid_foms_phys))
    fom_std = float(np.std(fid_foms_phys))
    print(f"\nFiducial FoM: {fom_median:.1f} +/- {fom_std:.1f}")
    print(f"  All {len(fid_foms_phys)}: {[f'{f:.1f}' for f in fid_foms_phys]}")

    # --- FoM on test set (sanity check) ---
    test_fom_median, test_fom_lo, test_fom_hi, _ = compute_fom(npe, val_x.numpy())
    test_fom_median /= theta_std_prod
    test_fom_lo /= theta_std_prod
    test_fom_hi /= theta_std_prod
    print(f"Test FoM: {test_fom_median:.1f} [{test_fom_lo:.1f}, {test_fom_hi:.1f}]")

    # --- Z-score calibration plot on test set ---
    z = npe.compute_zscores(val_x.numpy(), val_y.numpy())
    param_names = [r"$\Omega_m$", r"$S_8$"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x_grid = np.linspace(-4, 4, 200)
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(z[:, i], bins=30, density=True, alpha=0.7, label="NPE z-scores")
        ax.plot(x_grid, stats.norm.pdf(x_grid), "k--", lw=1.5, label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel("z-score")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}  (mean={z[:, i].mean():.2f}, std={z[:, i].std():.2f})")
        ax.legend()
        ax.set_xlim(-4, 4)
    fig.suptitle(f"NPE Calibration — lmax={lmax} {noise_level}")
    fig.tight_layout()

    zscore_path = run_dir / "zscore_calibration.png"
    fig.savefig(zscore_path, dpi=150)
    plt.close(fig)
    wandb.log({"zscore_calibration": wandb.Image(str(zscore_path))})

    # --- Save results ---
    wandb_logger.experiment.summary.update({
        "fom_fiducial_median": fom_median,
        "fom_fiducial_std": fom_std,
        "fom_test_median": test_fom_median,
        "zscore_omega_m_mean": float(z[:, 0].mean()),
        "zscore_omega_m_std": float(z[:, 0].std()),
        "zscore_s8_mean": float(z[:, 1].mean()),
        "zscore_s8_std": float(z[:, 1].std()),
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
        npe_steps=trainer.global_step,
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
    npe_lr: float = 2e-4,
    npe_max_steps: int = 20000,
    npe_patience: int = 100,
    n_augmentations: int = 10,
    compressor_tag: str = "",
):
    result = train_npe_and_evaluate.remote(
        lmax=lmax, noise_level=noise_level, backbone=backbone,
        npe_lr=npe_lr, npe_max_steps=npe_max_steps,
        npe_patience=npe_patience,
        n_augmentations=n_augmentations,
        compressor_tag=compressor_tag,
    )
    print(f"\nFiducial FoM: {result['fom_fiducial_median']:.1f} +/- {result['fom_fiducial_std']:.1f}")
    print(f"Test FoM: {result['fom_test_median']:.1f} [{result['fom_test_lo']:.1f}, {result['fom_test_hi']:.1f}]")
    print(f"NPE trained {result['npe_steps']} steps in {result['elapsed_s']:.0f}s")
