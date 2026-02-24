"""Synthetic validation: VMIM compressor + FoM convergence to Fisher theory.

Runs on Modal with GPU. Loads synthetic data from Modal volume, trains VMIM
compressor, evaluates with NPE, logs everything to wandb (entity: eiffl).

Usage:
    # First generate data (one-time):
    modal run scripts/generate_synthetic_data.py

    # Single run (50k train, diagonal VMIM)
    modal run scripts/validate_compressor.py

    # Full-covariance VMIM head
    modal run scripts/validate_compressor.py --full-cov

    # Data budget sweep
    modal run scripts/validate_compressor.py --sweep

    # Custom hyperparams
    modal run scripts/validate_compressor.py --n-train 10000 --lr 1e-3 --hidden-dim 512
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "scipy",
        "torch", "lightning", "wandb",
        "matplotlib",
    )
    .add_local_python_source("lensing")
)

# Pre-computed Fisher FoM values (from CAMB finite differences, f_sky=1/12,
# DES Y3 MagLim n(z), 20 linear ell bins). Includes DES Y3 shape noise.
FISHER_FOM = {200: 3742, 400: 6627, 600: 8897, 800: 10786, 1000: 12234}

app = modal.App("lensing-validate", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    image=image,
    volumes={RESULTS_DIR: vol},
    gpu="A100",
    timeout=3600,
    memory=16384,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_and_evaluate(
    n_train: int = 50000,
    lmax: int = 1000,
    max_epochs: int = 500,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    summary_dim: int = 2,
    full_cov: bool = False,
    patience: int = 50,
    seed: int = 42,
    data_file: str = "synthetic/synthetic_lmax1000_n70000_des_y3.npz",
):
    """Train VMIM compressor on synthetic data, evaluate, log to wandb."""
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
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset

    from lensing.sbi.compressor import VMIMCompressor
    from lensing.sbi.npe import train_npe, compute_fom

    t0 = time.time()

    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load data ---
    npz_path = Path(RESULTS_DIR) / data_file
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run generate_synthetic_data.py first."
        )

    data = np.load(npz_path)
    spectra = data["spectra"]
    theta = data["theta"]
    ell_eff = data["ell_eff"]
    f_sky = float(data["f_sky"])

    n_total = len(spectra)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)

    n_val = min(int(0.15 * n_total), n_total - n_train)
    n_test = n_total - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Normalize
    spec_mean = spectra[train_idx].mean(axis=0)
    spec_std = spectra[train_idx].std(axis=0)
    spec_std = np.maximum(spec_std, np.maximum(np.abs(spec_mean) * 1e-6, 1e-30))
    theta_mean = theta[train_idx].mean(axis=0)
    theta_std = theta[train_idx].std(axis=0)
    theta_std = np.where(theta_std == 0, 1.0, theta_std)

    # Pre-load everything on GPU as tensors — dataset is small (~38MB),
    # avoids CPU→GPU transfer bottleneck that causes ~1% GPU utilization.
    # Use large batches (n_train//4) since data fits entirely in GPU memory.
    batch_size = min(2048, max(256, n_train // 25))

    def _make_dl(idx, shuffle=False):
        x = torch.tensor((spectra[idx] - spec_mean) / spec_std, dtype=torch.float32).to(device)
        y = torch.tensor((theta[idx] - theta_mean) / theta_std, dtype=torch.float32).to(device)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                          num_workers=0)

    train_dl = _make_dl(train_idx, shuffle=True)
    val_dl = _make_dl(val_idx)

    input_dim = spectra.shape[1]
    print(f"Data: {n_train} train / {n_val} val / {n_test} test, input_dim={input_dim}")
    print(f"Batch size: {batch_size} (GPU-resident tensors)")

    # --- Data diagnostics ---
    x_train = (spectra[train_idx] - spec_mean) / spec_std
    y_train = (theta[train_idx] - theta_mean) / theta_std
    print(f"Spectra raw: mean={spectra[train_idx].mean():.3e}, std={spectra[train_idx].std():.3e}, "
          f"min={spectra[train_idx].min():.3e}, max={spectra[train_idx].max():.3e}")
    print(f"Spectra norm: mean={x_train.mean():.3f}, std={x_train.std():.3f}, "
          f"min={x_train.min():.3f}, max={x_train.max():.3f}")
    print(f"Theta norm: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"spec_std range: [{spec_std.min():.3e}, {spec_std.max():.3e}]")
    print(f"NaN check: spectra={np.isnan(spectra).sum()}, theta={np.isnan(theta).sum()}")
    print(f"Inf check: spectra={np.isinf(spectra).sum()}, theta={np.isinf(theta).sum()}")

    # --- Fisher FoM (pre-computed) ---
    fisher_fom = FISHER_FOM.get(lmax)
    if fisher_fom:
        print(f"Fisher FoM at lmax={lmax}: {fisher_fom}")

    # --- FoM tracker callback ---
    class FoMTracker(Callback):
        def __init__(self):
            self.epochs, self.val_foms, self.val_losses = [], [], []

        def on_validation_epoch_end(self, trainer, pl_module):
            vl = trainer.callback_metrics.get("val_loss")
            vf = trainer.callback_metrics.get("val_fom_median")
            if vl is not None and vf is not None:
                self.epochs.append(trainer.current_epoch)
                self.val_losses.append(float(vl))
                self.val_foms.append(float(vf))

    # --- Build tag ---
    tag = f"n{n_train}_lmax{lmax}"
    if full_cov:
        tag += "_fullcov"
    if summary_dim != 2:
        tag += f"_sdim{summary_dim}"

    run_dir = Path(RESULTS_DIR) / "synthetic_validation" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model ---
    model = VMIMCompressor(
        input_dim=input_dim,
        summary_dim=summary_dim,
        theta_dim=2,
        hidden_dim=hidden_dim,
        full_cov=full_cov,
        lr=lr,
        min_lr=1e-4,
        weight_decay=1e-4,
        plateau_patience=5,
        plateau_factor=0.5,
        theta_std=theta_std,
    )

    # --- Loggers + callbacks ---
    wandb_config = dict(
        n_train=n_train, lmax=lmax, input_dim=input_dim,
        summary_dim=summary_dim, hidden_dim=hidden_dim, full_cov=full_cov,
        lr=lr, batch_size=batch_size, patience=patience,
        max_epochs=max_epochs, fisher_fom=fisher_fom,
    )
    wandb_logger = WandbLogger(
        project="LensingFoM-synthetic",
        entity="eiffl",
        name=tag,
        config=wandb_config,
        save_dir=str(run_dir),
    )

    fom_tracker = FoMTracker()
    callbacks = [
        fom_tracker,
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(
            dirpath=str(run_dir), filename="compressor-best",
            monitor="val_loss", mode="min",
        ),
    ]

    # --- Train ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=wandb_logger,
        accelerator="auto",
    )
    trainer.fit(model, train_dl, val_dl)

    # Load best checkpoint
    best_path = callbacks[2].best_model_path
    if best_path:
        model = VMIMCompressor.load_from_checkpoint(
            best_path, theta_std=theta_std, weights_only=False
        )
    model = model.cpu()
    model.eval()

    # --- Evaluate VMIM head ---
    x_test = torch.tensor(
        (spectra[test_idx] - spec_mean) / spec_std, dtype=torch.float32
    )
    theta_test_norm = torch.tensor(
        (theta[test_idx] - theta_mean) / theta_std, dtype=torch.float32
    )
    with torch.no_grad():
        _, test_foms_vmim = model._nll_and_fom(x_test, theta_test_norm)
    test_fom_vmim = float(test_foms_vmim.median())
    print(f"\nVMIM test FoM: {test_fom_vmim:.1f}")

    # --- Train NPE ---
    test_spectra_norm = (spectra[test_idx] - spec_mean) / spec_std
    train_spectra_norm = (spectra[train_idx] - spec_mean) / spec_std
    val_spectra_norm = (spectra[val_idx] - spec_mean) / spec_std
    train_theta_norm = (theta[train_idx] - theta_mean) / theta_std
    val_theta_norm = (theta[val_idx] - theta_mean) / theta_std

    summaries_train = model.compress_arrays(train_spectra_norm)
    summaries_val = model.compress_arrays(val_spectra_norm)
    summaries_test = model.compress_arrays(test_spectra_norm)

    npe_model = train_npe(
        summaries_train, train_theta_norm,
        val_summaries=summaries_val, val_theta=val_theta_norm,
        max_epochs=300, patience=20,
    )
    npe_model = npe_model.cpu()
    test_fom_npe, _, _, _ = compute_fom(npe_model, summaries_test)
    print(f"NPE  test FoM: {test_fom_npe:.1f}")

    # --- Calibration ---
    mu_norm, sigma_norm = model.predict_posterior(test_spectra_norm)
    mu_phys = mu_norm * theta_std + theta_mean
    sigma_phys = sigma_norm * theta_std
    z_scores = (mu_phys - theta[test_idx]) / sigma_phys
    cov_1sig = float((np.abs(z_scores) < 1.0).mean())
    cov_2sig = float((np.abs(z_scores) < 2.0).mean())
    print(f"1σ coverage: {cov_1sig:.1%} (target 68.3%)")
    print(f"2σ coverage: {cov_2sig:.1%} (target 95.4%)")

    if fisher_fom:
        print(f"Fisher FoM:    {fisher_fom:.1f}")
        print(f"VMIM/Fisher:   {test_fom_vmim / fisher_fom:.1%}")

    # --- Diagnostic plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(fom_tracker.epochs, fom_tracker.val_losses, "b-", lw=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss"); ax.set_title("Validation Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(fom_tracker.epochs, fom_tracker.val_foms, "r-", lw=1, label="VMIM val")
    if fisher_fom:
        ax.axhline(fisher_fom, color="green", ls="--", lw=2, label=f"Fisher={fisher_fom:.0f}")
    ax.axhline(test_fom_vmim, color="k", ls=":", alpha=0.7, label=f"VMIM test={test_fom_vmim:.0f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("FoM"); ax.set_title("FoM vs Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    z_grid = np.linspace(-4, 4, 100)
    ax.plot(z_grid, np.exp(-0.5 * z_grid**2) / np.sqrt(2*np.pi), "k--", lw=1.5, label="N(0,1)")
    for i, name in enumerate([r"$\Omega_m$", r"$S_8$"]):
        ax.hist(z_scores[:, i], bins=40, density=True, alpha=0.5, label=name)
    ax.set_xlabel("z-score"); ax.set_ylabel("Density")
    ax.set_title(f"Calibration (1σ={cov_1sig:.1%})"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"VMIM test FoM: {test_fom_vmim:.1f}",
        f"NPE  test FoM: {test_fom_npe:.1f}",
        f"Fisher FoM:    {fisher_fom:.1f}" if fisher_fom else "",
        "", f"1σ coverage: {cov_1sig:.1%}  (target 68.3%)",
        f"2σ coverage: {cov_2sig:.1%}  (target 95.4%)", "",
    ]
    if fisher_fom:
        lines.append(f"VMIM/Fisher: {test_fom_vmim/fisher_fom:.1%}")
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes,
            fontsize=12, va="top", fontfamily="monospace")
    ax.set_title("Summary")

    fig.suptitle(f"Synthetic Validation ({tag})", fontsize=14)
    fig.tight_layout()
    plot_path = run_dir / f"diagnostics_{tag}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Log to wandb ---
    wandb_logger.experiment.summary.update({
        "test/vmim_fom": test_fom_vmim,
        "test/npe_fom": test_fom_npe,
        "test/coverage_1sig": cov_1sig,
        "test/coverage_2sig": cov_2sig,
        "test/vmim_fisher_ratio": test_fom_vmim / fisher_fom if fisher_fom else None,
    })
    wandb_logger.experiment.log({"diagnostics": wandb.Image(str(plot_path))})

    # --- Save results ---
    result = dict(
        n_train=n_train, lmax=lmax, input_dim=input_dim,
        summary_dim=summary_dim, hidden_dim=hidden_dim, full_cov=full_cov,
        lr=lr, batch_size=batch_size,
        vmim_fom=test_fom_vmim, npe_fom=test_fom_npe, fisher_fom=fisher_fom,
        coverage_1sig=cov_1sig, coverage_2sig=cov_2sig,
        epochs_trained=trainer.current_epoch,
        elapsed_s=time.time() - t0,
    )
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    wandb.finish()

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    return result


@app.local_entrypoint()
def main(
    n_train: int = 50000,
    lmax: int = 1000,
    max_epochs: int = 500,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    summary_dim: int = 2,
    full_cov: bool = False,
    patience: int = 50,
    seed: int = 42,
    sweep: bool = False,
):
    if sweep:
        for nt in [1000, 5000, 10000, 50000]:
            result = train_and_evaluate.remote(
                n_train=nt, lmax=lmax, max_epochs=max_epochs,
                lr=lr, hidden_dim=hidden_dim, summary_dim=summary_dim,
                full_cov=full_cov, patience=patience, seed=seed,
            )
            print(f"\nn_train={nt}: VMIM FoM={result['vmim_fom']:.1f}, "
                  f"NPE FoM={result['npe_fom']:.1f}, "
                  f"1σ={result['coverage_1sig']:.1%}")
    else:
        result = train_and_evaluate.remote(
            n_train=n_train, lmax=lmax, max_epochs=max_epochs,
            lr=lr, hidden_dim=hidden_dim, summary_dim=summary_dim,
            full_cov=full_cov, patience=patience, seed=seed,
        )
        print(f"\nResult: VMIM FoM={result['vmim_fom']:.1f}, "
              f"NPE FoM={result['npe_fom']:.1f}, "
              f"Coverage 1σ={result['coverage_1sig']:.1%}")
        if result.get("fisher_fom"):
            print(f"VMIM/Fisher: {result['vmim_fom']/result['fisher_fom']:.1%}")
