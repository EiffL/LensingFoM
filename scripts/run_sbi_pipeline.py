"""End-to-end SBI pipeline: VMIM compression + NPE for FoM vs lmax.

Downloads spectra parquet from Modal volume (if needed), trains VMIM
compressor, runs NPE, computes FoM, and produces FoM vs lmax plot.

Usage:
    # Single lmax (for testing)
    python scripts/run_sbi_pipeline.py --lmax 200

    # All lmax values
    python scripts/run_sbi_pipeline.py

    # Custom output directory
    python scripts/run_sbi_pipeline.py --output-dir results/sbi

    # Download parquet from Modal volume first
    python scripts/run_sbi_pipeline.py --download
"""

import argparse
import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lensing.sbi.compressor import VMIMCompressor
from lensing.sbi.dataset import SpectraDataModule
from lensing.sbi.npe import compute_fom, get_prior, train_npe

LMAX_VALUES = [200, 400, 600, 800, 1000]


def download_parquet(output_dir):
    """Download spectra parquet files from Modal volume."""
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lmax in LMAX_VALUES:
        local_path = output_dir / f"spectra_lmax{lmax}.parquet"
        if local_path.exists():
            print(f"  {local_path.name} already exists, skipping")
            continue

        remote_path = f"spectra_dataset/spectra_lmax{lmax}.parquet"
        print(f"  Downloading {remote_path}...")
        subprocess.run(
            ["modal", "volume", "get", "lensing-results", remote_path, str(local_path)],
            check=True,
        )
    print("Download complete.")


def run_single_lmax(lmax, data_dir, output_dir, max_epochs=200):
    """Run full SBI pipeline for a single lmax value.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    data_dir : Path
        Directory containing spectra parquet files.
    output_dir : Path
        Directory to save results.
    max_epochs : int
        Maximum training epochs for compressor.

    Returns
    -------
    result : dict
        FoM results for this lmax.
    """
    parquet_path = data_dir / f"spectra_lmax{lmax}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}. Run with --download first.")

    lmax_dir = output_dir / f"lmax{lmax}"
    lmax_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load and split data ---
    print(f"\n{'='*60}")
    print(f"lmax = {lmax}")
    print(f"{'='*60}")

    dm = SpectraDataModule(parquet_path, batch_size=256)
    dm.setup()

    n_train = len(dm.train_ds)
    n_npe = len(dm.val_ds)
    n_test = len(dm.test_ds)
    print(f"Split: {n_train} train / {n_npe} NPE / {n_test} test tiles")

    # --- Step 2: Train VMIM compressor ---
    print("\nTraining VMIM compressor...")

    # Estimate total steps for LR schedule
    steps_per_epoch = max(1, n_train // 256)
    total_steps = steps_per_epoch * max_epochs

    model = VMIMCompressor(
        input_dim=200,
        summary_dim=2,
        warmup_steps=steps_per_epoch * 2,
        total_steps=total_steps,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        ModelCheckpoint(
            dirpath=str(lmax_dir),
            filename="compressor-best",
            monitor="val_loss",
            mode="min",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=L.pytorch.loggers.CSVLogger(str(lmax_dir), name="compressor"),
    )
    trainer.fit(model, dm)

    # Load best checkpoint
    best_path = callbacks[1].best_model_path
    if best_path:
        model = VMIMCompressor.load_from_checkpoint(best_path)
    print(f"Best val_loss: {callbacks[1].best_model_score:.4f}")

    # --- Step 3: Compress all splits ---
    print("\nCompressing datasets...")
    model = model.cpu()
    train_summaries, train_theta = model.compress(dm.train_ds)
    npe_summaries, npe_theta = model.compress(dm.val_ds)
    test_summaries, test_theta = model.compress(dm.test_ds)

    # --- Step 4: Train NPE ---
    print("Training NPE...")
    prior = get_prior(npe_theta)
    posterior = train_npe(npe_summaries, npe_theta, prior)

    # --- Step 5: Compute FoM on test set ---
    print("Computing FoM on test set...")
    median, lo_16, hi_84, all_foms = compute_fom(posterior, test_summaries)

    result = {
        "lmax": lmax,
        "fom_median": float(median),
        "fom_lo_16": float(lo_16),
        "fom_hi_84": float(hi_84),
        "n_train": n_train,
        "n_npe": n_npe,
        "n_test": n_test,
    }

    # Save per-lmax results
    with open(lmax_dir / "fom_result.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(lmax_dir / "all_foms.npy", all_foms)
    np.save(lmax_dir / "test_summaries.npy", test_summaries)
    np.save(lmax_dir / "test_theta.npy", test_theta)

    print(f"FoM = {median:.1f} [{lo_16:.1f}, {hi_84:.1f}]")
    return result


def plot_fom_vs_lmax(results, output_dir):
    """Plot FoM vs lmax with error bars."""
    lmax_vals = [r["lmax"] for r in results]
    medians = [r["fom_median"] for r in results]
    lo = [r["fom_lo_16"] for r in results]
    hi = [r["fom_hi_84"] for r in results]

    yerr_lo = [m - l for m, l in zip(medians, lo)]
    yerr_hi = [h - m for m, h in zip(medians, hi)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(lmax_vals, medians, yerr=[yerr_lo, yerr_hi],
                fmt="o-", capsize=4, color="C0", label="SBI (VMIM + NPE)")
    ax.set_xlabel(r"$\ell_{\max}$")
    ax.set_ylabel(r"FoM$(\Omega_m, S_8)$")
    ax.set_title("Figure of Merit vs Angular Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "fom_vs_lmax.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fom_vs_lmax.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output_dir / 'fom_vs_lmax.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="SBI pipeline: FoM vs lmax")
    parser.add_argument("--lmax", type=int, default=None,
                        help="Run for a single lmax value (default: all)")
    parser.add_argument("--data-dir", type=str, default="data/spectra",
                        help="Directory with spectra parquet files")
    parser.add_argument("--output-dir", type=str, default="results/sbi",
                        help="Output directory for results")
    parser.add_argument("--download", action="store_true",
                        help="Download parquet from Modal volume first")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max training epochs for compressor")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("Downloading spectra parquet files...")
        download_parquet(data_dir)

    lmax_values = [args.lmax] if args.lmax else LMAX_VALUES

    results = []
    for lmax in lmax_values:
        result = run_single_lmax(lmax, data_dir, output_dir, args.max_epochs)
        results.append(result)

    # Save aggregated results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if len(results) > 1:
        plot_fom_vs_lmax(results, output_dir)

    print("\nDone! Results saved to", output_dir)


if __name__ == "__main__":
    main()
