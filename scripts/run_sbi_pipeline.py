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
from lensing.sbi.npe import compute_fom, train_npe

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

    n_comp_train = len(dm.comp_train_ds)
    n_comp_val = len(dm.comp_val_ds)
    n_npe = len(dm.npe_ds)
    n_test = len(dm.test_ds)
    print(f"Split: {n_comp_train} comp-train / {n_comp_val} comp-val / {n_npe} NPE / {n_test} test tiles")

    # --- Step 2: Train VMIM compressor ---
    print("\nTraining VMIM compressor...")

    # Estimate total steps for LR schedule
    steps_per_epoch = max(1, n_comp_train // 256)
    total_steps = steps_per_epoch * max_epochs

    input_dim = dm.comp_train_ds.spectra.shape[1]
    model = VMIMCompressor(
        input_dim=input_dim,
        summary_dim=2,
        warmup_steps=steps_per_epoch * 2,
        total_steps=total_steps,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=50, mode="min"),
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
        model = VMIMCompressor.load_from_checkpoint(best_path, weights_only=False)
    print(f"Best val_loss: {callbacks[1].best_model_score:.4f}")

    # --- Step 3: Compress all splits ---
    print("\nCompressing datasets...")
    model = model.cpu()
    npe_summaries, npe_theta = model.compress(dm.npe_ds)
    test_summaries, test_theta = model.compress(dm.test_ds)

    # --- Step 4: Train Gaussian NPE on NPE split (80/20 internal val) ---
    print("Training Gaussian NPE...")
    n_npe_train = int(0.8 * len(npe_summaries))
    npe_model = train_npe(
        npe_summaries[:n_npe_train], npe_theta[:n_npe_train],
        val_summaries=npe_summaries[n_npe_train:], val_theta=npe_theta[n_npe_train:],
    )

    # --- Step 5: Compute FoM on both NPE and test sets ---
    npe_model = npe_model.cpu()

    print("Computing FoM on NPE set (overfitting check)...")
    npe_median, npe_lo, npe_hi, npe_foms = compute_fom(npe_model, npe_summaries)

    print("Computing FoM on test set...")
    test_median, test_lo, test_hi, test_foms = compute_fom(npe_model, test_summaries)

    result = {
        "lmax": lmax,
        "fom_test_median": float(test_median),
        "fom_test_lo_16": float(test_lo),
        "fom_test_hi_84": float(test_hi),
        "fom_npe_median": float(npe_median),
        "fom_npe_lo_16": float(npe_lo),
        "fom_npe_hi_84": float(npe_hi),
        "n_comp_train": n_comp_train,
        "n_comp_val": n_comp_val,
        "n_npe": n_npe,
        "n_test": n_test,
    }

    # Save per-lmax results
    with open(lmax_dir / "fom_result.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(lmax_dir / "test_foms.npy", test_foms)
    np.save(lmax_dir / "npe_foms.npy", npe_foms)
    np.save(lmax_dir / "test_summaries.npy", test_summaries)
    np.save(lmax_dir / "test_theta.npy", test_theta)

    print(f"\n  FoM (NPE set):  {npe_median:.1f} [{npe_lo:.1f}, {npe_hi:.1f}]")
    print(f"  FoM (test set): {test_median:.1f} [{test_lo:.1f}, {test_hi:.1f}]")
    ratio = npe_median / test_median if test_median > 0 else float("inf")
    print(f"  NPE/test ratio: {ratio:.2f} (1.0 = no overfitting)")
    return result


def compute_fisher_theory(lmax_values, nz_fits_path="data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"):
    """Compute Fisher FoM theory prediction for given lmax values.

    Returns list of dicts with keys: lmax, fisher_fom, sigma_omega_m, sigma_s8.
    Returns None if jax-cosmo is not available or n(z) file is missing.
    """
    try:
        from lensing.io import load_des_y3_nz
        from lensing.sbi.fisher import compute_fisher_fom

        nz_path = Path(nz_fits_path)
        if not nz_path.exists():
            print(f"  n(z) file not found at {nz_path}, skipping Fisher prediction")
            return None

        print("\nComputing Fisher FoM theory prediction...")
        z_mid, nz_bins = load_des_y3_nz(nz_path)
        return compute_fisher_fom(z_mid, nz_bins, lmax_values)
    except ImportError as e:
        print(f"  jax-cosmo not available ({e}), skipping Fisher prediction")
        return None


def plot_fom_vs_lmax(results, output_dir, fisher_results=None):
    """Plot FoM vs lmax with error bars, showing test, NPE, and Fisher theory."""
    lmax_vals = [r["lmax"] for r in results]

    def _errbar(key_med, key_lo, key_hi):
        med = [r[key_med] for r in results]
        lo = [m - r[key_lo] for m, r in zip(med, results)]
        hi = [r[key_hi] - m for m, r in zip(med, results)]
        return med, [lo, hi]

    test_med, test_err = _errbar("fom_test_median", "fom_test_lo_16", "fom_test_hi_84")
    npe_med, npe_err = _errbar("fom_npe_median", "fom_npe_lo_16", "fom_npe_hi_84")

    if fisher_results is not None:
        # Two-panel plot: left = SBI + Fisher (dual y-axis), right = SBI only
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: dual y-axis showing Fisher and SBI together
        fisher_lmax = [r["lmax"] for r in fisher_results]
        fisher_fom = [r["fisher_fom"] for r in fisher_results]

        ax_fisher = ax_left
        ax_fisher.plot(fisher_lmax, fisher_fom, "D-", color="C2", markersize=7,
                       label="Fisher (noiseless theory)")
        ax_fisher.set_xlabel(r"$\ell_{\max}$")
        ax_fisher.set_ylabel(r"Fisher FoM$(\Omega_m, S_8)$", color="C2")
        ax_fisher.tick_params(axis="y", labelcolor="C2")

        ax_sbi = ax_fisher.twinx()
        ax_sbi.errorbar(lmax_vals, test_med, yerr=test_err,
                        fmt="o-", capsize=4, color="C0", label="SBI (test set)")
        ax_sbi.set_ylabel(r"SBI FoM$(\Omega_m, S_8)$", color="C0")
        ax_sbi.tick_params(axis="y", labelcolor="C0")

        # Combined legend
        lines_fisher, labels_fisher = ax_fisher.get_legend_handles_labels()
        lines_sbi, labels_sbi = ax_sbi.get_legend_handles_labels()
        ax_fisher.legend(lines_fisher + lines_sbi, labels_fisher + labels_sbi, loc="upper left")
        ax_left.set_title("SBI vs Fisher Theory")
        ax_left.grid(True, alpha=0.3)

        # Right panel: SBI only (test + NPE)
        ax_right.errorbar(lmax_vals, test_med, yerr=test_err,
                          fmt="o-", capsize=4, color="C0", label="Test set")
        ax_right.errorbar(lmax_vals, npe_med, yerr=npe_err,
                          fmt="s--", capsize=4, color="C1", alpha=0.7,
                          label="NPE set (overfitting check)")
        ax_right.set_xlabel(r"$\ell_{\max}$")
        ax_right.set_ylabel(r"FoM$(\Omega_m, S_8)$")
        ax_right.set_title("SBI Figure of Merit vs Angular Scale")
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

        fig.tight_layout()
    else:
        # Single panel: SBI only
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(lmax_vals, test_med, yerr=test_err,
                    fmt="o-", capsize=4, color="C0", label="Test set")
        ax.errorbar(lmax_vals, npe_med, yerr=npe_err,
                    fmt="s--", capsize=4, color="C1", alpha=0.7,
                    label="NPE set (overfitting check)")
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
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, just regenerate plot from existing results")
    parser.add_argument("--no-fisher", action="store_true",
                        help="Skip Fisher theory prediction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("Downloading spectra parquet files...")
        download_parquet(data_dir)

    lmax_values = [args.lmax] if args.lmax else LMAX_VALUES

    if args.plot_only:
        # Load existing results
        results_path = output_dir / "all_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"No results found at {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {results_path}")
    else:
        results = []
        for lmax in lmax_values:
            result = run_single_lmax(lmax, data_dir, output_dir, args.max_epochs)
            results.append(result)

        # Save aggregated results
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(results, f, indent=2)

    if len(results) > 1:
        fisher_results = None
        if not args.no_fisher:
            fisher_lmax = [r["lmax"] for r in results]
            fisher_results = compute_fisher_theory(fisher_lmax)
            if fisher_results is not None:
                # Save Fisher results alongside SBI results
                with open(output_dir / "fisher_results.json", "w") as f:
                    json.dump(fisher_results, f, indent=2)

        plot_fom_vs_lmax(results, output_dir, fisher_results)

    print("\nDone! Results saved to", output_dir)


if __name__ == "__main__":
    main()
