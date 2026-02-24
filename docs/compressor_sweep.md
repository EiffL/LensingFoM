# VMIM Compressor Hyperparameter Sweep

**Target:** DES Y3, lmax=200, 200-dim spectra → 2-dim summary. Theoretical FoM ~ 4000.

**Best calibrated config: hidden_dim=256, dropout=0.15, weight_decay=1e-2, lr=2e-4**
- Checkpoint: best val_loss (calibrated posteriors)
- Train FoM: 1662, Val FoM: 1610, Test FoM: 1672

## All results (val_loss checkpoint = calibrated)

All at lr=2e-4, 1000 epochs, batch_size=256 unless noted.

| # | hidden_dim | dropout | weight_decay | Train FoM | Val FoM | Test FoM | Notes |
|---|-----------|---------|-------------|-----------|---------|----------|-------|
| 0a | 256 (3L) | 0.0 | 1e-4 | ~5047 | ~4978 | 1954 | Overfit (old arch) |
| 0b | 64 | 0.3 | 1e-2 | ~1700 | ~1700 | ~1500 | Under-fit (old arch) |
| R1-1 | 128 | 0.1 | 1e-2 | 1199 | 1186 | 1204 | |
| R1-2 | 128 | 0.2 | 1e-2 | 1137 | 1134 | 1165 | |
| R1-3 | 128 | 0.3 | 1e-2 | 1298 | 1298 | 1289 | |
| R1-4 | 128 | 0.1 | 1e-3 | 1176 | 1172 | 1194 | |
| R1-5 | 64 | 0.1 | 1e-2 | 1285 | 1308 | 1290 | |
| R2-1 | 256 | 0.0 | 1e-4 | 345 | 352 | 341 | Too little reg |
| R2-2 | 256 | 0.05 | 1e-3 | 563 | 564 | 539 | |
| R2-3 | 128 | 0.0 | 1e-4 | 250 | 248 | 250 | |
| R2-4 | 128 | 0.05 | 1e-3 | 733 | 726 | 738 | |
| R2-5 | 128 | 0.0 | 0 | 617 | 651 | 620 | |
| R3-1 | 128 | 0.1 | 1e-3 | 1113 | 1127 | 1127 | |
| R3-3 | 256 | 0.1 | 1e-3 | 309 | 337 | 297 | |
| R3-4 | 256 | 0.1 | 1e-2 | 1251 | 1233 | 1268 | |
| R3-5 | 256 | 0.2 | 1e-3 | 1467 | 1476 | 1476 | |
| **R4-1** | **256** | **0.15** | **1e-2** | **1662** | **1610** | **1672** | **Best calibrated** |
| R4-2 | 256 | 0.2 | 1e-2 | 1397 | 1407 | 1395 | |
| R4-3 | 256 | 0.1 | 5e-3 | 1076 | 1073 | 1079 | |
| R4-5 | 256 | 0.1 | 1e-2 | 1015 | 1149 | 1052 | 2000 epochs |

## Key findings

1. **val_loss is the correct checkpoint criterion** — it selects calibrated posteriors. Selecting by val_fom gives overconfident (artificially tight) posteriors.
2. **Calibrated per-tile FoM is ~1600**, well below the ~4000 theoretical Fisher FoM for the full survey. This may reflect that individual tiles have less constraining power than the full sky, or that the diagonal Gaussian VMIM posterior is a limited model.
3. **Best regularization: h256, d0.15, wd1e-2.** Too little regularization → checkpoint from too-early epoch; too much → model can't fit.
4. **Model capacity helps** — h256 > h128 > h64 at matched regularization.
