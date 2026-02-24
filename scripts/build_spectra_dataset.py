"""Build a spectra dataset from extracted tiles on the Modal volume.

For each tile, computes all 10 binned auto/cross power spectra using
flat-sky FFT estimation. Saves one parquet file per (lmax, noise_level).

Usage:
    modal run scripts/build_spectra_dataset.py                          # all configs
    modal run scripts/build_spectra_dataset.py --lmax 600               # single lmax, all noise
    modal run scripts/build_spectra_dataset.py --lmax 600 --noise-level des_y3  # specific config
"""

import csv
from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]
LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]
N_TILES = 12
N_BINS = 20

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow")
    .add_local_python_source("lensing")
    .add_local_file("gower_street_runs.csv", "/pipeline/gower_street_runs.csv")
)

app = modal.App("lensing-spectra", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"
CSV_PATH = "/pipeline/gower_street_runs.csv"


def load_cosmo_params(csv_path):
    """Load all cosmological parameters from gower_street_runs.csv."""
    params = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # category row
        next(reader)  # column names
        for row in reader:
            sim_id = int(row[0])
            omega_m = float(row[3])
            sigma_8 = float(row[4])
            params[sim_id] = {
                "Omega_m": omega_m,
                "sigma_8": sigma_8,
                "S8": sigma_8 * (omega_m / 0.3) ** 0.5,
                "w": float(row[5]),
                "h": float(row[7]),
                "n_s": float(row[8]),
                "m_nu": float(row[9]),
                "Omega_b": float(row[10]),
            }
    return params


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=4096,
)
def build_spectra_for_config(lmax: int, noise_level: str) -> dict:
    """Compute power spectra for all tiles at one (lmax, noise_level), save as parquet."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lensing.spectra import SPEC_PAIRS, compute_all_spectra

    cosmo_params = load_cosmo_params(CSV_PATH)
    nside = LMAX_TO_NSIDE[lmax]

    out_dir = Path(RESULTS_DIR) / "spectra_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spectra_lmax{lmax}_noise_{noise_level}.parquet"

    records = []
    skipped = 0

    for sim_id in range(1, 792):
        sim_tag = f"sim{sim_id:05d}"
        npz_path = Path(RESULTS_DIR) / sim_tag / f"tiles_lmax{lmax}_noise_{noise_level}.npz"

        if not npz_path.exists():
            skipped += 1
            continue
        if sim_id not in cosmo_params:
            skipped += 1
            continue

        tiles = np.load(npz_path)["tiles"]  # (12, 4, nside, nside)
        cp = cosmo_params[sim_id]

        for tile_idx in range(N_TILES):
            ell_eff, spectra = compute_all_spectra(
                tiles[tile_idx], nside, lmax, n_bins=N_BINS
            )

            record = {
                "sim_id": sim_id,
                "orientation_id": tile_idx // 4,
                "tile_id": tile_idx % 4,
                "noise_level": noise_level,
                "ell_eff": ell_eff.tolist(),
                "Omega_m": cp["Omega_m"],
                "sigma_8": cp["sigma_8"],
                "S8": cp["S8"],
                "w": cp["w"],
                "h": cp["h"],
                "n_s": cp["n_s"],
                "Omega_b": cp["Omega_b"],
                "m_nu": cp["m_nu"],
            }
            # Store each spectrum as a separate column: cl_0_0, cl_0_1, ...
            for s, (i, j) in enumerate(SPEC_PAIRS):
                record[f"cl_{i}_{j}"] = spectra[s].tolist()

            records.append(record)

        if sim_id % 50 == 0:
            print(f"lmax={lmax}/{noise_level}: processed {sim_id}/791 sims ({len(records)} tiles)")

    # Write parquet
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(out_path))
    vol.commit()

    print(
        f"lmax={lmax}/{noise_level}: wrote {out_path.name} — "
        f"{len(records)} tiles, {skipped} sims skipped"
    )
    return {"lmax": lmax, "noise_level": noise_level, "total_tiles": len(records), "skipped": skipped}


@app.local_entrypoint()
def main(lmax: int = None, noise_level: str = None):
    if lmax is not None and noise_level is not None:
        result = build_spectra_for_config.remote(lmax, noise_level)
        print(result)
    elif lmax is not None:
        # All noise levels for one lmax
        configs = [(lmax, nl) for nl in NOISE_LEVELS]
        results = list(build_spectra_for_config.starmap(configs))
        for r in results:
            print(r)
    else:
        # All (lmax, noise_level) combinations
        configs = [(l, n) for l in LMAX_VALUES for n in NOISE_LEVELS]
        results = list(build_spectra_for_config.starmap(configs))
        for r in results:
            print(r)
        print(f"\nAll {len(configs)} spectra configs built.")
