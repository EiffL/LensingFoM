"""Build a HuggingFace dataset from extracted tiles on the Modal volume.

Runs on Modal for direct volume access. Builds parquet shards per
(lmax, noise_level), then pushes to HuggingFace Hub.

Usage:
    # Build parquet shards (all configs)
    modal run scripts/build_hf_dataset.py

    # Build for a specific config
    modal run scripts/build_hf_dataset.py --lmax 200 --noise-level des_y3

    # Push to HuggingFace Hub (requires HF_TOKEN secret on Modal)
    modal run scripts/build_hf_dataset.py --push
"""

import csv
from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]
LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]
N_TILES = 12  # 3 orientations x 4 equatorial tiles
SIMS_PER_SHARD = 50

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow", "datasets", "huggingface_hub")
    .add_local_file("gower_street_runs.csv", "/pipeline/gower_street_runs.csv")
)

app = modal.App("lensing-hf-dataset", image=image)
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
                "Omega_bh2": float(row[6]),
                "h": float(row[7]),
                "n_s": float(row[8]),
                "m_nu": float(row[9]),
                "Omega_b": float(row[10]),
            }
    return params


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=8192,
)
def build_parquet_for_config(lmax: int, noise_level: str) -> dict:
    """Build parquet shards for one (lmax, noise_level) combination."""
    import numpy as np
    from datasets import Dataset

    cosmo_params = load_cosmo_params(CSV_PATH)
    nside = LMAX_TO_NSIDE[lmax]

    out_dir = Path(RESULTS_DIR) / "hf_dataset" / f"lmax_{lmax}_{noise_level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    shard_idx = 0
    skipped = 0

    for shard_start in range(1, 792, SIMS_PER_SHARD):
        shard_end = min(shard_start + SIMS_PER_SHARD, 792)
        records = []

        for sim_id in range(shard_start, shard_end):
            sim_tag = f"sim{sim_id:05d}"
            npz_path = Path(RESULTS_DIR) / sim_tag / f"tiles_lmax{lmax}_noise_{noise_level}.npz"

            if not npz_path.exists():
                skipped += 1
                continue

            if sim_id not in cosmo_params:
                skipped += 1
                continue

            data = np.load(npz_path)
            tiles = data["tiles"]  # shape (12, 4, nside, nside)
            cp = cosmo_params[sim_id]

            for tile_idx in range(N_TILES):
                records.append({
                    "kappa": tiles[tile_idx].tolist(),  # (4, nside, nside)
                    "sim_id": sim_id,
                    "orientation_id": tile_idx // 4,
                    "tile_id": tile_idx % 4,
                    "noise_level": noise_level,
                    "Omega_m": cp["Omega_m"],
                    "sigma_8": cp["sigma_8"],
                    "S8": cp["S8"],
                    "w": cp["w"],
                    "h": cp["h"],
                    "n_s": cp["n_s"],
                    "Omega_b": cp["Omega_b"],
                    "m_nu": cp["m_nu"],
                })

        if records:
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            ds = Dataset.from_list(records)
            ds.to_parquet(str(shard_path))
            total_tiles += len(records)
            print(f"lmax={lmax}/{noise_level}: wrote {shard_path.name} ({len(records)} tiles, sims {shard_start}-{shard_end - 1})")
            del records, ds
        shard_idx += 1

    vol.commit()
    print(f"lmax={lmax}/{noise_level}: done — {total_tiles} tiles in {shard_idx} shards, {skipped} sims skipped")
    return {"lmax": lmax, "noise_level": noise_level, "total_tiles": total_tiles, "shards": shard_idx, "skipped": skipped}


@app.local_entrypoint()
def main(lmax: int = None, noise_level: str = None, push: bool = False):
    if push:
        print("Use scripts/push_hf_dataset.py for pushing to HuggingFace Hub.")
        return

    if lmax is not None and noise_level is not None:
        result = build_parquet_for_config.remote(lmax, noise_level)
        print(result)
    else:
        # Build all (lmax, noise_level) combinations
        configs = [(l, n) for l in LMAX_VALUES for n in NOISE_LEVELS]
        results = list(build_parquet_for_config.starmap(configs))
        for r in results:
            print(r)
        print(f"\nAll {len(configs)} configs built. Use scripts/push_hf_dataset.py to upload.")
