"""Modal pipeline for LensingFoM.

Stage 1: Born raytracing — download sims, raytrace, save convergence maps
Stage 2: Tile extraction — harmonic filter + noise + rotation → flat tiles
Stage 3: Build HF dataset — convert tiles to parquet shards
Stage 4: Push HF dataset — upload parquet shards to HuggingFace Hub
Stage 5: Build spectra — compute auto/cross power spectra from tiles

Usage:
    modal run pipeline.py --stage 1          # single stage
    modal run pipeline.py --stage 2 --sim-id 1  # single sim
    modal run pipeline.py --stage all        # run all stages sequentially
"""

import csv
import json
import tarfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libcfitsio-dev", "pkg-config")
    .pip_install(
        "numpy", "scipy", "healpy", "camb", "astropy", "httpx",
        "pyarrow", "datasets", "huggingface_hub",
    )
    .run_commands(
        "mkdir -p /pipeline && python -c \""
        "import httpx; "
        "r = httpx.get('https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/"
        "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'); "
        "open('/pipeline/des_y3_2pt.fits','wb').write(r.content)\""
    )
    .add_local_python_source("lensing")
    .add_local_file("gower_street_runs.csv", "/pipeline/gower_street_runs.csv")
)

app = modal.App("lensing-fom", image=image)

vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"
CSV_PATH = "/pipeline/gower_street_runs.csv"
NZ_PATH = "/pipeline/des_y3_2pt.fits"
NSIDE_OUT = 1024
SIM_URL = "http://star.ucl.ac.uk/GowerStreetSims/simulations/sim{sim_id:05d}.tar.gz"
LMAX_VALUES = [200, 400, 600, 800, 1000]
LMAX_TO_NSIDE = {200: 128, 400: 256, 600: 256, 800: 512, 1000: 512}
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]
NOISE_LEVEL_INDEX = {name: i for i, name in enumerate(NOISE_LEVELS)}
N_TILES = 12  # 3 orientations x 4 equatorial tiles
SIMS_PER_SHARD = 50
N_BINS = 20


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


# ---------------------------------------------------------------------------
# Stage 1: Born raytracing
# ---------------------------------------------------------------------------


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=1800,
    memory=2048,
    retries=modal.Retries(max_retries=3, initial_delay=5.0, backoff_coefficient=2.0),
    max_containers=10,
)
def process_simulation(sim_id: int) -> dict:
    import shutil
    import httpx
    import numpy as np

    from lensing.io import (
        load_des_y3_nz,
        load_shell_info,
        load_sim_params,
    )
    from lensing.raytracing import born_raytrace, compute_lensing_weights

    sim_tag = f"sim{sim_id:05d}"
    out_dir = Path(RESULTS_DIR) / sim_tag
    npz_path = out_dir / "kappa_maps.npz"

    # Skip if already done
    if npz_path.exists():
        print(f"{sim_tag}: already exists, skipping")
        return {"sim_id": sim_id, "status": "skipped"}

    # Download tarball
    url = SIM_URL.format(sim_id=sim_id)
    tarball = Path(f"/tmp/{sim_tag}.tar.gz")
    print(f"{sim_tag}: downloading from {url}")
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tarball, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    print(f"{sim_tag}: download complete ({tarball.stat().st_size / 1e9:.2f} GB)")

    # Extract
    print(f"{sim_tag}: extracting")
    with tarfile.open(tarball) as tf:
        tf.extractall("/tmp/")
    tarball.unlink()

    # Find the simulation directory (contains z_values.txt)
    sim_dir = None
    for p in Path("/tmp").rglob("z_values.txt"):
        sim_dir = p.parent
        break
    if sim_dir is None:
        raise RuntimeError(f"{sim_tag}: could not find z_values.txt after extraction")
    print(f"{sim_tag}: sim_dir = {sim_dir}")

    # Load inputs
    cosmo_params = load_sim_params(CSV_PATH, sim_id)
    shell_info = load_shell_info(sim_dir)
    z_nz, nz_bins = load_des_y3_nz(NZ_PATH)

    # Compute lensing weights and raytrace
    print(f"{sim_tag}: computing lensing weights")
    weights = compute_lensing_weights(shell_info, z_nz, nz_bins, cosmo_params)

    print(f"{sim_tag}: raytracing at nside={NSIDE_OUT}")
    kappa_maps = born_raytrace(sim_dir, shell_info, weights, nside_out=NSIDE_OUT)

    # Save results to Volume
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        npz_path,
        **{f"kappa_bin{i}": kappa_maps[i] for i in range(len(kappa_maps))},
    )

    metadata = {
        "sim_id": sim_id,
        "nside": NSIDE_OUT,
        "cosmo_params": cosmo_params,
        "n_shells_processed": int(weights.any(axis=0).sum()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    vol.commit()

    print(f"{sim_tag}: saved to {out_dir}")

    # Cleanup extracted simulation data
    shutil.rmtree(sim_dir, ignore_errors=True)

    return {"sim_id": sim_id, "status": "ok"}


# ---------------------------------------------------------------------------
# Stage 2: Tile extraction with noise
# ---------------------------------------------------------------------------


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=2048,
)
def extract_tiles(sim_id: int) -> dict:
    import numpy as np

    from lensing.tiles import LMAX_TO_NSIDE, extract_tiles_for_lmax

    sim_tag = f"sim{sim_id:05d}"
    sim_dir = Path(RESULTS_DIR) / sim_tag
    npz_path = sim_dir / "kappa_maps.npz"

    # Check source exists
    if not npz_path.exists():
        print(f"{sim_tag}: no kappa_maps.npz, skipping")
        return {"sim_id": sim_id, "status": "no_source"}

    # Check if already done (all lmax x noise_level files exist)
    all_tile_paths = [
        sim_dir / f"tiles_lmax{lmax}_noise_{nl}.npz"
        for lmax in LMAX_VALUES
        for nl in NOISE_LEVELS
    ]
    if all(p.exists() for p in all_tile_paths):
        print(f"{sim_tag}: all tiles already exist, skipping")
        return {"sim_id": sim_id, "status": "skipped"}

    # Load full-sky kappa maps
    data = np.load(npz_path)
    kappa_maps = [data[f"kappa_bin{i}"] for i in range(4)]

    for noise_level in NOISE_LEVELS:
        noise_idx = NOISE_LEVEL_INDEX[noise_level]
        # Deterministic seed per (sim_id, noise_level)
        rng = np.random.default_rng(sim_id * 1000 + noise_idx)
        # Pre-generate one noise realization per bin (reused across lmax)
        if noise_level != "noiseless":
            from lensing.tiles import NOISE_CONFIGS, add_shape_noise
            cfg = NOISE_CONFIGS[noise_level]
            noisy_maps = [
                add_shape_noise(
                    kappa_maps[b],
                    n_eff_arcmin2=cfg["n_eff_arcmin2"][b],
                    sigma_e=cfg["sigma_e"][b],
                    rng=rng,
                )
                for b in range(4)
            ]
        else:
            noisy_maps = kappa_maps

        for lmax in LMAX_VALUES:
            tile_path = sim_dir / f"tiles_lmax{lmax}_noise_{noise_level}.npz"
            if tile_path.exists():
                continue

            nside_down = LMAX_TO_NSIDE[lmax]
            # Shape: (12, 4, nside_down, nside_down)
            all_tiles = np.stack(
                [extract_tiles_for_lmax(noisy_maps[b], lmax) for b in range(4)],
                axis=1,
            )
            np.savez_compressed(tile_path, tiles=all_tiles)
            print(f"{sim_tag}: saved {tile_path.name} shape={all_tiles.shape}")

    vol.commit()
    return {"sim_id": sim_id, "status": "ok"}


# ---------------------------------------------------------------------------
# Stage 3: Build HuggingFace parquet shards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 4: Push HuggingFace dataset to Hub
# ---------------------------------------------------------------------------


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=3600,
    memory=4096,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_hub(
    repo_id: str = "EiffL/GowerStreetDESY3",
    lmax_filter: int = None,
    noise_filter: str = None,
) -> dict:
    """Push parquet shards to HuggingFace Hub."""
    import os
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    hf_dir = Path(RESULTS_DIR) / "hf_dataset"
    lmax_list = [lmax_filter] if lmax_filter else LMAX_VALUES
    noise_list = [noise_filter] if noise_filter else NOISE_LEVELS
    pushed = {}

    for lmax in lmax_list:
        for noise_level in noise_list:
            config_name = f"lmax_{lmax}_{noise_level}"
            config_dir = hf_dir / config_name
            if not config_dir.exists():
                print(f"{config_name}: no parquet shards found, skipping")
                continue

            parquet_files = sorted(config_dir.glob("shard_*.parquet"))
            if not parquet_files:
                print(f"{config_name}: no parquet files, skipping")
                continue

            for pf in parquet_files:
                api.upload_file(
                    path_or_fileobj=str(pf),
                    path_in_repo=f"data/{config_name}/{pf.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"{config_name}: uploaded {pf.name}")

            pushed[config_name] = len(parquet_files)
            print(f"{config_name}: pushed {len(parquet_files)} shards")

    return {"repo_id": repo_id, "pushed": pushed}


# ---------------------------------------------------------------------------
# Stage 5: Build power spectra dataset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Born raytracing",
    2: "Tile extraction",
    3: "Build HF dataset",
    4: "Push HF dataset",
    5: "Build spectra",
}


def _run_map_stage(stage_num, func, sim_ids):
    """Run a per-sim .map() stage and print summary."""
    results = list(func.map(sim_ids, return_exceptions=True))
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [
        (sim_ids[i], r)
        for i, r in enumerate(results)
        if isinstance(r, Exception)
    ]
    print(f"Stage {stage_num} ({STAGE_NAMES[stage_num]}): {len(successes)} succeeded, {len(failures)} failed")
    for sid, err in failures:
        print(f"  sim{sid:05d}: {err}")
    return len(failures) == 0


def _run_starmap_stage(stage_num, func, configs):
    """Run a per-config .starmap() stage and print summary."""
    results = list(func.starmap(configs))
    for r in results:
        print(r)
    print(f"Stage {stage_num} ({STAGE_NAMES[stage_num]}): {len(configs)} configs processed")
    return True


def _run_stage(stage_num, sim_id=None):
    """Dispatch a single stage."""
    sim_ids = list(range(1, 792))
    configs = [(l, n) for l in LMAX_VALUES for n in NOISE_LEVELS]

    if stage_num == 1:
        if sim_id is not None:
            print(process_simulation.remote(sim_id))
        else:
            _run_map_stage(1, process_simulation, sim_ids)

    elif stage_num == 2:
        if sim_id is not None:
            print(extract_tiles.remote(sim_id))
        else:
            _run_map_stage(2, extract_tiles, sim_ids)

    elif stage_num == 3:
        _run_starmap_stage(3, build_parquet_for_config, configs)

    elif stage_num == 4:
        result = push_to_hub.remote()
        print(result)
        print(f"Stage 4 ({STAGE_NAMES[4]}): done")

    elif stage_num == 5:
        _run_starmap_stage(5, build_spectra_for_config, configs)


@app.local_entrypoint()
def main(sim_id: int = None, stage: str = "1"):
    if stage == "all":
        for s in [1, 2, 3, 4, 5]:
            print(f"\n{'='*60}")
            print(f"Starting stage {s}: {STAGE_NAMES[s]}")
            print(f"{'='*60}\n")
            _run_stage(s, sim_id=sim_id if s <= 2 else None)
    else:
        _run_stage(int(stage), sim_id=sim_id)
