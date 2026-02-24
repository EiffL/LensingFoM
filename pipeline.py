"""Modal pipeline — Stage 1: Generate convergence maps; Stage 2: Extract filtered tiles."""

import json
import tarfile
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libcfitsio-dev", "pkg-config")
    .pip_install("numpy", "scipy", "healpy", "camb", "astropy", "httpx")
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


@app.function(
    volumes={RESULTS_DIR: vol},
    timeout=1800,
    memory=1024,
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

    # Check if already done (all lmax files exist)
    tile_paths = [sim_dir / f"tiles_lmax{lmax}.npz" for lmax in LMAX_VALUES]
    if all(p.exists() for p in tile_paths):
        print(f"{sim_tag}: tiles already exist, skipping")
        return {"sim_id": sim_id, "status": "skipped"}

    # Load full-sky kappa maps
    data = np.load(npz_path)
    kappa_maps = [data[f"kappa_bin{i}"] for i in range(4)]

    for lmax in LMAX_VALUES:
        tile_path = sim_dir / f"tiles_lmax{lmax}.npz"
        if tile_path.exists():
            continue

        nside_down = LMAX_TO_NSIDE[lmax]
        # Shape: (12, 4, nside_down, nside_down)
        all_tiles = np.stack(
            [extract_tiles_for_lmax(kappa_maps[b], lmax) for b in range(4)],
            axis=1,
        )
        np.savez_compressed(tile_path, tiles=all_tiles)
        print(f"{sim_tag}: saved {tile_path.name} shape={all_tiles.shape}")

    vol.commit()
    return {"sim_id": sim_id, "status": "ok"}


@app.local_entrypoint()
def main(sim_id: int = None, stage: int = 1):
    if stage == 1:
        if sim_id is not None:
            result = process_simulation.remote(sim_id)
            print(result)
        else:
            sim_ids = list(range(1, 792))
            results = list(process_simulation.map(sim_ids, return_exceptions=True))
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [
                (sim_ids[i], r)
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            print(f"{len(successes)} succeeded, {len(failures)} failed")
            for sid, err in failures:
                print(f"  sim{sid:05d}: {err}")
    elif stage == 2:
        if sim_id is not None:
            result = extract_tiles.remote(sim_id)
            print(result)
        else:
            sim_ids = list(range(1, 792))
            results = list(extract_tiles.map(sim_ids, return_exceptions=True))
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [
                (sim_ids[i], r)
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            print(f"Stage 2: {len(successes)} succeeded, {len(failures)} failed")
            for sid, err in failures:
                print(f"  sim{sid:05d}: {err}")
