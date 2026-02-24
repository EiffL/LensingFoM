"""Push parquet shards from the Modal volume to HuggingFace Hub.

Usage:
    modal run scripts/push_hf_dataset.py
    modal run scripts/push_hf_dataset.py --lmax 200 --noise-level des_y3

Requires a Modal secret named 'huggingface-secret' with key HF_TOKEN.
Create it with:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
"""

from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]
NOISE_LEVELS = ["noiseless", "des_y3", "lsst_y10"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
)

app = modal.App("lensing-push-hf", image=image)
vol = modal.Volume.from_name("lensing-results", create_if_missing=True)
RESULTS_DIR = "/results"


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


@app.local_entrypoint()
def main(lmax: int = None, noise_level: str = None):
    result = push_to_hub.remote(lmax_filter=lmax, noise_filter=noise_level)
    print(result)
