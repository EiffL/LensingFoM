"""Push parquet shards from the Modal volume to HuggingFace Hub.

Usage:
    modal run scripts/push_hf_dataset.py
    modal run scripts/push_hf_dataset.py --lmax 200  # push only one lmax

Requires a Modal secret named 'huggingface-secret' with key HF_TOKEN.
Create it with:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
"""

from pathlib import Path

import modal

LMAX_VALUES = [200, 400, 600, 800, 1000]

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
def push_to_hub(repo_id: str = "EiffL/GowerStreetDESY3", lmax_filter: int = None) -> dict:
    """Push parquet shards to HuggingFace Hub."""
    import os
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    hf_dir = Path(RESULTS_DIR) / "hf_dataset"
    lmax_list = [lmax_filter] if lmax_filter else LMAX_VALUES
    pushed = {}

    for lmax in lmax_list:
        lmax_dir = hf_dir / f"lmax_{lmax}"
        if not lmax_dir.exists():
            print(f"lmax={lmax}: no parquet shards found, skipping")
            continue

        parquet_files = sorted(lmax_dir.glob("shard_*.parquet"))
        if not parquet_files:
            print(f"lmax={lmax}: no parquet files, skipping")
            continue

        for pf in parquet_files:
            api.upload_file(
                path_or_fileobj=str(pf),
                path_in_repo=f"data/lmax_{lmax}/{pf.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"lmax={lmax}: uploaded {pf.name}")

        pushed[lmax] = len(parquet_files)
        print(f"lmax={lmax}: pushed {len(parquet_files)} shards")

    return {"repo_id": repo_id, "pushed": pushed}


@app.local_entrypoint()
def main(lmax: int = None):
    result = push_to_hub.remote(lmax_filter=lmax)
    print(result)
