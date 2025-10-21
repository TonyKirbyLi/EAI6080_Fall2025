import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login, snapshot_download

HF_TOKEN = "" 

BASE_DIR = Path(r"")
BASE_DIR.mkdir(parents=True, exist_ok=True)

DS_BIXBENCH = "futurehouse/BixBench"

def ensure_hf_login():
    """Login with token if provided"""
    if HF_TOKEN and HF_TOKEN.startswith("hf_"):
        try:
            login(token=HF_TOKEN, add_to_git_credential=True)
            print("[HF] Logged in successfully.")
        except Exception as e:
            print(f"[HF] Login failed: {e}")
    else:
        print("[HF] No token set (will try anonymous access).")

def robust_load(repo_id, split_candidates=("train", "test", "validation")):
    for sp in split_candidates:
        try:
            ds = load_dataset(repo_id, split=sp)
            print(f"[OK] Loaded {repo_id}:{sp} ({len(ds)} rows)")
            return ds, sp
        except Exception:
            continue
    d = load_dataset(repo_id)
    sp_name, ds = next(iter(d.items()))
    print(f"[OK] Loaded {repo_id}:{sp_name} ({len(ds)} rows)")
    return ds, sp_name

def export_table(ds, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ds.to_pandas()
    parquet = out_dir / f"{name}.parquet"
    csv = out_dir / f"{name}.csv"
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"[Saved] {parquet}")
    print(f"[Saved] {csv}")

def main():
    print("=== BixBench Dataset Downloader ===\n") 
    ensure_hf_login()
    bix_repo_dir = BASE_DIR / "BixBench_repo"
    bix_repo_dir.mkdir(exist_ok=True)
    
    print("[BixBench] Downloading full snapshot...")
    try:
        snapshot_download(
            repo_id=DS_BIXBENCH,
            repo_type="dataset",
            local_dir=str(bix_repo_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["*.json*", "*.parquet", "*.zip", "*.md", "README*"]
        )
        print("[BixBench] Snapshot download completed!")
    except Exception as e:
        print(f"[ERROR] BixBench snapshot download failed: {e}")
        return

    try:
        ds_bix, split_bix = robust_load(DS_BIXBENCH)
        export_table(ds_bix, BASE_DIR / "BixBench", f"BixBench_{split_bix}")
        print(f"[BixBench] Dataset exported successfully as BixBench_{split_bix}")
    except Exception as e:
        print(f"[ERROR] Failed to load and export BixBench: {e}")
        return

    print(f"\n[✅ Done] BixBench dataset saved under: {BASE_DIR}")
    print(f"Repository files: {bix_repo_dir}")
    print(f"Processed data: {BASE_DIR / 'BixBench'}")

if __name__ == "__main__":
    main()
