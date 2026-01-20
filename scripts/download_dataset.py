from pathlib import Path
from huggingface_hub import snapshot_download
import shutil


def main():
    # 1) Download repo snapshot
    local_dir = Path("../data/point_bridge_raw")
    snapshot_download(
        repo_id="siddhanthaldar/Point-Bridge",
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    # 2) Move mimicgen_data -> ../data/mimicgen_data
    src = local_dir / "mimicgen_data"
    dst = Path("../data/mimicgen_data")

    # Make sure parent exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If dst already exists and you want to overwrite, remove it first
    if dst.exists():
        shutil.rmtree(dst)

    src.rename(dst)  # move the directory

    # 3) Remove the remaining snapshot directory
    shutil.rmtree(local_dir)


if __name__ == "__main__":
    main()
