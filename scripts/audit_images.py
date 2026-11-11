"""Check downloaded photos integrity, size and flag near-duplicates."""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from fishfinder.data_pipeline.audit import cluster_pairs, find_near_duplicates, inspect_image
from fishfinder.utils.paths import INAT_RAW_DIR, INTERIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-side", type=int, default=256,
                        help="photos with a shorter side below this are dropped")
    parser.add_argument("--max-hash-distance", type=int, default=10,
                        help="max differing bits out of 256 for two photos to count as duplicates")
    args = parser.parse_args()

    df = pd.read_csv(INAT_RAW_DIR / "photos.csv")

    infos = [inspect_image(INAT_RAW_DIR / p) for p in tqdm(df["local_path"], desc="inspecting")]
    df["ok"] = [i["ok"] for i in infos]
    df["width"] = [i["width"] for i in infos]
    df["height"] = [i["height"] for i in infos]
    df["file_bytes"] = [i["file_bytes"] for i in infos]
    df["pixels"] = df["width"] * df["height"]
    df["too_small"] = df[["width", "height"]].min(axis=1) < args.min_side # checking if image is of intended size

    # Near-duplicate detection among the images that could be decoded.
    ok_idx = np.flatnonzero(df["ok"].to_numpy())
    hashes = np.stack([infos[i]["dhash"] for i in ok_idx])
    pairs = find_near_duplicates(hashes, args.max_hash_distance)
    labels = cluster_pairs(len(ok_idx), pairs)

    df["dup_cluster"] = -1
    df.loc[ok_idx, "dup_cluster"] = labels

    # In each cluster keep one photo: prefer the observation's primary photo
    # (position 0), then the larger image, then the lower photo ID.
    ranked = df[df["ok"]].sort_values(
        ["position", "pixels", "photo_id"], ascending=[True, False, True]
    )
    is_dup = ranked.duplicated("dup_cluster", keep="first").to_numpy()
    df["is_duplicate"] = False
    df.loc[ranked.index[is_dup], "is_duplicate"] = True
    keeper = ranked.drop_duplicates("dup_cluster", keep="first").set_index("dup_cluster")["photo_id"]
    df["duplicate_of"] = df["dup_cluster"].map(keeper).where(df["is_duplicate"])

    df["keep"] = df["ok"] & ~df["too_small"] & ~df["is_duplicate"]

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_DIR / "photos_audit.csv", index=False)

    summary = df.groupby("class_id", sort=False).agg(
        total=("photo_id", "size"),
        broken=("ok", lambda s: int((~s).sum())),
        too_small=("too_small", "sum"),
        duplicates=("is_duplicate", "sum"),
        kept=("keep", "sum"),
    )
    print(summary.to_string())
    print(f"\nTotal: {len(df)} photos, kept: {int(df['keep'].sum())}")
    print("\nImage side lengths (pixels):")
    print(df[["width", "height"]].describe().round(0).to_string())


if __name__ == "__main__":
    main()