"""Build the final dataset table with train/val/test splits."""
import pandas as pd

from fishfinder.data_pipeline.splits import assign_folds, fold_to_split
from fishfinder.utils.config import load_classes
from fishfinder.utils.paths import INTERIM_DIR, PROCESSED_DIR


def main() -> None:
    audit = pd.read_csv(INTERIM_DIR / "photos_audit.csv")
    df = audit[audit["keep"]].reset_index(drop=True)

    # A missing observer ID would break the grouping; give such rows a unique group.
    df["observer_id"] = df["observer_id"].fillna(-df["observation_id"]).astype(int)

    class_ids = [c["id"] for c in load_classes()]
    df["class_index"] = df["class_id"].map({c: i for i, c in enumerate(class_ids)})
    df["fold"] = assign_folds(df)
    df["split"] = df["fold"].map(fold_to_split)

    # Safety check: no observer may appear in more than one split.
    assert df.groupby("observer_id")["split"].nunique().max() == 1, "observer leakage!"

    columns = [
        "photo_id", "observation_id", "observer_id", "class_id", "class_index",
        "local_path", "fold", "split", "license", "attribution", "observation_url",
        "taxon_id", "scientific_name", "position",
    ]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(PROCESSED_DIR / "dataset.csv", index=False)

    counts = pd.crosstab(df["class_id"], df["split"])[["train", "val", "test"]]
    counts = counts.loc[class_ids]  # keep the order from classes.yaml
    counts["total"] = counts.sum(axis=1)
    print(counts.to_string())
    shares = (counts[["train", "val", "test"]].sum() / len(df) * 100).round(1)
    print(f"\nTotal: {len(df)} photos. Split shares, %:\n{shares.to_string()}")


if __name__ == "__main__":
    main()