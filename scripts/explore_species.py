"""Download species counts of an iNaturalist project and save them as CSV."""
import argparse

import pandas as pd

from fishfinder.data_pipeline.inaturalist import fetch_project_species_counts
from fishfinder.utils.paths import INAT_RAW_DIR

DEFAULT_PROJECT = "ryby-rossii-fish-of-russia"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="project ID or slug")
    parser.add_argument("--top", type=int, default=100, help="how many rows to print")
    args = parser.parse_args()

    rows = fetch_project_species_counts(args.project)
    df = pd.DataFrame(rows)
    # Keep species only (drop genera, subspecies, hybrids) and sort by popularity.
    df = df[df["rank"] == "species"].sort_values("observations", ascending=False)
    df = df.reset_index(drop=True)

    INAT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INAT_RAW_DIR / "project_species_counts.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} species to {out_path}")
    print(df.head(args.top).to_string())


if __name__ == "__main__":
    main()