"""Count openly licensed research-grade observations for every class in classes.yaml."""
import argparse

import pandas as pd
from tqdm import tqdm

from fishfinder.data_pipeline.inaturalist import count_observations
from fishfinder.utils.config import load_classes
from fishfinder.utils.paths import INTERIM_DIR

DEFAULT_PROJECT = "ryby-rossii-fish-of-russia"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="project ID or slug")
    args = parser.parse_args()

    classes = load_classes()
    n_taxa = sum(len(c["taxa"]) for c in classes)

    rows = []
    with tqdm(total=n_taxa, desc="taxa") as progress:
        for cls in classes:
            for taxon in cls["taxa"]:
                taxon_id = taxon["taxon_id"]
                rows.append(
                    {
                        "class_id": cls["id"],
                        "taxon_id": taxon_id,
                        "scientific_name": taxon["scientific_name"],
                        "in_project": count_observations(taxon_id, project=args.project),
                        "worldwide": count_observations(taxon_id),
                    }
                )
                progress.update(1)

    per_taxon = pd.DataFrame(rows)
    # Sum taxa into classes; the worldwide count includes the project's observations.
    per_class = per_taxon.groupby("class_id", sort=False)[["in_project", "worldwide"]].sum()
    per_class["outside_project"] = per_class["worldwide"] - per_class["in_project"]

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    per_taxon.to_csv(INTERIM_DIR / "taxon_observation_counts.csv", index=False)
    per_class.to_csv(INTERIM_DIR / "class_observation_counts.csv")

    print(per_class.to_string())


if __name__ == "__main__":
    main()