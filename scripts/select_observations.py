"""Select observations for every class and write a photo-level metadata table."""
import argparse
from pathlib import Path

import pandas as pd

from fishfinder.data_pipeline.inaturalist import fetch_observations, parse_observation
from fishfinder.data_pipeline.sampling import sample_observations
from fishfinder.utils.config import load_classes
from fishfinder.utils.paths import INAT_RAW_DIR

DEFAULT_PROJECT = "ryby-rossii-fish-of-russia"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="project ID or slug")
    parser.add_argument("--max-observations", type=int, default=300,
                        help="maximum number of observations per class")
    parser.add_argument("--max-per-observer", type=int, default=10,
                        help="maximum observations from one observer within a class")
    parser.add_argument("--max-photos-per-obs", type=int, default=3,
                        help="maximum photos taken from one observation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    summary = []
    for cls in load_classes():
        # 1. Collect every candidate observation of the class (all its taxa).
        # A dict keyed by observation ID removes duplicates.
        candidates = {}
        for taxon in cls["taxa"]:
            for raw in fetch_observations(taxon["taxon_id"], project=args.project):
                parsed = parse_observation(raw)
                if parsed is None:
                    continue
                parsed["taxon_id"] = taxon["taxon_id"]
                parsed["scientific_name"] = taxon["scientific_name"]
                candidates[parsed["observation_id"]] = parsed

        # 2. Random sample with a cap per observer.
        selected = sample_observations(
            list(candidates.values()),
            max_observations=args.max_observations,
            max_per_observer=args.max_per_observer,
            seed=args.seed,
        )

        # 3. Flatten to one row per photo.
        n_photos = 0
        for obs in selected:
            for photo in obs["photos"][: args.max_photos_per_obs]:
                extension = Path(photo["url"]).suffix or ".jpg"
                rows.append(
                    {
                        "class_id": cls["id"],
                        "taxon_id": obs["taxon_id"],
                        "scientific_name": obs["scientific_name"],
                        "observation_id": obs["observation_id"],
                        "observer_id": obs["observer_id"],
                        "observer_login": obs["observer_login"],
                        "observed_on": obs["observed_on"],
                        "photo_id": photo["photo_id"],
                        "position": photo["position"],
                        "license": photo["license"],
                        "attribution": photo["attribution"],
                        "observation_url": f"https://www.inaturalist.org/observations/{obs['observation_id']}",
                        "source_url": photo["url"],
                        # Path relative to data/raw/inaturalist/.
                        "local_path": f"images/{cls['id']}/{photo['photo_id']}{extension}",
                    }
                )
                n_photos += 1

        summary.append(
            {
                "class_id": cls["id"],
                "candidates": len(candidates),
                "observations": len(selected),
                "photos": n_photos,
                "observers": len({o["observer_id"] for o in selected}),
            }
        )
        print(f"{cls['id']}: {len(selected)} of {len(candidates)} observations selected")

    INAT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INAT_RAW_DIR / "photos.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary)
    print()
    print(summary_df.to_string(index=False))
    print(f"\nTotal photos: {len(rows)}. Saved to {out_path}")


if __name__ == "__main__":
    main()
