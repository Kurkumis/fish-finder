"""Download the photos listed in data/raw/inaturalist/photos.csv."""
import argparse
import time

import pandas as pd
import requests
from tqdm import tqdm

from fishfinder.data_pipeline.inaturalist import USER_AGENT, download_image
from fishfinder.utils.paths import INAT_RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="pause in seconds after each downloaded photo")
    parser.add_argument("--limit", type=int, default=None,
                        help="download only a random subset of N photos for testing")
    args = parser.parse_args()

    df = pd.read_csv(INAT_RAW_DIR / "photos.csv")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    failures = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="photos"):
        dest = INAT_RAW_DIR / row.local_path
        if dest.exists():
            continue  # already downloaded: this makes the script resumable
        dest.parent.mkdir(parents=True, exist_ok=True)
        error = download_image(row.source_url, dest, session)
        if error:
            failures.append({"photo_id": row.photo_id, "url": row.source_url, "error": error})
        else:
            time.sleep(args.delay)  # throttle to stay polite to the image server

    print(f"Done. Failed downloads: {len(failures)}")
    if failures:
        failures_path = INAT_RAW_DIR / "download_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Failure details: {failures_path}")


if __name__ == "__main__":
    main()
