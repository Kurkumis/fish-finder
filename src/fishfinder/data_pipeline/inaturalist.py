"""Thin client for the public iNaturalist REST API (v1)."""
import re
import time
from pathlib import Path

import requests
from PIL import Image

API_BASE = "https://api.inaturalist.org/v1"
USER_AGENT = "fish-finder/0.1 (pet project; https://github.com/Kurkumis/fish-finder)"
# iNaturalist asks API users to stay at roughly one request per second.
REQUEST_DELAY_S = 1.0

# Photo licenses that allow reuse (at least for non-commercial purposes).
# Photos under "all rights reserved" or no-derivatives licenses are excluded.
OPEN_PHOTO_LICENSES = ("cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa")

# The observations endpoint cannot be paged beyond this many results.
MAX_PAGINATED_RESULTS = 10_000


def _get(endpoint: str, params: dict, timeout: float = 30.0) -> dict:
    """Send a GET request to the iNaturalist API and return the parsed JSON."""
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()  # raise an exception on HTTP errors (4xx, 5xx)
    time.sleep(REQUEST_DELAY_S)  # be polite: throttle requests
    return response.json()


def photo_url(url: str, size: str = "large") -> str:
    """Convert a photo URL from the API to another size.

    The API returns thumbnail URLs such as .../photos/123/square.jpg.
    Available sizes: square, small, medium, large (1024 px), original.
    """
    return re.sub(r"/(square|small|medium|large|original)\.", f"/{size}.", url)


def fetch_project_species_counts(
    project: str,
    quality_grade: str = "research",
    locale: str = "ru",
    per_page: int = 200,
) -> list[dict]:
    """Return every taxon observed in a project, with its observation count.

    Uses the /observations/species_counts endpoint, which aggregates
    observations by taxon. `project` can be a project ID or its URL slug.
    `locale` controls the language of common names.

    Each returned row has the keys:
    taxon_id, scientific_name, common_name, rank, observations.
    """
    rows = []
    page = 1
    while True:
        payload = _get(
            "observations/species_counts",
            {
                "project_id": project,
                "quality_grade": quality_grade,
                "locale": locale,
                "per_page": per_page,
                "page": page,
            },
        )
        results = payload.get("results", [])
        for item in results:
            taxon = item["taxon"]
            rows.append(
                {
                    "taxon_id": taxon["id"],
                    "scientific_name": taxon.get("name"),
                    "common_name": taxon.get("preferred_common_name"),
                    "rank": taxon.get("rank"),
                    "observations": item["count"],
                }
            )
        # Stop when the page is empty or we have fetched everything.
        if not results or page * per_page >= payload.get("total_results", 0):
            break
        page += 1
    return rows


def count_observations(
    taxon_id: int,
    project: str | None = None,
    quality_grade: str = "research",
    photo_licenses: tuple[str, ...] = OPEN_PHOTO_LICENSES,
) -> int:
    """Count observations of a taxon (descendant taxa included).

    If `project` is given, only observations belonging to that project are
    counted; otherwise the count covers the whole world.
    """
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality_grade,
        "photo_license": ",".join(photo_licenses),
        "per_page": 1,  # we only need the total, not the observations themselves
    }
    if project is not None:
        params["project_id"] = project
    payload = _get("observations", params)
    return payload["total_results"]


def fetch_observations(
    taxon_id: int,
    project: str,
    quality_grade: str = "research",
    photo_licenses: tuple[str, ...] = OPEN_PHOTO_LICENSES,
    per_page: int = 200,
) -> list[dict]:
    """Fetch all observations of a taxon (descendants included) in a project.

    Only observations with at least one openly licensed photo are requested.
    Returns the raw JSON objects from the API.
    """
    observations = []
    page = 1
    while True:
        payload = _get(
            "observations",
            {
                "taxon_id": taxon_id,
                "project_id": project,
                "quality_grade": quality_grade,
                "photo_license": ",".join(photo_licenses),
                "photos": "true",
                "order_by": "id",
                "order": "asc",
                "per_page": per_page,
                "page": page,
            },
        )
        total = payload.get("total_results", 0)
        if total > MAX_PAGINATED_RESULTS:
            raise RuntimeError(
                f"Taxon {taxon_id} has {total} observations; "
                f"the API cannot page beyond {MAX_PAGINATED_RESULTS}."
            )
        results = payload.get("results", [])
        observations.extend(results)
        if not results or len(observations) >= total:
            break
        page += 1
    return observations


def parse_observation(
    obs: dict,
    photo_licenses: tuple[str, ...] = OPEN_PHOTO_LICENSES,
) -> dict | None:
    """Reduce a raw API observation to the fields we need.

    The license is checked per photo: photos of one observation may have
    different licenses. Returns None if no usable photo remains.
    """
    photos = []
    for position, photo in enumerate(obs.get("photos", [])):
        license_code = (photo.get("license_code") or "").lower()
        if license_code not in photo_licenses or not photo.get("url"):
            continue
        photos.append(
            {
                "photo_id": photo["id"],
                "position": position,  # 0 is the observation's primary photo
                "license": license_code,
                "attribution": photo.get("attribution"),
                "url": photo_url(photo["url"], "large"),
            }
        )
    if not photos:
        return None

    user = obs.get("user") or {}
    taxon = obs.get("taxon") or {}
    return {
        "observation_id": obs["id"],
        "observer_id": user.get("id"),
        "observer_login": user.get("login"),
        "observed_on": obs.get("observed_on"),
        "observed_taxon_id": taxon.get("id"),
        "observed_taxon_name": taxon.get("name"),
        "photos": photos,
    }


def download_image(
    url: str,
    dest: Path,
    session: requests.Session,
    retries: int = 3,
    timeout: float = 30.0,
) -> str | None:
    """Download an image to `dest`.

    Returns None on success, or the last error message if every attempt failed.
    The file is first written under a temporary name and verified, so a broken
    download never ends up at `dest`.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            tmp.write_bytes(response.content)
            with Image.open(tmp) as img:
                img.verify()  # raises an exception if the file is not a valid image
            tmp.replace(dest)
            return None
        except (requests.RequestException, OSError, SyntaxError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)  # back off before retrying: 2 s, 4 s, ...
    tmp.unlink(missing_ok=True)
    return error
