"""Thin client for the public iNaturalist REST API (v1)."""
import time

import requests

API_BASE = "https://api.inaturalist.org/v1"
# A descriptive User-Agent is good API etiquette: it tells the service who is calling.
USER_AGENT = "fish-finder/0.1 (pet project; https://github.com/Kurkumis/fish-finder)"
# iNaturalist asks API users to stay at roughly one request per second.
REQUEST_DELAY_S = 1.0


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