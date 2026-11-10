"""Selection of observations for the training set."""
import random
from collections import Counter


def sample_observations(
    observations: list[dict],
    max_observations: int,
    max_per_observer: int,
    seed: int = 42,
) -> list[dict]:
    """Randomly pick observations, limiting how many come from one observer.

    The list is sorted by observation ID first, so the result depends only on
    the seed and not on the order in which the API returned the data.
    """
    pool = sorted(observations, key=lambda o: o["observation_id"])
    random.Random(seed).shuffle(pool)

    selected = []
    per_observer = Counter()
    for obs in pool:
        if len(selected) >= max_observations:
            break
        if per_observer[obs["observer_id"]] >= max_per_observer:
            continue
        per_observer[obs["observer_id"]] += 1
        selected.append(obs)
    return selected
