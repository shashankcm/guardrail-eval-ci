"""Interval estimates for the evaluation.

Rates (for example, the share of attacks blocked) use the Wilson score interval, which behaves
well for small samples and for rates near 0 or 1. Differences between two detectors on the same
items use a paired bootstrap of the per-item differences. Both are standard library only, so
the evaluation runs anywhere, including CI.
"""

import math
import random

Z_95 = 1.96


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a proportion, as (low, high)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half_width = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == trials else min(1.0, centre + half_width)
    return low, high


def bootstrap_mean_interval(
    values: list, seed: int = 0, n_resamples: int = 10_000, alpha: float = 0.05
) -> tuple:
    """Mean of the values and a percentile bootstrap interval, as (mean, low, high)."""
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    values = list(values)
    n = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int(alpha / 2 * n_resamples)]
    high = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return sum(values) / n, low, high
