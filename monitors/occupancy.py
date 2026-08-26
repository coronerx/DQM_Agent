"""
occupancy.py — statistically correct occupancy tests, replacing the earlier
naive "ratio < 0.5" heuristic.

Two distinct questions, two distinct distributions:

  - TOTAL occupancy: "is the total event count in this chunk consistent
    with the expected rate?" Event arrivals in a fixed-size chunk are a
    counting process -- Poisson is the natural model (variance = mean),
    not a fixed threshold on the ratio to the reference count.

  - REGIONAL occupancy: "is the SHARE of events falling in a specific
    region (e.g. the Z-peak window) consistent with the reference share?"
    Each event independently either falls in the region or not -- a
    Bernoulli trial per event -- so the in-region count out of the chunk
    total is Binomial, not Poisson. This correctly separates "the chunk
    just has fewer events overall" (a total-occupancy problem) from
    "events are migrating out of this specific region while the total
    stays normal" (a regional problem) -- two genuinely different
    detector failure modes that a single ratio check conflates.
"""

import numpy as np
from scipy import stats


def poisson_count_test(observed_n: int, expected_n: float) -> dict:
    """Two-sided Poisson test: is observed_n consistent with a Poisson
    process whose mean is expected_n (taken from the reference chunk)?"""
    if expected_n <= 0:
        return {
            "test": "poisson_count", "observed_n": observed_n,
            "expected_n": expected_n, "p_value": float("nan"),
            "error": "expected_n must be positive",
        }

    # Two-sided p-value: 2 * min(P(X <= observed), P(X >= observed)), capped at 1.
    p_le = stats.poisson.cdf(observed_n, expected_n)
    p_ge = stats.poisson.sf(observed_n - 1, expected_n)
    p_value = min(1.0, 2 * min(p_le, p_ge))

    z_score = (observed_n - expected_n) / np.sqrt(expected_n)

    return {
        "test": "poisson_count",
        "observed_n": int(observed_n),
        "expected_n": float(expected_n),
        "z_score": round(float(z_score), 3),
        "p_value": float(p_value),
    }


def binomial_proportion_test(observed_k: int, observed_n: int, reference_p: float) -> dict:
    """Two-sided exact binomial test: given observed_n total events this
    chunk, is the count observed_k falling in some region consistent with
    the reference proportion reference_p (events-in-region / total, taken
    from the reference chunk)?"""
    if observed_n <= 0:
        return {
            "test": "binomial_proportion", "observed_k": observed_k,
            "observed_n": observed_n, "reference_p": reference_p,
            "p_value": float("nan"), "error": "observed_n must be positive",
        }

    result = stats.binomtest(observed_k, observed_n, reference_p, alternative="two-sided")
    observed_p = observed_k / observed_n

    return {
        "test": "binomial_proportion",
        "observed_k": int(observed_k),
        "observed_n": int(observed_n),
        "observed_proportion": round(observed_p, 4),
        "reference_proportion": round(reference_p, 4),
        "p_value": float(result.pvalue),
    }
