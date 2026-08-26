"""
distribution_shift.py — "does this chunk's distribution still look like the
reference distribution?"
"""

import numpy as np
from scipy import stats


def ks_test(channel: str, chunk_values, reference_values) -> dict:
    statistic, p_value = stats.ks_2samp(chunk_values, reference_values)
    return {
        "channel": channel,
        "test": "ks_2samp",
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def chi_square_test(channel: str, chunk_values, reference_values, bins: int = 10) -> dict:
    chunk_values = np.asarray(chunk_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    combined_min = min(chunk_values.min(), reference_values.min())
    combined_max = max(chunk_values.max(), reference_values.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    chunk_hist, _ = np.histogram(chunk_values, bins=bin_edges)
    ref_hist, _ = np.histogram(reference_values, bins=bin_edges)

    # TWO-SAMPLE chi-square test of homogeneity, not scipy's one-sample
    # goodness-of-fit test. This matters: scipy.stats.chisquare(f_obs,
    # f_exp) treats f_exp as fixed/known with zero uncertainty, but here
    # the "expected" reference histogram is ITSELF a finite random sample
    # with its own sampling noise. Treating it as noiseless understates
    # the true variance and badly inflates the false-positive rate --
    # measured empirically at ~50% instead of the nominal ~5% before this
    # fix (found via a regression test after a real 12-chunk agent run).
    # The two-sample formulation below (standard homogeneity test for a
    # 2xk contingency table) also naturally handles a bin populated by
    # only one sample without any separate epsilon-floor or bin-exclusion
    # hack -- a single stray event in an otherwise-empty bin now
    # contributes a small, honest amount rather than ~1,000,000.
    N1 = chunk_hist.sum()
    N2 = ref_hist.sum()
    total = chunk_hist + ref_hist

    valid_bins = total > 0  # only exclude bins where NEITHER sample has anything
    n_excluded = int((~valid_bins).sum())

    O1 = chunk_hist[valid_bins].astype(float)
    O2 = ref_hist[valid_bins].astype(float)
    T = total[valid_bins].astype(float)

    E1 = N1 * T / (N1 + N2)
    E2 = N2 * T / (N1 + N2)

    statistic = float(np.sum((O1 - E1) ** 2 / E1) + np.sum((O2 - E2) ** 2 / E2))
    dof = int(valid_bins.sum()) - 1
    p_value = float(stats.chi2.sf(statistic, dof)) if dof > 0 else float("nan")

    return {
        "channel": channel,
        "test": "chi_square_two_sample",
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "bins_used": int(valid_bins.sum()),
        "bins_excluded_both_empty": n_excluded,
    }


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return {}

    max_significant_rank = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            max_significant_rank = i

    results = {}
    for i, (label, p) in enumerate(items, start=1):
        results[label] = {
            "p_value": p,
            "rank": i,
            "significant_after_correction": i <= max_significant_rank,
        }
    return results
