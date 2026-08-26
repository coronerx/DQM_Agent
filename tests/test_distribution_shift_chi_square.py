"""
tests/test_distribution_shift_chi_square.py — regression coverage for TWO
real bugs found via a live 12-chunk agent run on real CMS data + injected
momentum drift, both in the same function:

  1. A single stray event landing in a mass bin where the REFERENCE had
     zero events inflated the chi-square statistic to ~1,000,000 from one
     data point (the original code floored expected counts at 1e-6, giving
     (1 - 1e-6)^2 / 1e-6 ~= 1e6). The agent spent multiple chunks treating
     this as a second, independent "catastrophic bin overflow" anomaly
     separate from the real momentum-scale drift it was also correctly
     tracking -- when it was very likely this artifact the whole time.

  2. Even without any stray events, the test's FALSE-POSITIVE RATE was
     measured at ~50% instead of the nominal ~5%, because the original
     implementation used scipy's one-sample goodness-of-fit test (treats
     the reference histogram as fixed/known with zero uncertainty) when
     the reference is actually a finite random sample with its own
     sampling noise. Fixed by switching to a proper two-sample chi-square
     test of homogeneity.

Run with: PYTHONPATH=. pytest tests/test_distribution_shift_chi_square.py -v
"""

import numpy as np

from monitors.distribution_shift import chi_square_test


def test_single_stray_event_in_empty_reference_bin_does_not_blow_up():
    """The exact reproduction of bug #1: reference has zero events in some
    tail bin, chunk has exactly one event there. Old code: chi2 statistic
    ~1,000,000 from a single data point. Fixed code: stays sane."""
    rng = np.random.default_rng(0)
    reference = rng.exponential(15, 1500)
    reference = reference[reference < 100]

    chunk = reference.copy()[:1499]
    chunk = np.append(chunk, reference.max() + 20)  # one event beyond reference's range

    result = chi_square_test("test", chunk, reference)

    assert result["statistic"] < 100  # sane, not ~1e6
    assert result["p_value"] > 0.5  # one stray event out of 1500 should not read as significant


def test_false_positive_rate_is_close_to_nominal_alpha():
    """Regression test for bug #2: across many trials where H0 is TRUE
    (both samples genuinely drawn from the same distribution), the
    fraction flagged at alpha=0.05 should be close to 5%, not ~50%."""
    n_trials = 200
    false_positives = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed + 1000)
        reference = rng.normal(90, 3, 2000)
        chunk = rng.normal(90, 3, 2000)
        result = chi_square_test("test", chunk, reference)
        if result["p_value"] < 0.05:
            false_positives += 1

    rate = false_positives / n_trials
    # Generous band (1%-12%) around the nominal 5% -- this is itself a
    # random quantity across 200 binomial trials, not testing for an exact
    # match, just confirming it's in the right ballpark and nowhere near
    # the ~50% the old code produced.
    assert 0.01 < rate < 0.12


def test_matching_distributions_still_pass_cleanly_with_the_fix():
    """Confirms the fix isn't overcorrecting into being too lenient --
    two genuinely different distributions should still fail decisively."""
    rng = np.random.default_rng(1)
    reference = rng.normal(90, 3, 1500)
    shifted_chunk = rng.normal(95, 3, 1500)  # clearly shifted

    result = chi_square_test("test", shifted_chunk, reference)
    assert result["p_value"] < 0.001
