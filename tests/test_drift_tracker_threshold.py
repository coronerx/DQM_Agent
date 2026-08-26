"""
tests/test_drift_tracker_threshold.py — regression coverage for a real,
repeatedly-confirmed bug: looks_like_sustained_drift essentially never
fired, on either synthetic or real CMS data, even given an obvious,
agent-identified +3.4 GeV drift over 12 chunks. Two compounding causes:

  1. A gating condition additionally required len(history) >= 10 chunks
     before the boolean could be True at all, regardless of trend strength.
  2. The threshold compared the slope against the RAW standard deviation of
     a window that included the drifting values themselves -- a real trend
     inflates its own yardstick, making the test hardest to trigger exactly
     when there's real drift to catch.

Fixed by switching to a linear regression significance test (slope
p-value from the residual scatter around the fitted trend line, not the
raw variance of a trending window).

Run with: PYTHONPATH=. pytest tests/test_drift_tracker_threshold.py -v
"""

import numpy as np

from monitors.drift_tracker import record_and_check, reset


def test_real_trajectory_now_correctly_flags_starting_at_chunk_5():
    """The actual Z-peak mean mass sequence from a real 12-chunk agent run
    (90.478 -> 93.845 GeV). The old code never flagged this across all 12
    chunks. The fix should flag it as early as statistically possible --
    chunk 5, the first chunk with enough history to assess a trend at all."""
    reset("test_real_trajectory")
    trajectory = [90.478, 90.510, 90.632, 90.785, 91.030, 91.577,
                  91.447, 91.617, 92.545, 92.702, 93.541, 93.845]

    first_flagged_at = None
    for i, v in enumerate(trajectory):
        result = record_and_check("test_real_trajectory", v)
        if result["trend_detectable"] and result["looks_like_sustained_drift"]:
            first_flagged_at = i + 1  # 1-indexed chunk number
            break

    assert first_flagged_at == 5


def test_false_positive_rate_on_flat_noisy_data_is_close_to_nominal():
    """Across many independent 12-chunk sequences with NO real trend (flat
    mean, pure noise), the flag rate should be close to the nominal 5%,
    not systematically higher (which would make it cry wolf) or zero
    (which would mean it never fires at all, the original bug)."""
    n_trials = 200
    false_positives = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        reset("test_flat")
        last_result = None
        for _ in range(12):
            v = rng.normal(91.0, 0.3)
            last_result = record_and_check("test_flat", v)
        if last_result["looks_like_sustained_drift"]:
            false_positives += 1

    rate = false_positives / n_trials
    assert 0.01 < rate < 0.12  # generous band around nominal 5%, not the old ~0%
