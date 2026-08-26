"""
tests/test_occupancy.py — fast, free, deterministic tests for the
Poisson/binomial occupancy tests. No API calls.

Run with: PYTHONPATH=. pytest tests/test_occupancy.py -v
"""

import pytest

from monitors.occupancy import poisson_count_test, binomial_proportion_test


def test_poisson_test_passes_normal_fluctuation():
    result = poisson_count_test(observed_n=598, expected_n=600)
    assert result["p_value"] > 0.05


def test_poisson_test_flags_severe_drop():
    """A dead-channel-scale drop (30 out of an expected 600) should be
    caught overwhelmingly -- this is the actual scenario the old
    'ratio < 0.5' heuristic was trying to approximate, now done properly."""
    result = poisson_count_test(observed_n=30, expected_n=600)
    assert result["p_value"] < 1e-10


def test_binomial_test_passes_normal_share_fluctuation():
    """Reference: 202/600 events in some region (p ~ 0.337). A chunk with
    198/600 in that region is normal sampling variation."""
    result = binomial_proportion_test(observed_k=198, observed_n=600, reference_p=202 / 600)
    assert result["p_value"] > 0.05


def test_binomial_test_flags_real_migration():
    """A chunk where the region's share has genuinely collapsed (120/600
    vs a reference share of ~0.337, i.e. ~202/600 expected) should be
    caught clearly."""
    result = binomial_proportion_test(observed_k=120, observed_n=600, reference_p=202 / 600)
    assert result["p_value"] < 0.001


def test_binomial_test_regression_chunk4_case_is_not_significant():
    """
    Regression test tied to a real finding: in an actual agent run, the
    model described chunk_4's Z-peak event count (189/600 vs a reference of
    202/600) as a 'concern' worth flagging, based on eyeballing the raw
    numbers rather than running a real test. This test confirms that case
    is NOT statistically significant (p > 0.05) -- i.e. it validates that
    the model's informal concern was not backed by real evidence, and
    confirms this tool would correctly tell it so if it had been called.
    """
    result = binomial_proportion_test(observed_k=189, observed_n=600, reference_p=202 / 600)
    assert result["p_value"] > 0.05


def test_poisson_and_binomial_answer_different_questions():
    """A chunk with a completely NORMAL total count can still have an
    ABNORMAL regional share -- confirms the two tests are not redundant
    with each other, which is the whole reason both exist."""
    total_result = poisson_count_test(observed_n=600, expected_n=600)
    share_result = binomial_proportion_test(observed_k=50, observed_n=600, reference_p=202 / 600)

    assert total_result["p_value"] > 0.9  # total count is exactly as expected
    assert share_result["p_value"] < 0.001  # but the region's share has collapsed
