"""
tests/test_dimuon_pipeline.py — cheap, deterministic tests for the dimuon
physics + statistics pipeline. NO Claude API calls anywhere in this file --
this is exactly the "layer 1" testing discussed earlier: run this constantly,
as many times as you want, for free.

Run with: PYTHONPATH=. pytest tests/test_dimuon_pipeline.py -v
"""

import numpy as np
import pytest

from dimuon.kinematics import invariant_mass, invariant_mass_array, MUON_MASS_GEV
from dimuon.generate import generate_chunk, sample_target_masses, Z_MASS_GEV
from monitors.distribution_shift import ks_test, benjamini_hochberg
from monitors.drift_tracker import record_and_check, reset as reset_drift


# ---------------------------------------------------------------------------
# 1. Physics ground truth: is the invariant mass formula itself correct?
# ---------------------------------------------------------------------------

def test_invariant_mass_hand_computed_case():
    """Two muons, both eta=0, back-to-back in phi, equal pT -- this is just
    the dimuon rest frame with zero total pT/pz, so the invariant mass has
    a simple closed form: m = 2 * sqrt(pt^2 + muon_mass^2)."""
    pt = 50.0
    m = invariant_mass(pt, 0.0, 0.0, pt, 0.0, np.pi)
    expected = 2 * np.sqrt(pt**2 + MUON_MASS_GEV**2)
    assert m == pytest.approx(expected, rel=1e-9)


def test_invariant_mass_nonnegative_for_random_inputs():
    """m_squared can go slightly negative from floating point noise near
    zero -- confirm the clip in kinematics.py actually prevents NaNs."""
    rng = np.random.default_rng(0)
    pt1, pt2 = rng.uniform(1, 100, 200), rng.uniform(1, 100, 200)
    eta1, eta2 = rng.uniform(-2.5, 2.5, 200), rng.uniform(-2.5, 2.5, 200)
    phi1, phi2 = rng.uniform(-np.pi, np.pi, 200), rng.uniform(-np.pi, np.pi, 200)
    masses = invariant_mass_array(pt1, eta1, phi1, pt2, eta2, phi2)
    assert np.all(np.isfinite(masses))
    assert np.all(masses >= 0)


# ---------------------------------------------------------------------------
# 2. Generator correctness: does the boosted construction actually preserve
#    the invariant mass it was built from? If this fails, every downstream
#    test is meaningless.
# ---------------------------------------------------------------------------

def test_generator_round_trip_recovers_target_mass():
    rng = np.random.default_rng(42)
    n = 500
    target_masses = sample_target_masses(n, rng, z_fraction=0.35)

    # Re-seed so generate_chunk's internal sampling doesn't consume the
    # same stream -- we just want to confirm invariant_mass(generated) ==
    # target for a fresh, independently generated batch.
    rng2 = np.random.default_rng(42)
    chunk = generate_chunk(n, rng2, momentum_scale=1.0)

    # true_mass is computed from the UNSCALED generated kinematics, so it
    # should match the sampled target masses (same rng seed/order).
    assert chunk["true_mass"] == pytest.approx(target_masses, rel=1e-6)


def test_generator_produces_a_visible_z_peak():
    """Not a precise check -- just confirms the mixture is doing roughly
    what it says: a meaningful fraction of events cluster near the Z mass."""
    rng = np.random.default_rng(1)
    chunk = generate_chunk(5000, rng, momentum_scale=1.0, z_fraction=0.35)
    near_z = np.abs(chunk["true_mass"] - Z_MASS_GEV) < 5.0  # within 5 GeV
    fraction_near_z = near_z.mean()
    assert fraction_near_z > 0.15  # well below the true 0.35 due to the wide Cauchy tails, but clearly peaked


# ---------------------------------------------------------------------------
# 3. The physics-to-calibration link: does scaling pT scale the reconstructed
#    mass by (very nearly) the same factor? This is the mechanism behind the
#    real technique of using the Z peak to calibrate muon momentum scale.
# ---------------------------------------------------------------------------

def test_momentum_miscalibration_shifts_reconstructed_mass():
    rng_ref = np.random.default_rng(7)
    rng_shifted = np.random.default_rng(7)  # same seed -> same underlying physics

    reference = generate_chunk(2000, rng_ref, momentum_scale=1.0)
    miscalibrated = generate_chunk(2000, rng_shifted, momentum_scale=1.02)

    ratio = np.mean(miscalibrated["reco_mass"]) / np.mean(reference["reco_mass"])
    # Massless approximation predicts ratio == 1.02 almost exactly, since
    # muon mass (~0.1 GeV) is negligible next to typical pT here.
    assert ratio == pytest.approx(1.02, abs=0.005)


# ---------------------------------------------------------------------------
# 4. Statistics layer: does the KS test correctly catch/ignore the shift?
# ---------------------------------------------------------------------------

def test_ks_test_flags_a_real_miscalibration():
    rng_ref = np.random.default_rng(11)
    rng_bad = np.random.default_rng(11)

    reference = generate_chunk(400, rng_ref, momentum_scale=1.0)
    miscalibrated = generate_chunk(400, rng_bad, momentum_scale=1.03)

    result = ks_test("dimuon_mass", miscalibrated["reco_mass"], reference["reco_mass"])
    assert result["p_value"] < 0.05


def test_ks_test_does_not_flag_matching_calibration():
    """Two independently sampled chunks at the SAME (correct) calibration
    should not typically be flagged as different from each other."""
    rng_a = np.random.default_rng(21)
    rng_b = np.random.default_rng(99)  # different seed -> independent sample

    chunk_a = generate_chunk(400, rng_a, momentum_scale=1.0)
    chunk_b = generate_chunk(400, rng_b, momentum_scale=1.0)

    result = ks_test("dimuon_mass", chunk_b["reco_mass"], chunk_a["reco_mass"])
    assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# 5. Multiple-testing correction, applied across several simulated detector
#    regions (e.g. rapidity bins), only one of which is actually miscalibrated.
# ---------------------------------------------------------------------------

def test_bh_correction_flags_only_the_true_miscalibration():
    reference_rng = np.random.default_rng(50)
    reference = generate_chunk(400, reference_rng, momentum_scale=1.0)

    pvalues = {}
    region_seeds = {"region_0": 51, "region_1": 52, "region_2": 53,
                     "region_3": 54, "region_4": 55}
    miscalibrated_region = "region_2"

    for region, seed in region_seeds.items():
        scale = 1.03 if region == miscalibrated_region else 1.0
        chunk = generate_chunk(400, np.random.default_rng(seed), momentum_scale=scale)
        result = ks_test(region, chunk["reco_mass"], reference["reco_mass"])
        pvalues[region] = result["p_value"]

    corrected = benjamini_hochberg(pvalues, alpha=0.05)

    assert corrected[miscalibrated_region]["significant_after_correction"] is True
    for region in region_seeds:
        if region != miscalibrated_region:
            assert corrected[region]["significant_after_correction"] is False


# ---------------------------------------------------------------------------
# 6. Drift tracker: a slow, developing momentum-scale drift across chunks.
# ---------------------------------------------------------------------------

def test_drift_tracker_slope_rises_with_gradual_miscalibration_drift():
    """
    Track the Z-PEAK mean, not the whole-spectrum mean.

    First version of this test tracked mean(reco_mass) across the full
    20-140 GeV spectrum and failed: the full spectrum has std ~31 GeV (wide
    background continuum + heavy Breit-Wigner tails), giving a per-chunk
    standard error (~31/sqrt(300) ~ 1.8 GeV) larger than the drift signal
    itself (0.3-1.1 GeV over the injected 0.5%-2% miscalibration). The
    monitored quantity was too noisy to see the effect -- which is exactly
    why real calibration monitoring tracks the Z peak position specifically
    (a narrow window around 91 GeV) rather than a whole-spectrum statistic.
    """
    reset_drift("dimuon_z_peak_mean_mass")

    slopes = []
    for i in range(8):
        scale = 1.0 + 0.005 * max(0, i - 3)  # flat for chunks 0-3, drifting after
        chunk = generate_chunk(600, np.random.default_rng(100 + i), momentum_scale=scale)
        near_z = np.abs(chunk["reco_mass"] - 91.19) < 10.0
        z_peak_mean = float(np.mean(chunk["reco_mass"][near_z]))
        result = record_and_check("dimuon_z_peak_mean_mass", z_peak_mean)
        if result["trend_detectable"]:
            slopes.append(result["linear_trend_slope"])

    assert len(slopes) >= 2
    assert slopes[-1] > slopes[0]  # slope should climb as the drift kicks in
