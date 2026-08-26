"""
demo/test_full_spectrum_blind_spot.py — DIAGNOSTIC, not part of the core
pipeline. Tests whether the agent's observed preference for region='z_peak'
(seen in demo/run_synthetic_shift.py's transcript -- it never chose
'full_spectrum' across 4 chunks) is a real blind spot.

Injects an anomaly ONLY in the background continuum (a fake artifact spike
around 45 GeV, e.g. mimicking a trigger threshold problem) while leaving
every single Z-peak-window event completely untouched. If the agent only
ever checks region='z_peak', it should be statistically UNABLE to notice
this -- that's the hypothesis being tested here, not assumed.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/test_full_spectrum_blind_spot.py
"""

import numpy as np

from dimuon.generate import sample_target_masses, _generate_pair_at_mass, Z_MASS_GEV
from dimuon.kinematics import invariant_mass_array
from reference import reference_distributions
from agent.loop import run_shift


def generate_chunk_with_background_anomaly(
    n: int, rng: np.random.Generator, anomaly_fraction: float = 0.15,
    anomaly_mass: float = 45.0, anomaly_width: float = 1.5, z_fraction: float = 0.35,
) -> dict:
    """Same physics as dimuon.generate.generate_chunk, EXCEPT: a fraction of
    the background (non-Z-window) events are replaced with a tight artificial
    spike at anomaly_mass. Z-window events are sampled identically to the
    normal generator and never touched afterward."""
    target_masses = sample_target_masses(n, rng, z_fraction=z_fraction)

    z_mask = np.abs(target_masses - Z_MASS_GEV) < 10.0
    background_indices = np.where(~z_mask)[0]

    n_anomalous = int(round(len(background_indices) * anomaly_fraction))
    anomalous_indices = rng.choice(background_indices, size=n_anomalous, replace=False)
    target_masses[anomalous_indices] = rng.normal(anomaly_mass, anomaly_width, n_anomalous)

    pt1 = np.empty(n); eta1 = np.empty(n); phi1 = np.empty(n)
    pt2 = np.empty(n); eta2 = np.empty(n); phi2 = np.empty(n)
    for i, m in enumerate(target_masses):
        pt1[i], eta1[i], phi1[i], pt2[i], eta2[i], phi2[i] = _generate_pair_at_mass(m, rng)

    reco_mass = invariant_mass_array(pt1, eta1, phi1, pt2, eta2, phi2)
    return {"pt1": pt1, "eta1": eta1, "phi1": phi1, "pt2": pt2, "eta2": eta2, "phi2": phi2,
            "reco_mass": reco_mass}


def summarize(chunk_id: str, chunk: dict) -> dict:
    mass = chunk["reco_mass"]
    z_mask = np.abs(mass - Z_MASS_GEV) < 10.0
    return {
        "chunk_id": chunk_id,
        "n_events": int(len(mass)),
        "mean_mass_full_spectrum": round(float(np.mean(mass)), 3),
        "n_events_near_z_peak": int(z_mask.sum()),
        "mean_mass_near_z_peak": round(float(np.mean(mass[z_mask])), 3) if z_mask.sum() > 0 else None,
    }


def main():
    from dimuon.generate import generate_chunk

    reference_chunk = generate_chunk(600, np.random.default_rng(0), momentum_scale=1.0)
    reference_distributions.build_reference(reference_chunk)

    scenario = [
        ("chunk_1", generate_chunk(600, np.random.default_rng(1), momentum_scale=1.0)),
        ("chunk_2", generate_chunk(600, np.random.default_rng(2), momentum_scale=1.0)),
        ("chunk_3", generate_chunk_with_background_anomaly(600, np.random.default_rng(3))),  # <- injected, Z window untouched
        ("chunk_4", generate_chunk(600, np.random.default_rng(4), momentum_scale=1.0)),
    ]

    print("Ground truth: chunk_3 has a background-only anomaly (45 GeV spike).")
    print("The Z-peak window (81-101 GeV) in chunk_3 is COMPLETELY UNTOUCHED --")
    print("if the agent only ever checks region='z_peak', it should be unable")
    print("to see this at all. Watch which region(s) it actually chooses.\n")

    chunks = [(chunk_id, data, summarize(chunk_id, data)) for chunk_id, data in scenario]
    reports = run_shift(chunks)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()


if __name__ == "__main__":
    main()
