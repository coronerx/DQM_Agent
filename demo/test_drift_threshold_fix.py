"""
demo/test_drift_threshold_fix.py — small, cheap live-agent test specifically
for the drift threshold fix. Every prior real drift observation came from
either the expensive 12-chunk real-data run, or was never fired at all
(the original bug). This scenario is deliberately minimal: 6 synthetic
chunks with a gradual momentum-scale ramp (0% -> 2.0%), chosen (validated
offline first, no API cost) to reliably trigger looks_like_sustained_drift
starting at chunk 5 -- the earliest chunk it's statistically possible to
fire at all.

What to watch for in the transcript:
  - Does check_drift's looks_like_sustained_drift actually come back True
    starting at chunk 5 (it never did in any prior run before the fix)?
  - Does the agent's OWN language change once that happens -- does it
    start citing "formal drift confirmed" or a specific trend_p_value,
    rather than just noting the raw slope/z-score numbers as before?
  - Does severity escalate accordingly once the fix gives it a real signal
    to act on?

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/test_drift_threshold_fix.py
"""

import numpy as np

from dimuon.generate import generate_chunk, Z_MASS_GEV
from reference import reference_distributions
from agent.loop import run_shift


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
    reference_chunk = generate_chunk(500, np.random.default_rng(0), momentum_scale=1.0)
    reference_distributions.build_reference(reference_chunk)

    scenario = []
    print("Ground truth: gradual momentum-scale drift, 0.0% -> 2.0% across 6 chunks")
    print("(validated offline: looks_like_sustained_drift should first fire True")
    print("at chunk_5, the earliest statistically possible point).\n")
    for i in range(1, 7):
        scale = 1.0 + 0.004 * (i - 1)
        chunk_id = f"chunk_{i}"
        data = generate_chunk(500, np.random.default_rng(i), momentum_scale=scale)
        print(f"  {chunk_id}: momentum_scale={scale:.3f} ({(scale - 1) * 100:.1f}%)")
        scenario.append((chunk_id, data, summarize(chunk_id, data)))
    print()

    reports = run_shift(scenario)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()


if __name__ == "__main__":
    main()
