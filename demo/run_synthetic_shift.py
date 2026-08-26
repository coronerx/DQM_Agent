"""
demo/run_synthetic_shift.py — the first REAL (non-mocked) run of the agent
loop. Uses synthetic data (dimuon.generate) rather than real CMS data for
this first run specifically, because you already know the ground truth
(which chunk has the injected miscalibration, if any) -- that makes it much
easier to judge whether the agent's REASONING was correct, not just whether
its final verdict happened to be right.

This costs real API calls. Read the printed transcript closely -- the point
of this run is judging Claude's decisions (did it choose z_peak over
full_spectrum when checking for calibration, did it call check_occupancy,
did it apply the correction before escalating), not just the final verdict.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/run_synthetic_shift.py
"""

import numpy as np

from dimuon.generate import generate_chunk
from reference import reference_distributions
from agent.loop import run_shift


def summarize(chunk_id: str, chunk: dict) -> dict:
    mass = chunk["reco_mass"]
    z_mask = np.abs(mass - 91.1876) < 10.0
    return {
        "chunk_id": chunk_id,
        "n_events": int(len(mass)),
        "mean_mass_full_spectrum": round(float(np.mean(mass)), 3),
        "n_events_near_z_peak": int(z_mask.sum()),
        "mean_mass_near_z_peak": round(float(np.mean(mass[z_mask])), 3) if z_mask.sum() > 0 else None,
    }


def main():
    # Build the reference from a clean baseline chunk.
    reference_chunk = generate_chunk(600, np.random.default_rng(0), momentum_scale=1.0)
    reference_distributions.build_reference(reference_chunk)

    # Four chunks: two clean, one with an obvious 3% miscalibration, one
    # more clean -- so you can see both a "nothing wrong" verdict and a
    # correctly-caught problem in the same run.
    scenario = [
        ("chunk_1", generate_chunk(600, np.random.default_rng(1), momentum_scale=1.0)),
        ("chunk_2", generate_chunk(600, np.random.default_rng(2), momentum_scale=1.0)),
        ("chunk_3", generate_chunk(600, np.random.default_rng(3), momentum_scale=1.03)),  # <- injected
        ("chunk_4", generate_chunk(600, np.random.default_rng(4), momentum_scale=1.0)),
    ]

    chunks = [(chunk_id, data, summarize(chunk_id, data)) for chunk_id, data in scenario]

    print("Ground truth (for YOUR reference while reading the transcript -- the agent doesn't see this):")
    for chunk_id, _, _ in chunks:
        note = " <- 3% miscalibration injected here" if chunk_id == "chunk_3" else ""
        print(f"  {chunk_id}{note}")
    print()

    reports = run_shift(chunks)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()


if __name__ == "__main__":
    main()
