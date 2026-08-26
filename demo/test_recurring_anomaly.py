"""
demo/test_recurring_anomaly.py — does the agent actually follow through on
its own stated escalation rule?

In the single-injection blind-spot test, the agent explicitly said things
like "if this recurs in chunks 4-5, escalate to anomaly" -- but that run
never tested it, since the injected problem never repeated. This scenario
injects the same background-only anomaly (45 GeV spike, Z-peak window
untouched) into TWO non-adjacent chunks, with clean chunks in between, so
we can see whether the agent:
  (a) recognizes the second occurrence as a recurrence of the first (using
      only the short context_digest, not a full transcript), and
  (b) actually escalates severity as a result, matching what it said it
      would do.

Kept intentionally small (6 chunks) given the real lesson from the last
big run: bigger scenarios don't teach you more per dollar once the
behavior you're testing only needs a few chunks to show up.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/test_recurring_anomaly.py
"""

import numpy as np

from demo.test_full_spectrum_blind_spot import generate_chunk_with_background_anomaly
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
    reference_chunk = generate_chunk(600, np.random.default_rng(0), momentum_scale=1.0)
    reference_distributions.build_reference(reference_chunk)

    # anomaly_fraction=0.20 was chosen after checking offline (no API cost)
    # that it reliably triggers BH-significance across many random seeds --
    # the previous default (0.15) only triggered about 25% of the time by
    # chance, which is exactly why the first version of this scenario
    # accidentally left chunk_2 unflagged and never tested what it was
    # designed to test.
    anomaly_chunks = {2, 4}
    scenario = []
    for i in range(1, 7):
        chunk_id = f"chunk_{i}"
        if i in anomaly_chunks:
            data = generate_chunk_with_background_anomaly(600, np.random.default_rng(i), anomaly_fraction=0.20)
        else:
            data = generate_chunk(600, np.random.default_rng(i), momentum_scale=1.0)
        scenario.append((chunk_id, data, summarize(chunk_id, data)))

    print("Ground truth: background-only anomaly injected in chunk_2 AND chunk_4")
    print("(non-adjacent, chunk_3 is clean in between). chunks 1, 3, 5, 6 are clean.")
    print("Watch whether the agent recognizes chunk_4 as a RECURRENCE of chunk_2's")
    print("issue, using only the short context_digest -- and whether it actually")
    print("escalates severity as a result, matching what it said in chunk_2.\n")

    reports = run_shift(scenario)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()


if __name__ == "__main__":
    main()
