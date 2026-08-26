"""
demo/run_real_data_drift_cheap.py — a deliberately CHEAP real-data version
of the drift test. The original 12-chunk real-data run cost ~$6, mostly
from a cost bug (full history carryover) that's since been fixed. This
script combines both cost levers: fewer chunks (6, not 12) AND the
already-fixed no-history-carryover + prompt-caching loop.

Uses the exact momentum-scale schedule already validated (offline, no API
cost) in demo/test_drift_threshold_fix.py to reliably trigger the fixed
looks_like_sustained_drift boolean at chunk_5 -- same schedule, just
applied to real CMS muon momenta instead of synthetic ones this time.

Requires the real Run2012BC_DoubleMuParked_Muons.root file (see
dimuon/real_data_loader.py for how to obtain it).

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/run_real_data_drift_cheap.py Run2012BC_DoubleMuParked_Muons.root
"""

import sys

import numpy as np

from dimuon.real_data_loader import load_events, chunk_events
from dimuon.kinematics import invariant_mass_array
from reference import reference_distributions
from agent.loop import run_shift
from demo.export_shift_json import export_reports_to_json

Z_MASS_GEV = 91.1876
N_CHUNKS = 6  # deliberately small -- keep this cheap


def summarize(chunk_id: str, reco_mass: np.ndarray) -> dict:
    z_mask = np.abs(reco_mass - Z_MASS_GEV) < 10.0
    return {
        "chunk_id": chunk_id,
        "n_events": int(len(reco_mass)),
        "mean_mass_full_spectrum": round(float(np.mean(reco_mass)), 3),
        "n_events_near_z_peak": int(z_mask.sum()),
        "mean_mass_near_z_peak": round(float(np.mean(reco_mass[z_mask])), 3) if z_mask.sum() > 0 else None,
    }


def build_scenario(root_file: str, chunk_size: int = 400, max_events: int = 20_000):
    print(f"Loading events from {root_file} ...")
    events = load_events(source=root_file, max_events=max_events)
    raw_chunks = list(chunk_events(events, chunk_size=chunk_size))
    print(f"Loaded {len(events['pt1'])} events -> {len(raw_chunks)} chunks of size {chunk_size}\n")

    if len(raw_chunks) < N_CHUNKS + 1:
        raise RuntimeError(
            f"Only {len(raw_chunks)} chunks available -- need at least "
            f"{N_CHUNKS + 1} (1 reference + {N_CHUNKS} test). Reduce "
            f"chunk_size or increase max_events."
        )

    reference_raw = raw_chunks[0]
    reference_mass = invariant_mass_array(
        reference_raw["pt1"], reference_raw["eta1"], reference_raw["phi1"],
        reference_raw["pt2"], reference_raw["eta2"], reference_raw["phi2"],
    )
    reference_distributions.build_reference({"reco_mass": reference_mass})

    # Same schedule validated in test_drift_threshold_fix.py: 0% -> 2.0%
    # across 6 chunks, confirmed offline to fire looks_like_sustained_drift
    # exactly at chunk_5.
    scenario = []
    print("Momentum-scale schedule (real muon momenta underneath every chunk):")
    for i in range(N_CHUNKS):
        raw = raw_chunks[i + 1]
        scale = 1.0 + 0.004 * i
        chunk_id = f"chunk_{i + 1}"

        pt1 = raw["pt1"] * scale
        pt2 = raw["pt2"] * scale
        reco_mass = invariant_mass_array(pt1, raw["eta1"], raw["phi1"], pt2, raw["eta2"], raw["phi2"])
        chunk_data = {"reco_mass": reco_mass}
        chunk_summary = summarize(chunk_id, reco_mass)

        print(f"  {chunk_id}: n={len(reco_mass)}  scale={scale:.3f} ({(scale - 1) * 100:.1f}%)")
        scenario.append((chunk_id, chunk_data, chunk_summary))

    print()
    return scenario


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo/run_real_data_drift_cheap.py <path_to_root_file>")
        sys.exit(1)
    root_file = sys.argv[1]

    scenario = build_scenario(root_file)
    print(f"Running {len(scenario)} REAL chunks through the live agent (cheap: "
          f"no history carryover, prompt caching enabled).\n")

    reports = run_shift(scenario)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()

    out_path = "real_data_drift_cheap_export.json"
    export_reports_to_json(reports, out_path)
    print(f"\nOpen demo/dashboard.html and load {out_path} to see this run visually.")


if __name__ == "__main__":
    main()
