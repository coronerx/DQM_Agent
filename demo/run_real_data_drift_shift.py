"""
demo/run_real_data_drift_shift.py — the two biggest remaining untested
pieces at once: (1) real CMS data through the actual agent loop (every
prior real-data validation used standalone scripts, never agent.loop), and
(2) drift detection actually firing in a live run (every prior run stopped
at 4 chunks; drift_tracker needs 5+ before it even activates).

Real data alone likely won't show drift (the false-positive run already
found 0/12 flags on unperturbed real data), so a gradual synthetic
momentum-scale drift is layered on top of chunks 5+ -- real muon momenta,
synthetic ramp, so there's an actual developing trend for check_drift to
catch mid-flight rather than a single sudden jump.

COST WARNING: this makes many real API calls (roughly 12 chunks x 4-5
tool-call rounds each). Consider running with a smaller n_test_chunks
first if you want a cheaper sanity check before the full run.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/run_real_data_drift_shift.py Run2012BC_DoubleMuParked_Muons.root

NOTE: not executed against the real file from this environment (no network
access to CERN's servers from here). The data-prep logic (scaling,
chunking, summarizing) has been validated with synthetic data of the same
shape; the live-agent behavior itself can only be confirmed on your machine.
"""

import sys

import numpy as np

from dimuon.real_data_loader import load_events, chunk_events
from dimuon.kinematics import invariant_mass_array
from reference import reference_distributions
from agent.loop import run_shift
from demo.export_shift_json import export_reports_to_json

Z_MASS_GEV = 91.1876


def summarize(chunk_id: str, reco_mass: np.ndarray) -> dict:
    z_mask = np.abs(reco_mass - Z_MASS_GEV) < 10.0
    return {
        "chunk_id": chunk_id,
        "n_events": int(len(reco_mass)),
        "mean_mass_full_spectrum": round(float(np.mean(reco_mass)), 3),
        "n_events_near_z_peak": int(z_mask.sum()),
        "mean_mass_near_z_peak": round(float(np.mean(reco_mass[z_mask])), 3) if z_mask.sum() > 0 else None,
    }


def build_scenario(root_file: str, chunk_size: int = 1500, max_events: int = 50_000,
                    n_flat_chunks: int = 4, drift_step: float = 0.005):
    print(f"Loading events from {root_file} ...")
    events = load_events(source=root_file, max_events=max_events)
    raw_chunks = list(chunk_events(events, chunk_size=chunk_size))
    print(f"Loaded {len(events['pt1'])} events -> {len(raw_chunks)} chunks of size {chunk_size}\n")

    if len(raw_chunks) < n_flat_chunks + 3:
        raise RuntimeError(
            f"Only {len(raw_chunks)} chunks available -- need at least "
            f"{n_flat_chunks + 3} to run a meaningful drift scenario. "
            f"Reduce chunk_size or increase max_events."
        )

    reference_raw = raw_chunks[0]
    reference_mass = invariant_mass_array(
        reference_raw["pt1"], reference_raw["eta1"], reference_raw["phi1"],
        reference_raw["pt2"], reference_raw["eta2"], reference_raw["phi2"],
    )
    reference_distributions.build_reference({"reco_mass": reference_mass})

    test_raw_chunks = raw_chunks[1:]
    scenario = []
    print("Momentum-scale schedule (real data underneath every chunk):")
    for i, raw in enumerate(test_raw_chunks):
        chunk_id = f"chunk_{i + 1}"
        if i < n_flat_chunks:
            scale = 1.0
        else:
            scale = 1.0 + drift_step * (i - n_flat_chunks + 1)

        pt1 = raw["pt1"] * scale
        pt2 = raw["pt2"] * scale
        reco_mass = invariant_mass_array(pt1, raw["eta1"], raw["phi1"], pt2, raw["eta2"], raw["phi2"])
        chunk_data = {"reco_mass": reco_mass}
        chunk_summary = summarize(chunk_id, reco_mass)

        note = "" if scale == 1.0 else f"  <-- {(scale - 1) * 100:.1f}% injected"
        print(f"  {chunk_id}: n={len(reco_mass)}  scale={scale:.3f}{note}")
        scenario.append((chunk_id, chunk_data, chunk_summary))

    print()
    return scenario


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo/run_real_data_drift_shift.py <path_to_root_file>")
        sys.exit(1)
    root_file = sys.argv[1]

    scenario = build_scenario(root_file)
    print(f"Running {len(scenario)} chunks through the live agent -- this makes real API calls.\n")

    reports = run_shift(scenario)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()

    out_path = "real_data_drift_shift_export.json"
    export_reports_to_json(reports, out_path)
    print(f"\nOpen demo/dashboard.html and load {out_path} to see this run visually.")


if __name__ == "__main__":
    main()
