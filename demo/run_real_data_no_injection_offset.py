"""
demo/run_real_data_no_injection_offset.py — replication check for the 6/10
flagged-chunks finding from run_real_data_no_injection.py.

That run used the FIRST ~80,000 raw entries in the file (sequential, not
random). This script pulls a DIFFERENT, non-overlapping stretch further
into the file (entry_start=200_000 by default) -- same zero-injection
design, same full current pipeline, same chunk size and count -- to check
whether the same Z-peak-occupancy pattern shows up again, something
different shows up, or the run comes back clean.

Three possible outcomes and what each would mean:
  - Similar Z-peak occupancy depletion again -> starts to look like a
    persistent or recurring real detector/data-taking effect, not a
    one-off. Interesting, would want a 3rd slice before saying more.
  - A DIFFERENT kind of flag (not Z-peak occupancy specifically) -> real,
    but likely a distinct real-data event at a different point in the run
    rather than a shared underlying cause.
  - Comes back clean (or near-0/10) -> supports the first run's finding
    being a genuine, LOCALIZED real effect specific to that early stretch
    of the run, not a general property of this dataset or a residual bug.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/run_real_data_no_injection_offset.py Run2012BC_DoubleMuParked_Muons.root

NOTE: not executed against the real file from this environment (no
network access to CERN's servers from here).
"""

import sys

from demo.run_real_data_drift_shift import build_scenario
from agent.loop import run_shift
from demo.export_shift_json import export_reports_to_json

N_TEST_CHUNKS = 10
ENTRY_START = 200_000  # well past the ~80,000 entries the first run covered -- no overlap


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo/run_real_data_no_injection_offset.py <path_to_root_file>")
        sys.exit(1)
    root_file = sys.argv[1]

    full_scenario = build_scenario(root_file, drift_step=0.0, entry_start=ENTRY_START)
    scenario = full_scenario[:N_TEST_CHUNKS]
    print(f"Using {len(scenario)} of {len(full_scenario)} available real chunks, "
          f"starting at entry {ENTRY_START} (non-overlapping with the earlier run).\n")

    print(f"Running {len(scenario)} REAL, UNMODIFIED chunks from an OFFSET slice through the live agent.\n")
    reports = run_shift(scenario)

    for r in reports:
        print("=" * 60)
        print(f"{r['chunk_id']}")
        print("=" * 60)
        print(r["report"])
        print()

    n_flagged = sum(1 for r in reports if any(
        c["tool"] == "escalate_finding" and c["input"].get("severity") != "normal"
        for c in r["tool_calls"]
    ))
    print(f"\n{'='*60}\nSUMMARY: {n_flagged} / {len(reports)} chunks flagged as watch/anomaly "
          f"on real, unmodified data (OFFSET slice, entry_start={ENTRY_START}).\n{'='*60}")

    out_path = "real_data_no_injection_offset_export.json"
    export_reports_to_json(reports, out_path)
    print(f"\nCompare this SUMMARY line against the first run's 6/10 result.")


if __name__ == "__main__":
    main()
