"""
demo/run_real_data_no_injection.py — the actual experiment: does real,
completely UNMODIFIED CMS data trigger anything, using the full CURRENT
pipeline (KS + the fixed two-sample chi-square + both occupancy tests +
the fixed drift-significance test)?

This is a genuinely open question, not something already answered:
  - The earlier real_data_false_positive_test.py (0/12 false alarms) only
    used the KS test -- the two-sample chi-square fix didn't exist yet.
  - Every live-agent real-data run since has had a synthetic drift
    injected on top for testing purposes -- none has run the live agent
    on purely real, unmodified data with today's full toolkit.

If this stays clean: a real specificity confirmation with the actual
current toolkit, closing a real gap. If it flags something: a genuinely
interesting finding -- something real in this dataset, not injected.

Kept to n_test_chunks=10 (not the full ~19+ available) to stay
cost-conscious while still giving the drift-significance test (needs 5+
chunks) real history to work with.

Run with:
    export ANTHROPIC_API_KEY=your_key_here
    PYTHONPATH=. python3 demo/run_real_data_no_injection.py Run2012BC_DoubleMuParked_Muons.root

NOTE: not executed against the real file from this environment (no
network access to CERN's servers from here) -- the chunk-selection and
scenario-building logic has been validated offline; the live-agent
behavior can only be confirmed on your machine.
"""

import sys

from demo.run_real_data_drift_shift import build_scenario
from agent.loop import run_shift
from demo.export_shift_json import export_reports_to_json

N_TEST_CHUNKS = 10


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo/run_real_data_no_injection.py <path_to_root_file>")
        sys.exit(1)
    root_file = sys.argv[1]

    # drift_step=0.0 means every chunk stays at scale=1.0 -- pure, unmodified
    # real data throughout, no injection of any kind.
    full_scenario = build_scenario(root_file, drift_step=0.0)
    scenario = full_scenario[:N_TEST_CHUNKS]
    print(f"Using {len(scenario)} of {len(full_scenario)} available real chunks "
          f"(capped for cost -- all real, zero injection).\n")

    print(f"Running {len(scenario)} REAL, UNMODIFIED chunks through the live agent.\n")
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
          f"on real, unmodified data.\n{'='*60}")

    out_path = "real_data_no_injection_export.json"
    export_reports_to_json(reports, out_path)
    print(f"\nOpen demo/dashboard.html or demo/agent_story.html and load {out_path} to see this run.")


if __name__ == "__main__":
    main()
