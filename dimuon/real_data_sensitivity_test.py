"""
dimuon/real_data_sensitivity_test.py — SENSITIVITY test, the counterpart to
real_data_false_positive_test.py. Given the confirmed baseline (0/12 false
alarms on real, untouched data), does the pipeline correctly detect a real
problem when one actually exists?

A synthetic momentum-scale miscalibration is applied to exactly ONE chunk of
REAL data (scaling only that chunk's reconstructed pT, leaving eta/phi and
every other chunk untouched) -- the real-data counterpart of
test_bh_correction_flags_only_the_true_miscalibration in
tests/test_dimuon_pipeline.py, which did the same thing on fully synthetic
data.

Run with:
    PYTHONPATH=. python3 dimuon/real_data_sensitivity_test.py \
        Run2012BC_DoubleMuParked_Muons.root [momentum_scale] [chunk_index]

Defaults: momentum_scale=1.02 (2% miscalibration), chunk_index=5 (0-indexed
among the test chunks, i.e. the 6th one).

NOTE: not executed against the real file from this environment (no network
access to CERN's servers from here) -- verify on your own machine. The
injection/correction logic itself has been validated against synthetic data
of the same shape (see the sandbox check described alongside this file).
"""

import sys

from dimuon.real_data_loader import load_events, chunk_events
from dimuon.kinematics import invariant_mass_array
from monitors.distribution_shift import ks_test, benjamini_hochberg


def main(root_file: str, momentum_scale: float = 1.02, miscalibrated_chunk_index: int = 5,
         chunk_size: int = 1500, max_events: int = 50_000, alpha: float = 0.05):
    print(f"Loading events from {root_file} ...")
    events = load_events(source=root_file, max_events=max_events)
    chunks = list(chunk_events(events, chunk_size=chunk_size))
    if len(chunks) < 4:
        print(f"Only {len(chunks)} chunks at chunk_size={chunk_size} -- "
              f"reduce chunk_size or raise max_events for a meaningful test.")
        return

    reference_chunk = chunks[0]
    test_chunks = chunks[1:]

    if not (0 <= miscalibrated_chunk_index < len(test_chunks)):
        print(f"miscalibrated_chunk_index must be 0..{len(test_chunks) - 1}")
        return

    reference_mass = invariant_mass_array(
        reference_chunk["pt1"], reference_chunk["eta1"], reference_chunk["phi1"],
        reference_chunk["pt2"], reference_chunk["eta2"], reference_chunk["phi2"],
    )

    injected_label = f"chunk_{miscalibrated_chunk_index + 1}"
    print(f"Injecting a {(momentum_scale - 1) * 100:.1f}% momentum-scale "
          f"miscalibration into {injected_label} only. Every other chunk is "
          f"untouched real data.\n")

    pvalues = {}
    for i, chunk in enumerate(test_chunks, start=1):
        label = f"chunk_{i}"
        scale = momentum_scale if (i - 1) == miscalibrated_chunk_index else 1.0
        pt1 = chunk["pt1"] * scale
        pt2 = chunk["pt2"] * scale
        chunk_mass = invariant_mass_array(
            pt1, chunk["eta1"], chunk["phi1"], pt2, chunk["eta2"], chunk["phi2"]
        )
        result = ks_test(label, chunk_mass, reference_mass)
        pvalues[label] = result["p_value"]
        marker = "  <-- miscalibrated" if scale != 1.0 else ""
        print(f"{label}: n={len(chunk_mass):5d}  scale={scale:.3f}  "
              f"raw p_value={result['p_value']:.4f}{marker}")

    print()
    corrected = benjamini_hochberg(pvalues, alpha=alpha)
    print(f"After Benjamini-Hochberg correction (alpha={alpha}):")
    for label, result in sorted(corrected.items(), key=lambda kv: kv[1]["rank"]):
        flag = "FLAGGED" if result["significant_after_correction"] else "clear"
        marker = "  <-- miscalibrated" if label == injected_label else ""
        print(f"  {label}: p={result['p_value']:.4f}  rank={result['rank']}  [{flag}]{marker}")

    flagged = {l for l, r in corrected.items() if r["significant_after_correction"]}
    print()
    if flagged == {injected_label}:
        print(f"SUCCESS: only {injected_label} (the actually-miscalibrated chunk) "
              f"was flagged.")
    elif injected_label in flagged:
        print(f"{injected_label} was correctly flagged, but so were: "
              f"{flagged - {injected_label}} -- worth checking whether those are real "
              f"effects in the data or genuine false alarms before assuming either.")
    else:
        print(f"MISS: {injected_label} was NOT flagged despite the injected "
              f"miscalibration. Try a larger momentum_scale (e.g. 1.03-1.05) or a "
              f"bigger chunk_size -- more events per chunk gives more statistical "
              f"power to detect a small shift, same lesson as the drift-tracker "
              f"tuning finding from the synthetic tests.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dimuon/real_data_sensitivity_test.py <path_to_root_file> "
              "[momentum_scale] [miscalibrated_chunk_index]")
        sys.exit(1)
    root_file = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.02
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(root_file, momentum_scale=scale, miscalibrated_chunk_index=idx)
