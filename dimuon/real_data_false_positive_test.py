"""
dimuon/real_data_false_positive_test.py — validate SPECIFICITY before
sensitivity: does the KS test + Benjamini-Hochberg pipeline correctly stay
quiet when there's no real problem, using genuinely real data (not
synthetic)?

This matters as a prerequisite to testing miscalibration detection: if this
script shows a high false-alarm rate on real, presumably-stable data, then
a later "successfully detected the injected miscalibration" result would be
uninterpretable -- you wouldn't know if the detection was real signal or
just the system's baseline noisiness. Establish the baseline first.

Requires the real Run2012BC_DoubleMuParked_Muons.root file (see
dimuon/real_data_loader.py for how to obtain it).

Run with:
    PYTHONPATH=. python3 dimuon/real_data_false_positive_test.py Run2012BC_DoubleMuParked_Muons.root

NOTE: not executed against the real file from this environment -- verify
on your own machine. The logic below has been checked with synthetic data
of the same shape.
"""

import sys

import numpy as np

from dimuon.real_data_loader import load_events, chunk_events
from monitors.distribution_shift import ks_test, benjamini_hochberg


def main(root_file: str, chunk_size: int = 1500, max_events: int = 50_000, alpha: float = 0.05):
    print(f"Loading events from {root_file} ...")
    events = load_events(source=root_file, max_events=max_events)
    n_total = len(events["pt1"])
    print(f"Loaded {n_total} opposite-charge dimuon events")

    chunks = list(chunk_events(events, chunk_size=chunk_size))
    if len(chunks) < 3:
        print(f"Only {len(chunks)} chunk(s) at chunk_size={chunk_size} -- "
              f"reduce chunk_size or increase max_events for a meaningful test.")
        return

    # First chunk is the reference; everything after is tested against it.
    reference_chunk = chunks[0]
    test_chunks = chunks[1:]
    print(f"Using chunk 0 ({len(reference_chunk['pt1'])} events) as reference; "
          f"testing {len(test_chunks)} subsequent chunks against it.\n")

    from dimuon.kinematics import invariant_mass_array
    reference_mass = invariant_mass_array(
        reference_chunk["pt1"], reference_chunk["eta1"], reference_chunk["phi1"],
        reference_chunk["pt2"], reference_chunk["eta2"], reference_chunk["phi2"],
    )

    pvalues = {}
    for i, chunk in enumerate(test_chunks, start=1):
        chunk_mass = invariant_mass_array(
            chunk["pt1"], chunk["eta1"], chunk["phi1"],
            chunk["pt2"], chunk["eta2"], chunk["phi2"],
        )
        result = ks_test(f"chunk_{i}", chunk_mass, reference_mass)
        pvalues[f"chunk_{i}"] = result["p_value"]
        print(f"chunk_{i}: n={len(chunk_mass):5d}  raw p_value={result['p_value']:.4f}")

    print()
    corrected = benjamini_hochberg(pvalues, alpha=alpha)
    n_flagged = sum(1 for r in corrected.values() if r["significant_after_correction"])

    print(f"After Benjamini-Hochberg correction (alpha={alpha}):")
    for label, result in sorted(corrected.items(), key=lambda kv: kv[1]["rank"]):
        flag = "FLAGGED" if result["significant_after_correction"] else "clear"
        print(f"  {label}: p={result['p_value']:.4f}  rank={result['rank']}  [{flag}]")

    print(f"\n{n_flagged} / {len(test_chunks)} chunks flagged as significant.")
    if n_flagged == 0:
        print("No false alarms on real, presumably-stable data -- good baseline "
              "before testing whether the pipeline can detect a real injected problem.")
    else:
        print(f"Some chunks flagged with no injected problem. Before assuming this is "
              f"a bug: check whether real running conditions actually changed between "
              f"these chunks (trigger prescale changes, luminosity conditions, etc. are "
              f"real effects in this dataset, not necessarily false positives) -- this "
              f"is worth investigating rather than dismissing either way.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dimuon/real_data_false_positive_test.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])
