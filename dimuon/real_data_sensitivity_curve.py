"""
dimuon/real_data_sensitivity_curve.py — instead of one pass/fail miscalibration
test, apply a RANGE of momentum-scale miscalibrations to the SAME real chunk
and trace how the KS test p-value responds. This answers "what's the smallest
miscalibration this pipeline can actually catch?" rather than just "can it
catch a 3% one?"

Using the same chunk repeatedly (rather than a different chunk per scale)
isolates the injected effect from chunk-to-chunk baseline noise -- the
earlier false-positive run showed real chunks vary somewhat on their own,
so comparing across different chunks at different scales would conflate
"how sensitive is the test" with "how noisy is this particular chunk."

Run with:
    PYTHONPATH=. python3 dimuon/real_data_sensitivity_curve.py Run2012BC_DoubleMuParked_Muons.root

NOTE: not executed against the real file from this environment; verify on
your own machine. The curve-tracing logic has been checked against synthetic
data of similar statistical character.
"""

import sys

from dimuon.real_data_loader import load_events, chunk_events
from dimuon.kinematics import invariant_mass_array
from monitors.distribution_shift import ks_test

DEFAULT_SCALES = [1.000, 1.002, 1.004, 1.006, 1.008, 1.010, 1.015, 1.020, 1.030, 1.050]


def main(root_file: str, test_chunk_index: int = 2, chunk_size: int = 1500,
          max_events: int = 50_000, scales=None):
    scales = scales or DEFAULT_SCALES

    print(f"Loading events from {root_file} ...")
    events = load_events(source=root_file, max_events=max_events)
    chunks = list(chunk_events(events, chunk_size=chunk_size))
    if len(chunks) < test_chunk_index + 2:
        print(f"Not enough chunks ({len(chunks)}) for test_chunk_index={test_chunk_index} "
              f"-- reduce chunk_size or raise max_events.")
        return

    reference_chunk = chunks[0]
    test_chunks = chunks[1:]
    target_chunk = test_chunks[test_chunk_index]

    reference_mass = invariant_mass_array(
        reference_chunk["pt1"], reference_chunk["eta1"], reference_chunk["phi1"],
        reference_chunk["pt2"], reference_chunk["eta2"], reference_chunk["phi2"],
    )

    print(f"Using test_chunks[{test_chunk_index}] (n={len(target_chunk['pt1'])}) as the "
          f"target chunk, applying each scale below to it in turn:\n")

    results = []
    threshold_scale = None
    for scale in scales:
        pt1 = target_chunk["pt1"] * scale
        pt2 = target_chunk["pt2"] * scale
        mass = invariant_mass_array(
            pt1, target_chunk["eta1"], target_chunk["phi1"],
            pt2, target_chunk["eta2"], target_chunk["phi2"],
        )
        result = ks_test("target", mass, reference_mass)
        p = result["p_value"]
        results.append((scale, p))

        pct = (scale - 1) * 100
        crossed = ""
        if p < 0.05 and threshold_scale is None and scale != 1.0:
            threshold_scale = scale
            crossed = "  <-- first scale below p=0.05"
        print(f"scale={scale:.3f}  ({pct:+.1f}%)  p_value={p:.6f}{crossed}")

    print()
    if threshold_scale:
        idx = scales.index(threshold_scale)
        prev_scale = scales[idx - 1] if idx > 0 else 1.0
        print(f"Approximate detection threshold: somewhere between "
              f"{(prev_scale - 1) * 100:.2f}% and {(threshold_scale - 1) * 100:.2f}% "
              f"miscalibration, at n={len(target_chunk['pt1'])} events in this chunk. "
              f"NOTE: this threshold is specific to this chunk size -- more events per "
              f"chunk would let you detect smaller miscalibrations, the same trade-off "
              f"as the drift-tracker window-size tuning from the synthetic tests.")
    else:
        print("No scale in the tested range crossed p=0.05 -- try larger scales or more events.")

    try:
        import matplotlib.pyplot as plt

        pcts = [(s - 1) * 100 for s, _ in results]
        pvals = [max(p, 1e-10) for _, p in results]  # floor for log-scale plotting

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pcts, pvals, "o-", color="#2563eb")
        ax.axhline(0.05, color="#dc2626", linestyle="--", label="alpha = 0.05")
        ax.set_yscale("log")
        ax.set_xlabel("Injected momentum-scale miscalibration (%)")
        ax.set_ylabel("KS test p-value (log scale, floored at 1e-10 for display)")
        ax.set_title(f"Detection sensitivity vs. miscalibration size\n"
                     f"(real CMS data, n={len(target_chunk['pt1'])} events per chunk)")
        ax.legend()
        plt.tight_layout()
        out_path = "sensitivity_curve.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved {out_path}")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot.)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dimuon/real_data_sensitivity_curve.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])
