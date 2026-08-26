"""
demo/export_chunk_histograms_sample.py — the synthetic-data counterpart to
export_chunk_histograms.py. Reproduces the EXACT same scenario as
demo/test_drift_threshold_fix.py (same seeds, same 0%->2% schedule) so the
histograms genuinely match sample_shift_export.json's chunks -- not the
real-data chunks from export_chunk_histograms.py, which use a different
generator entirely despite sharing the same chunk_1..chunk_6 naming.

No API calls. Free, fast, safe to rerun.

Run with:
    PYTHONPATH=. python3 demo/export_chunk_histograms_sample.py
"""

import json

import numpy as np

from dimuon.generate import generate_chunk

BINS = np.linspace(20, 140, 41)  # matches export_chunk_histograms.py exactly


def main():
    bin_centers = ((BINS[:-1] + BINS[1:]) / 2).tolist()
    out = {"mass_bins": bin_centers, "chunks": []}

    for i in range(1, 7):
        scale = 1.0 + 0.004 * (i - 1)  # identical schedule to test_drift_threshold_fix.py
        chunk = generate_chunk(500, np.random.default_rng(i), momentum_scale=scale)
        counts, _ = np.histogram(chunk["reco_mass"], bins=BINS)
        out["chunks"].append({"chunk_id": f"chunk_{i}", "counts": counts.tolist()})

    with open("chunk_histograms_sample.json", "w") as f:
        json.dump(out, f)
    print("Wrote chunk_histograms_sample.json -- no API calls made, this was free.")


if __name__ == "__main__":
    main()
