"""
demo/export_chunk_histograms.py — regenerates the SAME real-data scenario
as run_real_data_drift_cheap.py (same file, same chunk_size, same momentum
schedule -- deterministic, so this reproduces byte-identical chunks) and
saves per-chunk mass histograms. Does NOT call the Claude API at all --
this is pure local computation, free, and safe to rerun as many times as
you want.

Run with:
    PYTHONPATH=. python3 demo/export_chunk_histograms.py Run2012BC_DoubleMuParked_Muons.root
"""

import sys

import numpy as np

from demo.run_real_data_drift_cheap import build_scenario

BINS = np.linspace(20, 140, 41)  # 40 bins, 3 GeV wide


def main(root_file: str):
    scenario = build_scenario(root_file)

    bin_centers = ((BINS[:-1] + BINS[1:]) / 2).tolist()
    out = {"mass_bins": bin_centers, "chunks": []}

    for chunk_id, chunk_data, _ in scenario:
        counts, _ = np.histogram(chunk_data["reco_mass"], bins=BINS)
        out["chunks"].append({"chunk_id": chunk_id, "counts": counts.tolist()})

    import json
    with open("chunk_histograms_real.json", "w") as f:
        json.dump(out, f)
    print("Wrote chunk_histograms_real.json -- no API calls made, this was free.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 demo/export_chunk_histograms.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])
