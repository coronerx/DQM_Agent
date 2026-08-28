"""
demo/scan_z_peak_fraction.py — free, offline (no API calls) diagnostic to
map out WHERE in the file the Z-peak fraction shifts, following two live
agent runs that found dramatically different reference compositions.

v2: the first version of this script had a real bug -- it bucketed by
POST-SELECTION event count and labeled the x-axis "raw entry number,"
which silently distorted the position mapping (since only ~35-40% of raw
entries pass the two-opposite-charge-muon cut). This version tracks the
TRUE raw entry index per selected event and buckets by fixed raw-entry
windows, so the x-axis and any reference markers are actually correct.

Run with:
    PYTHONPATH=. python3 demo/scan_z_peak_fraction.py Run2012BC_DoubleMuParked_Muons.root
"""

import sys

import numpy as np

from dimuon.real_data_loader import load_events
from dimuon.kinematics import invariant_mass_array

Z_MASS_GEV = 91.1876
RAW_WINDOW_SIZE = 4000  # raw entries per bucket (~1500 selected events expected, given ~37.6% selection rate)
SCAN_MAX_ENTRIES = 500_000


def main(root_file: str):
    print(f"Loading up to {SCAN_MAX_ENTRIES} raw entries from {root_file} ...")
    events = load_events(source=root_file, max_events=SCAN_MAX_ENTRIES, entry_start=0,
                          return_raw_entry_index=True)
    raw_idx = events["raw_entry_index"]
    print(f"Loaded {len(events['pt1'])} opposite-charge dimuon events "
          f"(true raw entries spanning {raw_idx.min()} to {raw_idx.max()}).\n")

    reco_mass = invariant_mass_array(
        events["pt1"], events["eta1"], events["phi1"], events["pt2"], events["eta2"], events["phi2"]
    )
    z_flag = np.abs(reco_mass - Z_MASS_GEV) < 10.0

    bin_edges = np.arange(0, raw_idx.max() + RAW_WINDOW_SIZE, RAW_WINDOW_SIZE)
    bin_starts, fractions, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (raw_idx >= lo) & (raw_idx < hi)
        n = mask.sum()
        if n < 20:  # too few selected events in this raw window to be meaningful
            continue
        bin_starts.append(int(lo))
        fractions.append(float(z_flag[mask].mean()))
        counts.append(int(n))

    print(f"Scanned {len(fractions)} fixed raw-entry windows of {RAW_WINDOW_SIZE} entries each.\n")
    print(f"{'raw entry start':>16}  {'n selected':>11}  {'z-peak fraction':>16}")
    for start, n, frac in zip(bin_starts, counts, fractions):
        print(f"{start:>16}  {n:>11}  {frac:>16.3f}")

    fractions_arr = np.array(fractions)
    print(f"\nOverall: mean={fractions_arr.mean():.3f}  std={fractions_arr.std():.3f}  "
          f"min={fractions_arr.min():.3f}  max={fractions_arr.max():.3f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(bin_starts, fractions, "o-", color="#378ADD")
        ax.axvline(200_000, color="#791F1F", linestyle="--", linewidth=1,
                   label="entry_start=200,000 (second live-agent run's TRUE raw position)")
        ax.set_xlabel("true raw entry number in file")
        ax.set_ylabel(f"Z-peak fraction (per {RAW_WINDOW_SIZE}-raw-entry window)")
        ax.set_title("Z-peak fraction vs. TRUE raw file position (fixed, not post-selection count)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("z_peak_fraction_scan_v2.png", dpi=150)
        print("\nSaved z_peak_fraction_scan_v2.png -- the red line is now actually correct.")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot.)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 demo/scan_z_peak_fraction.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 demo/scan_z_peak_fraction.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])
