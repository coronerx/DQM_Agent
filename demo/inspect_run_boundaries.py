"""
demo/inspect_run_boundaries.py — checks whether this file retains CMS's
standard `run` / `luminosityBlock` / `event` provenance branches, and if
so, maps out where the RUN NUMBER changes across the file -- to check
whether those boundaries line up with the Z-peak-fraction composition
shifts already found (elevated regions around raw entries ~196,000-208,000
and ~324,000-420,000).

If this works: we can identify the ACTUAL CMS run number(s) responsible
for the composition shift, which is a real, checkable physical answer
("different run = different trigger/luminosity conditions"), not just a
statistical inference.

Entirely local computation once the file is downloaded -- no API calls,
no network beyond opening the local file.

Run with:
    PYTHONPATH=. python3 demo/inspect_run_boundaries.py Run2012BC_DoubleMuParked_Muons.root
"""

import sys

import numpy as np
import uproot

SCAN_MAX_ENTRIES = 500_000


def main(root_file: str):
    tree = uproot.open(root_file)["Events"]
    branches = tree.keys()

    print(f"Tree has {len(branches)} branches. Checking for provenance branches...\n")
    has_run = "run" in branches
    has_lumi = "luminosityBlock" in branches
    has_event = "event" in branches
    print(f"  'run' branch present:             {has_run}")
    print(f"  'luminosityBlock' branch present: {has_lumi}")
    print(f"  'event' branch present:           {has_event}")

    if not has_run:
        print("\nNo 'run' branch -- this reduced file does not retain run-number "
              "provenance. The composition-shift finding stands on its own "
              "statistical merits, but can't be traced to a specific CMS run "
              "this way. Full branch list for reference:")
        for b in branches:
            print(f"  {b}")
        return

    print(f"\nGood -- 'run' branch exists. Scanning the first {SCAN_MAX_ENTRIES} "
          f"raw entries for run-number changes...\n")

    fields = ["run"] + (["luminosityBlock"] if has_lumi else [])
    arrays = tree.arrays(fields, entry_start=0, entry_stop=SCAN_MAX_ENTRIES, library="np")
    runs = arrays["run"]

    change_points = np.where(np.diff(runs) != 0)[0] + 1
    boundaries = [0] + change_points.tolist() + [len(runs)]

    print(f"{'raw entry start':>16}  {'raw entry end':>14}  {'run number':>12}  {'n entries':>10}")
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        run_number = int(runs[start])
        print(f"{start:>16}  {end:>14}  {run_number:>12}  {end - start:>10}")

    print(f"\n{len(boundaries) - 1} distinct run(s) found in the first {SCAN_MAX_ENTRIES} raw entries.")
    print("\nCompare these run-boundary positions directly against "
          "z_peak_fraction_scan_v2.png's elevated regions "
          "(~196,000-208,000 and ~324,000-420,000) -- if a run boundary "
          "falls inside or near either range, that's the real, physical "
          "explanation: those events come from a genuinely different CMS "
          "run, with its own trigger/luminosity conditions.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 demo/inspect_run_boundaries.py <path_to_root_file>")
        sys.exit(1)
    main(sys.argv[1])
