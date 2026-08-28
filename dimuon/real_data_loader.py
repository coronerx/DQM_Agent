"""
dimuon/real_data_loader.py — load REAL CMS 2012 Run2012BC dimuon events from
the CERN Open Data Portal, and cross-validate this project's own
invariant_mass_array() against real (not synthetic) data.

Install first:  pip install uproot awkward

Run with:  PYTHONPATH=. python dimuon/real_data_loader.py

NOTE: this has NOT been executed against the live CERN servers from this
environment -- my sandbox's network allowlist doesn't include
opendata.cern.ch / eospublic.cern.ch. Verify this runs correctly on your
own machine before relying on it.
"""

import numpy as np
import uproot
import awkward as ak

from dimuon.kinematics import invariant_mass_array

REMOTE_URL = (
    "root://eospublic.cern.ch//eos/opendata/cms/derived-data/"
    "AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root"
)

# If XRootD streaming is unreliable on your network, download the file
# first with:
#   curl http://opendata.cern.ch/record/12342/files/Run2012BC_DoubleMuParked_Muons.root \
#        -o Run2012BC_DoubleMuParked_Muons.root
# then pass that local filename as `source` below instead of REMOTE_URL.


def load_events(source: str = REMOTE_URL, max_events: int | None = 50_000,
                 entry_start: int = 0, return_raw_entry_index: bool = False) -> dict:
    """
    Selects events with exactly two opposite-charge muons (the standard
    dimuon selection) and returns their kinematics as plain numpy arrays,
    one entry per event: pt1, eta1, phi1, pt2, eta2, phi2.

    entry_start: skip this many raw entries before reading -- lets you pull
    an INDEPENDENT, non-overlapping slice further into the file, rather
    than always the same first `max_events` entries.

    return_raw_entry_index: if True, also returns "raw_entry_index" -- the
    TRUE original raw tree entry number for each selected event (not the
    post-selection position). Needed for anything that plots against
    actual file position, since only ~35-40% of raw entries pass the
    two-opposite-charge-muon cut -- conflating "post-selection count" with
    "raw entry number" silently distorts any x-axis built from it.
    """
    tree = uproot.open(source)["Events"]
    arrays = tree.arrays(
        ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"],
        entry_start=entry_start,
        entry_stop=entry_start + max_events if max_events is not None else None,
        library="ak",
    )
    raw_entry_index = entry_start + np.arange(len(arrays))

    two_muon_mask = arrays["nMuon"] == 2
    two_muon = arrays[two_muon_mask]
    raw_entry_index = raw_entry_index[np.asarray(two_muon_mask)]

    opposite_charge_mask = two_muon["Muon_charge"][:, 0] != two_muon["Muon_charge"][:, 1]
    selected = two_muon[opposite_charge_mask]
    raw_entry_index = raw_entry_index[np.asarray(opposite_charge_mask)]

    result = {
        "pt1": ak.to_numpy(selected["Muon_pt"][:, 0]),
        "eta1": ak.to_numpy(selected["Muon_eta"][:, 0]),
        "phi1": ak.to_numpy(selected["Muon_phi"][:, 0]),
        "pt2": ak.to_numpy(selected["Muon_pt"][:, 1]),
        "eta2": ak.to_numpy(selected["Muon_eta"][:, 1]),
        "phi2": ak.to_numpy(selected["Muon_phi"][:, 1]),
    }
    if return_raw_entry_index:
        result["raw_entry_index"] = raw_entry_index
    return result


def chunk_events(events: dict, chunk_size: int = 2000):
    """
    Split loaded real events into sequential chunks -- the real-data
    equivalent of demo/simulate.py's synthetic chunks, for feeding into
    reference_distributions.build_reference() and the monitors.
    """
    n = len(events["pt1"])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield {k: v[start:end] for k, v in events.items()}


if __name__ == "__main__":
    import sys

    # Prefer a local file if you've downloaded one (XRootD streaming needs
    # fsspec-xrootd, which is a pain to install -- plain HTTPS download is
    # much more reliable):
    #   curl http://opendata.cern.ch/record/12342/files/Run2012BC_DoubleMuParked_Muons.root \
    #        -o Run2012BC_DoubleMuParked_Muons.root
    #   python3 dimuon/real_data_loader.py Run2012BC_DoubleMuParked_Muons.root
    source = sys.argv[1] if len(sys.argv) > 1 else REMOTE_URL

    print(f"Loading real CMS 2012 dimuon events from: {source}")
    events = load_events(source=source, max_events=50_000)
    n = len(events["pt1"])
    print(f"Loaded {n} opposite-charge dimuon events")

    # Cross-validation: this project's own kinematics code, run on REAL data,
    # should reproduce the same physics (a visible Z peak) that the official
    # CMS tutorial's C++/ROOT version produces.
    our_mass = invariant_mass_array(
        events["pt1"], events["eta1"], events["phi1"],
        events["pt2"], events["eta2"], events["phi2"],
    )
    print(f"Mean invariant mass: {our_mass.mean():.2f} GeV")
    near_z = np.abs(our_mass - 91.19) < 10.0
    print(f"Fraction of events within 10 GeV of the Z peak: {near_z.mean():.3f}")
    print("(Compare this qualitatively against the known CMS dimuon spectrum plot -- "
          "a clear peak near 91 GeV confirms our kinematics.py is computing this "
          "correctly on real, not just synthetic, data.)")

    # Save the histogram so you can SEE the spectrum, not just summary numbers.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(our_mass, bins=300, range=(0.25, 140), log=True, color="#2563eb")
        for label, mass in [("J/psi", 3.097), ("Upsilon", 9.46), ("Z", 91.19)]:
            ax.axvline(mass, color="gray", linestyle="--", linewidth=0.8)
            ax.text(mass, ax.get_ylim()[1] * 0.7, label, rotation=90,
                     fontsize=8, ha="right", va="top")
        ax.set_xlabel("Dimuon invariant mass (GeV)")
        ax.set_ylabel("Events (log scale)")
        ax.set_title("Real CMS 2012 dimuon spectrum -- computed by this project's kinematics.py")
        plt.tight_layout()
        out_path = "dimuon_spectrum_real_data.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved spectrum plot to {out_path}")
    except ImportError:
        print("(matplotlib not installed -- skipping plot. pip install matplotlib to get one.)")
