"""
reference_distributions.py — the known-good baseline every chunk is
compared against: both the reco_mass distribution (for KS/chi-square shape
tests) and the event count (for occupancy checks, since shape tests alone
can't detect a pure count drop -- see LAB_MANUAL.md Section 5.1).
"""

import numpy as np

_reference_mass: np.ndarray | None = None
_reference_n_events: int | None = None


def build_reference(chunk_data: dict) -> None:
    global _reference_mass, _reference_n_events
    _reference_mass = np.asarray(chunk_data["reco_mass"], dtype=float)
    _reference_n_events = len(_reference_mass)


def get_reference_mass() -> np.ndarray:
    if _reference_mass is None:
        raise RuntimeError("No reference set. Call build_reference() first.")
    return _reference_mass


def get_reference_n_events() -> int:
    if _reference_n_events is None:
        raise RuntimeError("No reference set. Call build_reference() first.")
    return _reference_n_events


def has_reference() -> bool:
    return _reference_mass is not None
