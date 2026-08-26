"""
kinematics.py — the physics ground truth: given two muons' transverse
momentum (pT), pseudorapidity (eta), and azimuthal angle (phi), compute the
invariant mass of the dimuon system.

This is the standard collider convention: particles are described by
(pT, eta, phi, mass) rather than raw (px, py, pz, E), because pT/eta/phi are
what a real detector actually measures. Converting to a four-vector and back
is the same operation every dimuon analysis (including the real CMS Open
Data example notebook) does first.
"""

import numpy as np

MUON_MASS_GEV = 0.1056583745  # PDG value


def four_vector(pt: float, eta: float, phi: float, mass: float = MUON_MASS_GEV):
    """Convert (pT, eta, phi, mass) to a lab-frame four-vector (px, py, pz, E)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, E


def to_pt_eta_phi(px: float, py: float, pz: float):
    """Inverse of four_vector's momentum part -- lab-frame Cartesian back to (pT, eta, phi)."""
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    # Guard the pt->0 edge case (particle moving along the beam axis) --
    # shouldn't occur for the physics we generate, but don't divide by zero.
    if pt < 1e-12:
        eta = np.sign(pz) * 50.0  # effectively infinite eta
    else:
        eta = 0.5 * np.log((p_mag + pz) / (p_mag - pz))
    return pt, eta, phi


def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2,
                    mass1: float = MUON_MASS_GEV, mass2: float = MUON_MASS_GEV) -> float:
    """
    The actual physics quantity: the invariant mass of a two-particle system.
    This is frame-independent -- it's the same number no matter which
    reference frame the momenta happen to be measured in, which is exactly
    why it's the quantity used to identify resonances (Z boson, etc.)
    regardless of how the collision happened to be oriented.
    """
    px1, py1, pz1, E1 = four_vector(pt1, eta1, phi1, mass1)
    px2, py2, pz2, E2 = four_vector(pt2, eta2, phi2, mass2)

    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m_squared = E**2 - px**2 - py**2 - pz**2
    # Numerical noise can push a near-zero m_squared slightly negative;
    # physically m_squared >= 0 always for a real two-particle system.
    return np.sqrt(max(m_squared, 0.0))


def invariant_mass_array(pt1, eta1, phi1, pt2, eta2, phi2,
                          mass1: float = MUON_MASS_GEV, mass2: float = MUON_MASS_GEV):
    """Vectorized version for arrays of events (numpy arrays in, array out)."""
    pt1, eta1, phi1 = np.asarray(pt1), np.asarray(eta1), np.asarray(phi1)
    pt2, eta2, phi2 = np.asarray(pt2), np.asarray(eta2), np.asarray(phi2)

    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)

    E = E1 + E2
    px, py, pz = px1 + px2, py1 + py2, pz1 + pz2
    m_squared = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.clip(m_squared, 0.0, None))
