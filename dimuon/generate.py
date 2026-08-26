"""
generate.py — synthetic but physically correct dimuon events.

Approach: sample a target invariant mass from a realistic spectrum (a
falling Drell-Yan-like continuum plus a Breit-Wigner Z resonance), construct
the muon pair EXACTLY at that mass in the dimuon center-of-mass frame
(isotropic decay angle), then apply a random longitudinal boost -- physically
motivated, since real quark/antiquark pairs at the LHC rarely carry equal and
opposite momentum fractions, so real dimuon systems are typically boosted
along the beam axis.

Because a longitudinal boost preserves invariant mass exactly, an unscaled
generated event's reconstructed mass (via kinematics.invariant_mass) should
recover the sampled target mass to floating-point precision -- this is what
tests/test_dimuon_pipeline.py checks first, since if this round-trip doesn't
hold, nothing built on top of it can be trusted.

momentum_scale lets you inject a synthetic detector miscalibration: scaling
reconstructed pT by a constant factor (leaving eta/phi alone) approximates
a mis-set muon momentum scale, which is a real, well-known systematic in
these measurements -- CMS/ATLAS use the observed Z peak position specifically
to calibrate this. Because muon mass is tiny compared to typical pT here,
this scales the reconstructed invariant mass by very nearly the same factor.
"""

import numpy as np

from dimuon.kinematics import MUON_MASS_GEV, invariant_mass_array

Z_MASS_GEV = 91.1876
Z_WIDTH_GEV = 2.4952


def sample_target_masses(n: int, rng: np.random.Generator, z_fraction: float = 0.35,
                          mass_range=(20.0, 140.0)) -> np.ndarray:
    """Mixture: z_fraction from a Breit-Wigner around the Z mass, the rest
    from a falling power-law continuum background (roughly mimicking the
    real Drell-Yan continuum shape below the Z peak)."""
    n_z = int(round(n * z_fraction))
    n_bg = n - n_z

    # Breit-Wigner via a scaled/shifted Cauchy distribution, clipped to range.
    z_masses = rng.standard_cauchy(n_z) * (Z_WIDTH_GEV / 2) + Z_MASS_GEV
    z_masses = np.clip(z_masses, mass_range[0], mass_range[1])

    # Falling background: draw via inverse-CDF of a power law ~ m^-3,
    # a reasonable stand-in shape for the real continuum.
    u = rng.uniform(0, 1, n_bg)
    lo, hi = mass_range
    alpha = 2.0  # power-law index (m^-3 -> CDF exponent -2)
    bg_masses = (u * (hi**-alpha - lo**-alpha) + lo**-alpha) ** (-1 / alpha)

    masses = np.concatenate([z_masses, bg_masses])
    rng.shuffle(masses)
    return masses


def _generate_pair_at_mass(target_mass: float, rng: np.random.Generator,
                            boost_rapidity_sigma: float = 0.6):
    """Construct one muon pair with EXACT invariant mass = target_mass."""
    E_star = target_mass / 2
    p_star = np.sqrt(max(E_star**2 - MUON_MASS_GEV**2, 0.0))

    costheta = rng.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)
    phi_star = rng.uniform(0, 2 * np.pi)

    px1 = p_star * sintheta * np.cos(phi_star)
    py1 = p_star * sintheta * np.sin(phi_star)
    pz1 = p_star * costheta
    E1 = E_star

    px2, py2, pz2, E2 = -px1, -py1, -pz1, E_star

    # Random longitudinal boost (along the beam axis, z).
    Y = rng.normal(0, boost_rapidity_sigma)
    cosh_Y, sinh_Y = np.cosh(Y), np.sinh(Y)

    def boost_z(px, py, pz, E):
        return px, py, pz * cosh_Y + E * sinh_Y, E * cosh_Y + pz * sinh_Y

    px1, py1, pz1, E1 = boost_z(px1, py1, pz1, E1)
    px2, py2, pz2, E2 = boost_z(px2, py2, pz2, E2)

    from dimuon.kinematics import to_pt_eta_phi
    pt1, eta1, phi1 = to_pt_eta_phi(px1, py1, pz1)
    pt2, eta2, phi2 = to_pt_eta_phi(px2, py2, pz2)
    return pt1, eta1, phi1, pt2, eta2, phi2


def generate_chunk(n: int, rng: np.random.Generator, momentum_scale: float = 1.0,
                    z_fraction: float = 0.35) -> dict:
    """
    Generate n dimuon events. Returns arrays of muon kinematics plus the
    reconstructed invariant mass (with momentum_scale applied, as a real
    reconstruction algorithm would compute it from measured, possibly
    miscalibrated, momenta).
    """
    target_masses = sample_target_masses(n, rng, z_fraction=z_fraction)

    pt1 = np.empty(n); eta1 = np.empty(n); phi1 = np.empty(n)
    pt2 = np.empty(n); eta2 = np.empty(n); phi2 = np.empty(n)

    for i, m in enumerate(target_masses):
        pt1[i], eta1[i], phi1[i], pt2[i], eta2[i], phi2[i] = _generate_pair_at_mass(m, rng)

    # Apply the (possibly miscalibrated) momentum scale to what a detector
    # would actually report -- this is the "measured" pT, separate from the
    # true generated pT above.
    reco_pt1 = pt1 * momentum_scale
    reco_pt2 = pt2 * momentum_scale

    reco_mass = invariant_mass_array(reco_pt1, eta1, phi1, reco_pt2, eta2, phi2)
    true_mass = invariant_mass_array(pt1, eta1, phi1, pt2, eta2, phi2)

    return {
        "pt1": reco_pt1, "eta1": eta1, "phi1": phi1,
        "pt2": reco_pt2, "eta2": eta2, "phi2": phi2,
        "reco_mass": reco_mass,
        "true_mass": true_mass,  # only used for validation/tests, not "measured"
    }
