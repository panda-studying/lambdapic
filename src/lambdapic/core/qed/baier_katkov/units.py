"""Unit conventions for the Baier--Katkov reference module.

This module implements the unit bookkeeping described in ``Baier-Katkov.md``
(notably sec. 15, ``t_nat = t_SI / hbar``, ``r_nat = r_SI / (hbar c)``).

Two *equivalent* natural-unit pictures are supported and clearly separated:

1. **``energy_unit = "m_e"`` (Compton units, the default used internally).**
   We set ``c = hbar = 1`` *and* take the electron mass as the mass unit so that
   ``m_e = 1``.  In this picture the natural time/length/energy units are

   .. math::
       t_unit = hbar / (m_e c^2),   x_unit = hbar / (m_e c),
       E_unit = m_e c^2,            omega_unit = m_e c^2 / hbar.

   This is the cleanest choice for the actual computation and for the
   classical-limit / Airy comparisons in `Baier-Katkov.md` sec. 7 and 9.

2. **``energy_unit = "joule"`` (the literal sec. 15 mapping).**  Here
   ``c = hbar = 1`` but *energies are still measured in joules*, so
   ``t_nat = t_SI / hbar`` and ``r_nat = r_SI / (hbar c)`` are both
   dimensionful numbers with SI unit ``J^-1``.  The electron mass is then
   ``m_e = m_e c^2`` (in joules, ~8.19e-14), not 1.  This picture is only a
   rescaling of picture 1 by ``m_e c^2``; the dimensionless phase ``kx`` is
   identical in both.

The two pictures are related by a pure energy rescaling, so a dimensionless
physical quantity (a phase, or ``dW``) is unchanged while a quantity carrying
energy dimensions is rescaled by powers of ``m_e c^2``.

All SI constants are taken from :mod:`scipy.constants` (CODATA).
"""

from __future__ import annotations

from scipy.constants import alpha, c, e, hbar, m_e

__all__ = [
    "alpha", "c", "e", "hbar", "m_e",
    "fine_structure", "electron_mass", "electron_mass_energy",
    "natural_time_unit", "natural_length_unit", "natural_energy_unit",
    "natural_frequency_unit",
    "si_time_to_natural", "natural_time_to_si",
    "si_length_to_natural", "natural_length_to_si",
    "si_energy_to_natural", "natural_energy_to_si",
    "si_frequency_to_natural", "natural_frequency_to_si",
    "natural_omega_to_photon_energy_ev", "photon_energy_ev_to_natural_omega",
    "si_to_natural", "natural_to_si",
]

# --------------------------------------------------------------------------
# SI constants (CODATA, SI units)
# --------------------------------------------------------------------------
fine_structure = float(alpha)        # dimensionless ~ 1/137.036
electron_mass = float(m_e)           # kg
electron_mass_energy = float(m_e * c ** 2)   # J  (m_e c^2)

# In Heaviside--Lorentz natural units alpha = e^2 / (4 pi), so
# e^2 / (4 pi^2) = alpha / pi.  We never use the SI charge magnitude `e`
# directly in the integrand; only `alpha` enters the radiation prefactor.

# --------------------------------------------------------------------------
# Compton-unit natural scales (picture 1, m_e = 1)
# --------------------------------------------------------------------------
natural_time_unit = hbar / (m_e * c ** 2)       # s  (unit of time)
natural_length_unit = hbar / (m_e * c)          # m  (unit of length)
natural_energy_unit = m_e * c ** 2              # J  (unit of energy)
natural_frequency_unit = m_e * c ** 2 / hbar    # rad/s (unit of angular freq.)


# --------------------------------------------------------------------------
# SI <-> natural conversions (Compton units, picture 1)
# --------------------------------------------------------------------------
def si_time_to_natural(t_si):
    """Convert time [s] -> natural time [m_e^-1] (i.e. hbar/(m_e c^2))."""
    return t_si / natural_time_unit


def natural_time_to_si(t_nat):
    """Convert natural time -> seconds."""
    return t_nat * natural_time_unit


def si_length_to_natural(x_si):
    """Convert length [m] -> natural length [m_e^-1].  Equivalently x_SI/(hbar c)."""
    return x_si / natural_length_unit


def natural_length_to_si(x_nat):
    """Convert natural length -> metres."""
    return x_nat * natural_length_unit


def si_energy_to_natural(E_si):
    """Convert energy [J] -> natural energy [m_e]."""
    return E_si / natural_energy_unit


def natural_energy_to_si(E_nat):
    """Convert natural energy -> joules."""
    return E_nat * natural_energy_unit


def si_frequency_to_natural(omega_si):
    """Convert angular frequency [rad/s] -> natural angular frequency [m_e]."""
    return omega_si / natural_frequency_unit


def natural_frequency_to_si(omega_nat):
    """Convert natural angular frequency -> rad/s."""
    return omega_nat * natural_frequency_unit


def natural_omega_to_photon_energy_ev(omega_nat):
    """Natural photon angular frequency -> photon energy in eV (hbar*omega)."""
    return omega_nat * natural_energy_unit / e


def photon_energy_ev_to_natural_omega(E_ev):
    """Photon energy [eV] -> natural angular frequency."""
    return E_ev * e / natural_energy_unit


# --------------------------------------------------------------------------
# Generic vectorised converters (thin wrappers)
# --------------------------------------------------------------------------
def si_to_natural(t_si=None, x_si=None, p_si=None, E_si=None, omega_si=None):
    """Convert a collection of SI inputs to Compton natural units.

    Each keyword may be a scalar or array; missing keywords are omitted.
    ``p_si`` (momentum in kg m/s) maps to normalized momentum u = gamma beta.
    """
    out = {}
    if t_si is not None:
        out["t"] = si_time_to_natural(t_si)
    if x_si is not None:
        out["x"] = si_length_to_natural(x_si)
    if E_si is not None:
        out["E"] = si_energy_to_natural(E_si)
    if omega_si is not None:
        out["omega"] = si_frequency_to_natural(omega_si)
    if p_si is not None:
        out["u"] = p_si / (m_e * c)
    return out


def natural_to_si(t_nat=None, x_nat=None, E_nat=None, omega_nat=None):
    """Inverse of :func:`si_to_natural` for the supported quantities."""
    out = {}
    if t_nat is not None:
        out["t"] = natural_time_to_si(t_nat)
    if x_nat is not None:
        out["x"] = natural_length_to_si(x_nat)
    if E_nat is not None:
        out["E"] = natural_energy_to_si(E_nat)
    if omega_nat is not None:
        out["omega"] = natural_frequency_to_si(omega_nat)
    return out
