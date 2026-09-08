"""Analytic references for validating the Baier--Katkov integrator.

Constant-field (LCFA) synchrotron formulas in the module's natural units
(``c = hbar = m_e = 1``, ``epsilon = gamma``).  For uniform circular motion
the field along the trajectory is *exactly* constant, so these closed forms
are the analytic evaluation of the very double-time integral the module
computes numerically (Baier & Katkov 1967; Sokolov & Ternov; Ritus 1985).
Comparing against them tests the *implementation* -- prefactor, recoil
phase, ``omega^2/gamma^2`` term, ``dot`` vs ``trace`` kernel -- not the
theory.

The photon-number rate per unit laboratory time and per unit
``delta = omega / epsilon`` (spin averaged, polarization summed) is

.. math::

    \\frac{dW}{dt\\,d\\delta} = \\frac{\\alpha}{\\sqrt3\\,\\pi\\,\\varepsilon}
    \\Big[\\Big(1-\\delta+\\frac{1}{1-\\delta}\\Big) K_{2/3}(y)
          - \\int_y^\\infty K_{1/3}(x)\\,dx\\Big],
    \\qquad y = \\frac{2\\delta}{3\\chi(1-\\delta)},

with ``chi`` the quantum parameter (``chi = gamma^2 beta^2 / rho`` on a
circle of radius ``rho``).  The equivalent Airy form used by the online
LCFA tables in ``core/qed/optical_depth_tables.py``,

.. math::

    \\frac{dW}{dt\\,d\\delta} = -\\frac{\\alpha}{\\varepsilon}
    \\Big[\\int_z^\\infty \\mathrm{Ai}(x)\\,dx
          + \\Big(\\frac{2}{z} + \\delta\\chi\\sqrt z\\Big)\\mathrm{Ai}'(z)\\Big],
    \\qquad z = \\Big(\\frac{\\delta}{\\chi(1-\\delta)}\\Big)^{2/3},

is provided as an independent implementation (``form="airy"``); the two are
identical through ``Ai(z) = (1/pi) sqrt(z/3) K_{1/3}(y)``,
``Ai'(z) = -(z/(pi sqrt3)) K_{2/3}(y)`` and
``2 + delta chi z^{3/2} = (1 - delta) + 1/(1 - delta)``.

In the classical limit ``chi -> 0`` (``delta -> 0`` at fixed ``delta/chi``)
the bracket becomes ``int_y^inf K_{5/3}`` and the energy spectrum reduces to
Schwinger's ``dP/dw = (sqrt3 / 2 pi) (alpha gamma / rho) F(w / w_c)``, whose
integral is the Larmor power ``(2/3) alpha chi^2``.

Closed orbits.  A one-turn double-time integral of the BK kernel is free of
truncation (boundary) terms only when the integrand is periodic, i.e. when
the *recoil-scaled* frequency ``omega epsilon / epsilon'`` is a harmonic of
the orbital frequency.  The physical emission lines therefore sit at
``omega_m = m Omega / (1 + m Omega / epsilon)`` (the quantum recoil shift of
the harmonics) and are spaced by ``Delta omega_m = Omega (epsilon'/epsilon)^2``;
the per-turn line energy is ``E_m = Delta omega_m * d2E/domega`` so the
code's one-turn ``dE/domega`` evaluated *at* ``omega_m`` is directly the
continuum spectral density ``T dP/domega`` -- no Jacobian.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import airy, kv, roots_legendre

from .units import fine_structure

__all__ = [
    "circle_chi",
    "circle_omega_c",
    "recoil_shifted_harmonic",
    "quantum_synchrotron_rate",
    "classical_synchrotron_rate",
    "synchrotron_F",
    "synchrotron_power",
    "erber_g",
    "orbit_direction_grid",
]


# --------------------------------------------------------------------------
# circle kinematics
# --------------------------------------------------------------------------
def circle_chi(gamma, rho):
    """Quantum parameter ``chi = gamma beta B / B_cr = gamma^2 beta^2 / rho``."""
    return gamma ** 2 * (1.0 - 1.0 / gamma ** 2) / rho


def circle_omega_c(gamma, rho):
    """Classical critical frequency ``w_c = (3/2) gamma^3 Omega``, ``Omega = beta/rho``."""
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    return 1.5 * gamma ** 3 * beta / rho


def recoil_shifted_harmonic(m, Omega, epsilon):
    """Emitted frequency of harmonic ``m`` with BK recoil, ``m Omega / (1 + m Omega/eps)``.

    Solves ``omega * epsilon / (epsilon - omega) = m Omega``; this is the
    frequency at which the one-turn BK double integral is periodic (no
    boundary terms).  ``epsilon = inf`` gives the classical ``m Omega``.
    """
    m = np.asarray(m, dtype=np.float64)
    return m * Omega / (1.0 + m * Omega / epsilon)


# --------------------------------------------------------------------------
# constant-field spectra
# --------------------------------------------------------------------------
_QUAD = dict(limit=200, epsabs=0.0, epsrel=1e-10)


def _int_K13(y):
    return quad(lambda x: kv(1.0 / 3.0, x), y, np.inf, **_QUAD)[0]


def _int_K53(y):
    return quad(lambda x: kv(5.0 / 3.0, x), y, np.inf, **_QUAD)[0]


def _int_Ai(z):
    return quad(lambda x: airy(x)[0], z, np.inf, **_QUAD)[0]


#: ``K_nu(y) ~ exp(-y)`` underflows double precision beyond this; the rate is
#: then below 1e-260 of its peak and is returned as exactly zero.
_Y_MAX = 600.0


def quantum_synchrotron_rate(delta, chi, epsilon, form="macdonald"):
    """Quantum synchrotron photon rate ``dW/(dt d delta)`` (lab time, natural units).

    Parameters
    ----------
    delta : float or array in (0, 1)
        photon energy fraction ``omega / epsilon``.
    chi : float
        quantum parameter.
    epsilon : float
        electron energy (``gamma`` for ``m = 1``).
    form : ``"macdonald"`` (K_{1/3}, K_{2/3}) or ``"airy"`` (the LCFA-table form).
    """
    if form not in ("macdonald", "airy"):
        raise ValueError("form must be 'macdonald' or 'airy'")
    delta = np.asarray(delta, dtype=np.float64)
    out = np.empty_like(delta)
    for i, d in np.ndenumerate(delta):
        if not 0.0 < d < 1.0:
            out[i] = 0.0
            continue
        y = 2.0 * d / (3.0 * chi * (1.0 - d))
        if y > _Y_MAX:
            out[i] = 0.0
            continue
        if form == "macdonald":
            bracket = (1.0 - d + 1.0 / (1.0 - d)) * kv(2.0 / 3.0, y) - _int_K13(y)
            out[i] = fine_structure / (np.sqrt(3.0) * np.pi * epsilon) * bracket
        else:
            z = (d / (chi * (1.0 - d))) ** (2.0 / 3.0)
            ai_prime = airy(z)[1]
            out[i] = -(fine_structure / epsilon) * (
                _int_Ai(z) + (2.0 / z + d * chi * np.sqrt(z)) * ai_prime
            )
    return out if out.ndim else float(out)


def classical_synchrotron_rate(delta, chi, epsilon):
    """Classical (recoil-free) limit of :func:`quantum_synchrotron_rate`.

    ``dW/(dt d delta) = alpha / (sqrt3 pi eps) int_{y0}^inf K_{5/3}``,
    ``y0 = 2 delta / (3 chi)``.  The energy spectrum per unit frequency,
    ``dP/domega = delta * rate``, is Schwinger's
    ``(sqrt3 / 2 pi) (alpha gamma / rho) F(omega / w_c)``.  Unlike the
    quantum rate it is not cut off at ``delta = 1``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    out = np.empty_like(delta)
    for i, d in np.ndenumerate(delta):
        y0 = 2.0 * d / (3.0 * chi)
        if d <= 0.0 or y0 > _Y_MAX:
            out[i] = 0.0
            continue
        out[i] = fine_structure / (np.sqrt(3.0) * np.pi * epsilon) * _int_K53(y0)
    return out if out.ndim else float(out)


def synchrotron_F(x):
    """Schwinger's ``F(x) = x int_x^inf K_{5/3}(xi) dxi``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i, v in np.ndenumerate(x):
        out[i] = v * _int_K53(v) if v > 0.0 else 0.0
    return out if out.ndim else float(out)


def synchrotron_power(chi, epsilon, quantum=True):
    """Radiated power ``P = int eps delta dW/(dt d delta) d delta``.

    Quantum: integrated over ``delta in (0, 1)``; equals
    ``(2/3) alpha chi^2 g(chi)`` with ``g`` the exact quantum suppression
    factor (:func:`erber_g` is the usual fit).  Classical: integrated over
    ``delta in (0, inf)`` (the recoil-free spectrum has no cutoff) and equals
    the Larmor power ``(2/3) alpha chi^2`` identically.
    """
    if quantum:
        val, _ = quad(lambda d: epsilon * d * quantum_synchrotron_rate(d, chi, epsilon),
                      0.0, 1.0, limit=200)
    else:
        val, _ = quad(lambda d: epsilon * d * classical_synchrotron_rate(d, chi, epsilon),
                      0.0, np.inf, limit=200)
    return val


def erber_g(chi):
    """Baier--Katkov--Strakhovenko fit of the quantum power suppression ``g(chi)``.

    ``g ~ [1 + 4.8 (1 + chi) ln(1 + 1.7 chi) + 2.44 chi^2]^(-2/3)``, accurate to
    a few per cent over all ``chi``.
    """
    chi = np.asarray(chi, dtype=np.float64)
    return (1.0 + 4.8 * (1.0 + chi) * np.log(1.0 + 1.7 * chi) + 2.44 * chi ** 2) ** (-2.0 / 3.0)


# --------------------------------------------------------------------------
# angular quadrature for a circle in the x-y plane
# --------------------------------------------------------------------------
def orbit_direction_grid(gamma, n_inner=24, n_outer=8, inner_half_width=None):
    """Directions and solid-angle weights for an orbit in the x-y plane.

    Emission is azimuthally symmetric about ``z`` and mirror symmetric in the
    orbital plane, so directions are taken at ``phi = 0`` on the upper
    hemisphere, ``n = (cos psi, 0, sin psi)`` with ``psi`` the elevation,
    and the weights carry ``2 pi cos(psi) dpsi`` times 2 for both
    hemispheres.  ``psi`` is sampled with Gauss--Legendre on
    ``[0, inner_half_width]`` (the synchrotron cone, default ``8/gamma``) and
    on ``[inner_half_width, pi/2]``.
    """
    if inner_half_width is None:
        inner_half_width = min(8.0 / gamma, np.pi / 2.0)
    x_in, w_in = roots_legendre(n_inner)
    psi_in = 0.5 * inner_half_width * (x_in + 1.0)
    wpsi_in = 0.5 * inner_half_width * w_in
    x_out, w_out = roots_legendre(n_outer)
    span = np.pi / 2.0 - inner_half_width
    psi_out = inner_half_width + 0.5 * span * (x_out + 1.0)
    wpsi_out = 0.5 * span * w_out
    psi = np.concatenate([psi_in, psi_out])
    wpsi = np.concatenate([wpsi_in, wpsi_out])
    dirs = np.column_stack([np.cos(psi), np.zeros_like(psi), np.sin(psi)])
    weights = 2.0 * 2.0 * np.pi * np.cos(psi) * wpsi
    return dirs, weights
