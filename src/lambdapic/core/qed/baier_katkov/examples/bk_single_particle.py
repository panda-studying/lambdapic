"""Minimal Baier--Katkov single-particle example.

Builds a *closed* synchrotron circle (one full turn, ``r(0)=r(T)``,
``u(0)=u(T)``) and computes the angle-integrated ``dW/domega`` (and
``dE/domega``) spectrum at the recoil-shifted synchrotron harmonics
``omega_m = m Omega / (1 + m Omega / epsilon)``.  These are the frequencies at
which the one-turn BK double-time integral is periodic, so the
hard-truncation boundary terms vanish and the angle-integrated spectrum is
genuinely positive (and equals the classical Liénard--Wiechert spectrum in
the soft-photon limit); it is never clipped.  Evaluating at the bare
``m Omega`` instead would leave an ``O(m omega/epsilon)`` truncation artifact.

Run from anywhere with::

    python -m lambdapic.core.qed.baier_katkov.examples.bk_single_particle

or directly as a script.  Natural units: c = hbar = 1, m_e = 1.
"""

from __future__ import annotations

import numpy as np

from ..types import Parameters, Trajectory
from ..integrator import compute_spectrum
from ..reference import recoil_shifted_harmonic
from ..result import format_spectrum, save_spectrum_text


def full_circle_trajectory(gamma=10.0, rho=200.0, n_samples=512):
    """A relativistic electron on one full synchrotron circle (x-y plane).

    ``rho`` is the curvature radius (natural units).  The record is closed:
    position and normalized momentum ``u = gamma beta`` match at the endpoints.
    Returns a :class:`Trajectory`.
    """
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    Omega = beta / rho
    T = 2.0 * np.pi / Omega
    t = np.linspace(0.0, T, n_samples)

    phi = Omega * t
    r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])

    u_perp = gamma * beta
    u = np.column_stack([-u_perp * np.sin(phi), u_perp * np.cos(phi), np.zeros_like(t)])

    return Trajectory(time=t, position=r, momentum=u, mass=1.0, charge=1.0), Omega


def main() -> None:
    gamma = 10.0
    traj, Omega = full_circle_trajectory(gamma=gamma, rho=200.0, n_samples=512)

    # incident electron energy = gamma * m = gamma (natural units, m = 1)
    params = Parameters(epsilon=gamma, mass=1.0, charge=1.0)

    # recoil-shifted synchrotron harmonics: a closed orbit radiates at
    # omega_m = m Omega / (1 + m Omega/eps), where the BK one-turn integral is
    # periodic.  (Soft-photon regime omega/eps << 1 here, so BK ~ LW; see
    # validation.py V3/V8.)
    n_harmonics = 8
    omega_grid = recoil_shifted_harmonic(np.arange(1, n_harmonics + 1), Omega, gamma)

    # Full-sphere angle integration.  The circle lies in the x-y plane and its
    # emission is azimuthally symmetric about the z-axis, so n_phi=1 with a
    # Gauss--Legendre theta quadrature (theta_max=pi) is exact.
    spectrum = compute_spectrum(
        traj, params, omega_grid,
        theta_max=np.pi,
        n_theta=16, n_phi=1,
        axis=np.array([0.0, 0.0, 1.0]),
    )

    print(format_spectrum(spectrum))
    save_spectrum_text(spectrum, "bk_spectrum.txt")
    print("\nWrote bk_spectrum.txt")


if __name__ == "__main__":
    main()
