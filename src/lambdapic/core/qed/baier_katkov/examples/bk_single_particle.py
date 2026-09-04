"""Minimal Baier--Katkov single-particle example.

Builds a *closed* synchrotron circle (one full turn, ``r(0)=r(T)``,
``u(0)=u(T)``) and computes the angle-integrated ``dW/domega`` (and
``dE/domega``) spectrum at the synchrotron harmonics ``omega = m Omega``.
A closed orbit removes the hard-truncation boundary terms of the double-time
integral, so the angle-integrated spectrum is genuinely positive (and equals
the classical Liénard--Wiechert spectrum in the soft-photon limit); it is
never clipped.

Run from anywhere with::

    python -m lambdapic.core.qed.baier_katkov.examples.bk_single_particle

or directly as a script.  Natural units: c = hbar = 1, m_e = 1.
"""

from __future__ import annotations

import numpy as np

from ..types import Parameters, Trajectory
from ..integrator import compute_spectrum
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

    # synchrotron harmonics: a closed orbit radiates only at omega = m Omega.
    # (Soft-photon regime omega/eps << 1 here, so BK ~ LW; see validation.py.)
    n_harmonics = 8
    omega_grid = np.array([m * Omega for m in range(1, n_harmonics + 1)])

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
