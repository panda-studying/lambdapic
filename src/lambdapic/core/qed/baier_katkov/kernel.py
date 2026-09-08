"""Vertex / spinor kernels for the Baier--Katkov double-time integral.

This module implements the **spin-averaged, polarization-summed** kernels of
`Baier-Katkov.md` sec. 7, together with the classical Liénard--Wiechert
amplitude used as a soft-photon reference (eq. 7.4).

Two kernel forms are provided:

* :func:`trace_kernel` -- the trace form (eq. 7.1)
  ``N12 = -m^2/(eps eps') - ((eps^2 + eps'^2)/(4 eps'^2)) (b1 - b2)^2``.
* :func:`dot_kernel` -- the velocity / dot-product form (eq. 7.3)
  ``Ndot = [(eps^2 + eps'^2)(b1 . b2 - 1) + omega^2/gamma^2] / (2 eps'^2)``.

**Relationship (an exact on-shell identity).**  With
``(b1 - b2)^2 = b1^2 + b2^2 - 2 b1.b2`` and ``b_i^2 = 1 - m^2/eps^2`` for
two samples of the same on-shell energy ``eps`` (``gamma = eps/m``), the
constant pieces combine to ``m^2 (eps - eps')^2 / (2 eps^2 eps'^2)
= omega^2 / (2 eps'^2 gamma^2)`` and the two forms are *pointwise identical*.
They differ only once the two times carry different local energies
``eps1 != eps2`` (the sec. 4.2 generalization), where the trace form with
``b_i^2 = 1 - m^2/eps_i^2`` is the fundamental one.  In the classical limit
``omega/eps -> 0`` both reduce to ``b1 . b2 - 1``, which equals the classical
transverse-velocity kernel ``n x (n x b1) . n x (n x b2)`` up to
total-derivative (boundary) terms.

Validation history: an earlier transcription of the trace form carried the
coefficient ``(eps^2 + eps'^2)/(4 m^2)`` inside ``-m^2/(eps eps') [1 + ...]``,
i.e. ``(eps^2 + eps'^2)/(4 eps eps')`` instead of ``(eps^2 + eps'^2)/(4 eps'^2)``
-- one factor ``eps/eps'`` short.  Against the exact constant-field quantum
synchrotron spectrum (validation V8) that version fell below the exact
result by ~``(1 - omega/eps)`` while the dot form agreed to 0.1%; the
coefficient above restores the identity.

Both forms carry the vacuum subtraction: the ``-1`` of the dot form and the
``-m^2/(eps eps')`` contact term of the trace form are the same thing, so
neither needs an explicit endpoint / vacuum subtraction beyond what a closed
or windowed record already provides.
"""

from __future__ import annotations

import numpy as np
from scipy.special import airy

__all__ = [
    "trace_kernel",
    "dot_kernel",
    "classical_velocity_kernel",
    "classical_amplitude",
    "airy_identity",
]


def _eps_prime(epsilon, omega, epsilon_prime=None):
    if epsilon_prime is not None:
        return epsilon_prime
    return epsilon - omega


def trace_kernel(beta1, beta2, epsilon, omega, mass=1.0, epsilon_prime=None):
    """Trace kernel ``N12`` of eq. 7.1 (spin-averaged, pol.-summed).

    ``N12 = -m^2/(eps eps') - (eps^2 + eps'^2)/(4 eps'^2) (b1 - b2)^2``.
    ``epsilon_prime`` defaults to ``epsilon - omega``; pass
    ``epsilon_prime=epsilon`` for the recoilless (classical) limit.  On shell
    (``|b_i|^2 = 1 - m^2/eps^2``) this is identical to :func:`dot_kernel`.
    """
    eps_p = _eps_prime(epsilon, omega, epsilon_prime)
    db2 = np.sum((np.asarray(beta1) - np.asarray(beta2)) ** 2, axis=-1)
    return -(mass ** 2 / (epsilon * eps_p)) - (
        (epsilon ** 2 + eps_p ** 2) / (4.0 * eps_p ** 2)
    ) * db2


def dot_kernel(beta1, beta2, epsilon, omega, mass=1.0, epsilon_prime=None):
    """Velocity / dot-product kernel ``Ndot`` of eq. 7.3.

    ``gamma`` is taken as ``epsilon / mass`` (local-energy approximation,
    ``m = 1`` in natural units).  ``epsilon_prime`` defaults to
    ``epsilon - omega``; pass ``epsilon_prime=epsilon`` for the classical limit.
    """
    eps_p = _eps_prime(epsilon, omega, epsilon_prime)
    gamma = epsilon / mass
    b1 = np.asarray(beta1)
    b2 = np.asarray(beta2)
    dot = np.sum(b1 * b2, axis=-1)
    return (
        (epsilon ** 2 + eps_p ** 2) * (dot - 1.0) + omega ** 2 / gamma ** 2
    ) / (2.0 * eps_p ** 2)


def classical_velocity_kernel(beta1, beta2, n):
    """Transverse classical kernel ``n x (n x b1) . n x (n x b2)``.

    Equals ``b1 . b2 - (n.b1)(n.b2)``.  Used by the LW reference spectrum.
    """
    b1 = np.asarray(beta1)
    b2 = np.asarray(beta2)
    n = np.asarray(n)
    return np.sum(b1 * b2, axis=-1) - np.tensordot(b1, n, axes=([-1], [0])) * np.tensordot(
        b2, n, axes=([-1], [0])
    )


def classical_amplitude(time, position, beta, omega, n):
    """Classical LW far-field amplitude (eq. 7.4, up to prefactor).

    Returns the complex number ``A = int dt v(t) exp(i omega [t - n.r(t)])``
    with ``v = n x (n x beta)``, integrated by the trapezoid rule over the
    *recorded* samples.  The phase is anchored at the first sample to avoid
    cancellation (see :mod:`.phase`).
    """
    from .phase import phase_along

    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    # transverse velocity  v = n x (n x beta) = (n.beta) n - beta
    n_dot_b = np.tensordot(beta, n, axes=([-1], [0]))
    v = n_dot_b[:, None] * n[None, :] - beta

    psi = phase_along(time, omega, n, position)
    integrand = v * np.exp(1j * psi)[:, None]
    return np.trapezoid(integrand, time, axis=0)


def airy_identity(alpha, b):
    """Airy representation of ``int_-oo^oo exp(-i (alpha tau + b tau^3)) dtau``.

    From `Baier-Katkov.md` eq. 9.6: ``= 2 pi (3b)^(-1/3) Ai(alpha / (3b)^(1/3))``.
    This is a *phase-integral identity* used in validation only; it checks the
    sign convention ``exp(-i Phi)`` and the cubic rescaling of sec. 9.
    """
    alpha = float(alpha)
    b = float(b)
    x = alpha / (3.0 * b) ** (1.0 / 3.0)
    return 2.0 * np.pi * (3.0 * b) ** (-1.0 / 3.0) * airy(x)[0]
