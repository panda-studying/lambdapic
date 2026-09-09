"""Phase factors for the Baier--Katkov double-time integral.

Sign conventions (all consistent with `Baier-Katkov.md` sec. 0 and sec. 5--6):

* 4-wavevector ``k = (omega, omega n)`` with ``|n| = 1``; metric ``(+---)``;
* ``kx = omega t - k . r = omega (t - n . r)``;
* double-time recoil phase
  ``Phi(t1, t2) = (epsilon/epsilon') [ kx(t2) - kx(t1) ]``
  with ``epsilon' = epsilon - omega`` (eq. 5.3 / 6.1);
* the probability integrand carries ``exp(-i Phi)``.

Numerical note (cf. `Baier-Katkov-multiparticle.md` sec. 11): the phase is
formed from *differences* ``(t2 - t1)`` and ``n . (r2 - r1)`` rather than from
two independently large absolute phases ``omega t - k . r``.  This keeps the
ultra-relativistic small-angle phase ``~ (gamma^-2 + theta^2)`` from being
eaten by cancellation of large numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "classical_phase",
    "recoil_phase",
    "local_recoil_phase",
    "phase_along",
    "recoil_frequency",
]


def classical_phase(t1, t2, omega, n, r1, r2):
    """Classical double-time phase ``omega[(t2 - t1) - n . (r2 - r1)]``.

    Parameters
    ----------
    t1, t2 : scalar or array
    omega : float
    n : (3,) unit vector
    r1, r2 : (3,) or (..., 3) arrays
    """
    dt = np.asarray(t2, dtype=np.float64) - np.asarray(t1, dtype=np.float64)
    dr = np.asarray(r2, dtype=np.float64) - np.asarray(r1, dtype=np.float64)
    return omega * (dt - np.tensordot(dr, n, axes=([-1], [0])))


def recoil_frequency(omega, epsilon, epsilon_prime=None):
    """Effective recoil-scaled frequency ``omega * epsilon / epsilon'``.

    ``epsilon'`` defaults to ``epsilon - omega``.  This is *not* a change of
    the photon frequency; it is the rescaling of the phase inside the
    formation-length integral (eq. 5.3).  Requires ``epsilon' > 0``, i.e.
    ``omega < epsilon`` in the BK recoil convention.
    """
    if epsilon_prime is None:
        epsilon_prime = epsilon - omega
    if np.any(np.asarray(epsilon_prime, dtype=np.float64) <= 0.0):
        raise ValueError(
            "epsilon' must be positive (got epsilon' = "
            f"{epsilon_prime} for omega = {omega}, epsilon = {epsilon})"
        )
    return omega * epsilon / epsilon_prime


def recoil_phase(t1, t2, omega, n, r1, r2, epsilon, epsilon_prime=None):
    """BK double-time phase ``Phi = (epsilon/epsilon') omega[(t2 - t1) - n.(r2-r1)]``.

    Anti-symmetric under exchange of 1 and 2, so that the kernel
    ``K(t2, t1) = K(t1, t2)^*`` and the double integral is real.
    """
    return recoil_frequency(omega, epsilon, epsilon_prime) * (
        (np.asarray(t2, dtype=np.float64) - np.asarray(t1, dtype=np.float64))
        - np.tensordot(
            np.asarray(r2, dtype=np.float64) - np.asarray(r1, dtype=np.float64),
            n,
            axes=([-1], [0]),
        )
    )


def local_recoil_phase(t1, t2, omega, n, r1, r2, epsilon1, epsilon2,
                       epsilon_prime1=None, epsilon_prime2=None):
    """BK local-energy double-time phase (sec. 4.2 generalization).

    ``Phi = omega [ f2 (t2 - n.r2) - f1 (t1 - n.r1) ]`` with the per-vertex
    recoil factor ``f_i = eps_i / eps'_i`` and ``eps'_i = eps_i - omega``
    (or an explicit ``epsilon_prime_i``).  The two vertices carry
    independent local energies ``eps(t1)`` and ``eps(t2)``, so the fixed-
    energy ``recoil_phase`` is recovered when ``epsilon1 == epsilon2``.

    Anti-symmetric under the simultaneous exchange of the vertex labels 1
    and 2 (times and energies), so Hermitian symmetry of the double-time
    kernel is preserved.  As with :func:`recoil_phase`, the argument is
    evaluated from differences: with ``x = t - n.r``,
    ``f2 x2 - f1 x1 = fbar (x2 - x1) + df (x1 + x2)/2`` keeps the
    ultra-relativistic small-angle phase safe from cancellation while the
    energy variation enters only through the small term ``df``.

    Parameters
    ----------
    t1, t2 : scalar or broadcastable arrays
    omega : float
    n : (3,) unit vector
    r1, r2 : (3,) or (..., 3) arrays
    epsilon1, epsilon2 : scalar or broadcastable arrays (local energies)
    epsilon_prime1, epsilon_prime2 : scalar or broadcastable arrays, optional
        Per-vertex final-state energies; default ``eps_i - omega``.
    """
    e1 = np.asarray(epsilon1, dtype=np.float64)
    e2 = np.asarray(epsilon2, dtype=np.float64)
    ep1 = e1 - omega if epsilon_prime1 is None else np.asarray(epsilon_prime1, dtype=np.float64)
    ep2 = e2 - omega if epsilon_prime2 is None else np.asarray(epsilon_prime2, dtype=np.float64)
    if np.any(ep1 <= 0.0) or np.any(ep2 <= 0.0):
        raise ValueError(
            "per-vertex epsilon' must be positive "
            f"(got epsilon'_1 = {ep1}, epsilon'_2 = {ep2} for omega = {omega})"
        )
    f1 = e1 / ep1
    f2 = e2 / ep2
    t1a = np.asarray(t1, dtype=np.float64)
    t2a = np.asarray(t2, dtype=np.float64)
    x1 = t1a - np.tensordot(np.asarray(r1, dtype=np.float64), n, axes=([-1], [0]))
    x2 = t2a - np.tensordot(np.asarray(r2, dtype=np.float64), n, axes=([-1], [0]))
    fbar = 0.5 * (f1 + f2)
    df = f2 - f1
    return omega * (fbar * (x2 - x1) + df * 0.5 * (x1 + x2))


def phase_along(t, omega, n, r, t0=None, r0=None):
    """Single-time (amplitude) phase ``omega[(t - t0) - n.(r - r0)]``.

    Used by the classical Liénard--Wiechert amplitude reference.  When
    ``t0/r0`` are None the phase is ``omega[t - n.r]``; passing them anchors the
    phase relative to the first sample, avoiding large-number cancellation.
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if t0 is None:
        t0 = t[0]
    if r0 is None:
        r0 = r[0]
    return omega * ((t - t0) - np.tensordot(r - r0, n, axes=([-1], [0])))
