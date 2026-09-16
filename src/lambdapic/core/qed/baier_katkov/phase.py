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


def local_phase_tables(time, position, f):
    """Cumulative phase tables for the local-energy double sum.

    Returns ``(T, R)`` with ``T[k, i]`` and ``R[k, i, :]`` the trapezoid
    cumulative sums of ``f`` along the record::

        T[k, i] = sum_{m<i} (f[k,m] + f[k,m+1])/2 * (t[m+1] - t[m])
        R[k, i] = sum_{m<i} (f[k,m] + f[k,m+1])/2 * (r[m+1] - r[m])

    so that the pair phase is

    .. math:: \\Phi_{ij} = \\omega_k [(T[k,j] - T[k,i])
                                      - n . (R[k,j] - R[k,i])],

    i.e. ``omega`` times the path integral of ``f (1 - n.v) dt`` between the two
    vertices.

    Why the path integral and not an endpoint expression: with a *varying*
    recoil factor the phase is ``omega [f_j x_j - f_i x_i]`` with
    ``x = t - n.r``, which is **not translation invariant** --
    ``x -> x + c`` shifts it by ``omega c (f_j - f_i)``.  Since that factor
    weights the whole integrand, the spectrum would depend on where the origin
    of the recorded coordinates lies (measured: shifting a PIC-scale record by
    (5000, 300) changed ``d2_energy`` by 140% at ``omega = 0.2``, while the
    fixed-energy path is invariant to round-off).  Only the accumulated form
    above is invariant, and it reduces **exactly** to the fixed-energy phase
    when ``f`` is constant, because then ``T[j] - T[i] = f (t_j - t_i)`` and
    ``R[j] - R[i] = f (r_j - r_i)``.
    """
    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    f = np.atleast_2d(np.asarray(f, dtype=np.float64))
    dt = np.diff(time)                                   # (Nt-1,)
    dr = np.diff(position, axis=0)                       # (Nt-1, 3)
    # the trapezoid weight is what multiplies the *increment* of each
    # coordinate -- the time integral gets 1/2(f_m+f_m+1) dt, the space integral
    # 1/2(f_m+f_m+1) dr.  (Reusing the time weight for the space part scales it
    # by dt; caught by the constant-f unit check, where it made R disagree with
    # f (r_i - r_0).)
    half_f = 0.5 * (f[:, 1:] + f[:, :-1])                # (Nk, Nt-1)
    T = np.concatenate([np.zeros((f.shape[0], 1)),
                        np.cumsum(half_f * dt[None, :], axis=1)], axis=1)
    R = np.concatenate([np.zeros((f.shape[0], 1, 3)),
                        np.cumsum(half_f[:, :, None] * dr[None, :, :], axis=1)],
                       axis=1)
    return T, R


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
    kernel is preserved.

    .. warning::
        This two-point form is **only valid when the recoil factor is the
        same at both vertices**.  ``f2 x2 - f1 x1`` with ``x = t - n.r`` is
        not translation invariant for ``f1 != f2``, and rewriting it as
        ``fbar (x2 - x1) + df (x1 + x2)/2`` -- a "difference-stable" form that
        :func:`local_phase_tables` explains is *algebraically identical* --
        does not help: the ``df`` term carries the absolute coordinate.
        Callers with ``eps1 != eps2`` must use the accumulated form
        :func:`local_phase_tables` provides.

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
    if np.any(e1 != e2):
        raise ValueError(
            "local_recoil_phase is the two-point form omega[f2 x2 - f1 x1] and "
            "is only correct when the recoil factor is the same at both "
            "vertices; with eps1 != eps2 it is not translation invariant "
            "(shifting x by c shifts the phase by omega c (f2 - f1)). Use "
            "phase.local_phase_tables and Phi = omega[(T_j - T_i) - n.(R_j - R_i)] "
            "-- the accumulated form the integrator uses."
        )
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
