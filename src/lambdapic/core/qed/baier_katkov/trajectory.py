"""Trajectory input and interpolation.

The first (reference) version accepts the simplest possible input: a set of
discrete samples ``(t_i, r_i)`` with an optional normalized-momentum history
``u_i = gamma beta``, and interpolates them **linearly**.

Explicit assumptions (do not silently violate them):

* Samples are ordered in time, ``t_0 < t_1 < ... < t_{N-1}`` (uniform spacing
  is *not* required, but the reference integrator below uses a trapezoid rule
  that assumes a sorted grid).
* When ``momentum`` is provided it is treated as the normalized momentum
  ``u = gamma beta`` (the same convention as λPIC's ``ux, uy, uz``).  The
  velocity is then ``beta = u / sqrt(1 + |u|^2)``, which is exact on the mass
  shell rather than a numerical derivative.
* When ``momentum`` is absent, ``beta`` is reconstructed by central
  differencing ``r(t)`` (see :func:`types.Trajectory.beta`); this amplifies
  sampling noise and should be avoided for production use.
* The trajectory is a **finite** record of the classical background motion.
  Finite-record endpoint effects are *not* removed here (no adiabatic switch,
  no asymptotic padding); they are documented and left for later stages.  If
  the record already contains stochastic LCFA recoil, it is not a recoilless
  background and must not be fed to this module (double counting).
"""

from __future__ import annotations

import numpy as np

from . import units as _units
from .types import Trajectory

__all__ = [
    "Trajectory",
    "as_trajectory",
    "as_trajectory_si",
    "uniform_trajectory",
    "linear_interpolate",
]


def as_trajectory(time, position, momentum=None, energy=None) -> Trajectory:
    """Construct a :class:`Trajectory` from raw arrays.

    Ordering, positivity and finiteness are checked by
    :meth:`Trajectory.__post_init__`; mass and charge live on
    :class:`~.types.Parameters`, which alone governs the spectrum.

    ``energy=None`` means the **fixed-incident-energy** phase: the right choice
    for a record whose ``|momentum|`` is constant, and silently wrong for one
    that is not.  This constructor does not second-guess the caller -- for a
    PIC record use :func:`as_trajectory_si`, which derives the energy history
    from the momentum when it is not given.
    """
    return Trajectory(time=time, position=position, momentum=momentum,
                      energy=energy)


def as_trajectory_si(t_si, x_si, u=None, energy=None, n_samples=None) -> Trajectory:
    """Build a :class:`Trajectory` from a PIC record (the SI unit adapter).

    Parameters
    ----------
    t_si : (Nt,) array
        Sample times in **seconds** (SI).  Produced by
        :class:`~lambdapic.callback.trajectory.TrajectoryRecorder`.
    x_si : (Nt, 3) array
        Positions in **metres** (SI).
    u : (Nt, 3) array, optional
        Normalized momentum ``u = gamma beta``.  This is *passed through
        unconverted*: it is already dimensionless and is the same quantity the
        PIC stores as ``ux, uy, uz``, so ``units.si_to_natural``'s ``p_si``
        entry (which takes kg m/s) is deliberately **not** used here.
    energy : (Nt,) array, optional
        Local electron energy ``eps(t_i)`` in **natural units** (for an
        electron that is just ``gamma = 1/inv_gamma``).  **Left out, it is
        derived on shell from** ``u``: a PIC record's energy history is not
        extra information, it is exactly ``sqrt(1 + |u|^2)`` -- the same
        on-shell relation the kernel assumes (``|b_i|^2 = 1 - m^2/eps_i^2``).
        This matters because the integrator switches to the local-energy phase
        only when ``Trajectory.energy`` is set: without the derivation an
        accelerating record is silently evaluated at a *fixed* incident
        energy, and that is not the same answer -- measured on the production
        LWFA record (``gamma`` 1.0 -> 2.4, interaction window), where the
        fixed evaluation is additionally off shell, the two differ by 0.06%
        with the ``dot`` kernel inside the resolvable band but by factors of
        0.46-17 with the ``trace`` kernel, and the fixed evaluation's own two
        kernels disagree there.  Pass an explicit array to override: a
        *constant* array evaluates at a fixed ``eps`` deliberately, which is a
        diagnostic rather than an answer on such a record.  ``u=None`` (a
        record without momentum) still gives the fixed-energy path.
    n_samples : int or array, optional
        Length of the valid prefix.  The recorder writes fixed-shape
        NaN-padded arrays so that a mid-record gap stays visible; slice to
        ``n_samples`` here, because a NaN in ``time`` would otherwise be caught
        by ``Trajectory.__post_init__`` and a NaN in ``position`` would make
        every returned spectral value NaN.  A length-1 array is accepted (the
        recorder stores it that way; ``int(np.array([n]))`` raises on NumPy >= 2).
    """
    t_si = np.asarray(t_si, dtype=np.float64)
    x_si = np.asarray(x_si, dtype=np.float64)
    if t_si.ndim != 1:
        raise ValueError("t_si must be 1-D")
    if x_si.shape != (t_si.shape[0], 3):
        raise ValueError("x_si must have shape (Nt, 3)")
    for name, arr in (("u", u), ("energy", energy)):
        if arr is not None and np.asarray(arr).shape != ((t_si.shape[0], 3) if name == "u"
                                                         else (t_si.shape[0],)):
            raise ValueError(f"{name} has the wrong shape for this record")

    if n_samples is not None:
        n_arr = np.atleast_1d(np.asarray(n_samples))
        if n_arr.size != 1:
            raise ValueError("n_samples must be a scalar or a length-1 array")
        n = int(n_arr[0])
        if not 2 <= n <= t_si.shape[0]:
            raise ValueError(
                f"n_samples must satisfy 2 <= n_samples <= {t_si.shape[0]}, got {n}"
            )
        t_si, x_si = t_si[:n], x_si[:n]
        u = None if u is None else np.asarray(u)[:n]
        energy = None if energy is None else np.asarray(energy)[:n]

    nat = _units.si_to_natural(t_si=t_si, x_si=x_si)
    if energy is None and u is not None:
        # The record's energy history is not extra information: the PIC stores
        # the normalized momentum, so eps(t) = sqrt(1 + |u|^2) is exactly the
        # on-shell energy the kernel assumes.  Leaving it unset would silently
        # evaluate the record at one fixed eps -- see the `energy` parameter
        # docstring for what that costs.
        u_arr = np.asarray(u, dtype=np.float64)
        energy = np.sqrt(1.0 + np.einsum("ij,ij->i", u_arr, u_arr))
    return as_trajectory(nat["t"], nat["x"], momentum=u, energy=energy)


def uniform_trajectory(r_of_t, t0, t1, n_samples, **kwargs) -> Trajectory:
    """Sample a callable ``r_of_t(t) -> (3,)`` on a uniform grid ``[t0, t1]``.

    ``r_of_t`` may also return a ``(Nt, 3)`` array when called with an array of
    times.  Useful for building analytical test trajectories.
    """
    time = np.linspace(t0, t1, n_samples)
    r = np.asarray(r_of_t(time), dtype=np.float64)
    if r.shape == (3,):
        r = np.tile(r, (n_samples, 1))
    if r.shape != (n_samples, 3):
        raise ValueError("r_of_t must return shape (3,) or (Nt, 3)")
    return as_trajectory(time, r, **kwargs)


def linear_interpolate(r, time, t_query) -> np.ndarray:
    """Linear interpolation of a (``Nt, 3``) position history onto query times.

    Returns ``(Nq, 3)``.  ``t_query`` must lie within ``[time[0], time[-1]]``.
    """
    r = np.asarray(r, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
    if np.any(t_query < time[0]) or np.any(t_query > time[-1]):
        raise ValueError("query times outside the recorded interval")
    idx = np.searchsorted(time, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(time) - 2)
    t0 = time[idx]
    t1 = time[idx + 1]
    frac = (t_query - t0) / (t1 - t0)
    frac = np.clip(frac, 0.0, 1.0)[:, None]
    return r[idx] * (1.0 - frac) + r[idx + 1] * frac
