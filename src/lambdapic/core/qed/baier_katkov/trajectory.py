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

from .types import Trajectory

__all__ = [
    "Trajectory",
    "as_trajectory",
    "uniform_trajectory",
    "linear_interpolate",
]


def as_trajectory(time, position, momentum=None, energy=None, mass=1.0, charge=1.0) -> Trajectory:
    """Construct a :class:`Trajectory` from raw arrays, checking the ordering."""
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError("time must be 1-D")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time samples must be strictly increasing")
    return Trajectory(time=time, position=position, momentum=momentum,
                      energy=energy, mass=mass, charge=charge)


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
