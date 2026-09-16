"""Banded-truncation diagnostics for the Baier--Katkov double-time integral.

The double sum costs ``O(Nt^2)`` per frequency.  The roadmap originally proposed
truncating it to pairs with ``|t2 - t1| <~ B tau_f``, which would make it
``O(Nt)``.  ``REVIEW_AND_ROADMAP.md`` section 8.2 showed that this is invalid on
a closed orbit -- the pair correlation does not decay there, so a band removes
cancelling terms rather than a decaying tail.  Whether it is valid for an
aperiodic record (what a PIC actually produces) is still open, and these
functions are the measurement that decides it.

Two quantities matter, and the second is the one that gates the first:

``lag_profile``
    Bins the contributions of the double sum by physical lag ``|t_j - t_i|``.
    Its cumulative sum is the truncation curve ``C(W)``: if ``C(W)`` reaches the
    full sum for ``W`` of a few ``tau_f`` and stays there while ``W`` is still
    much smaller than the record, a band is usable.

``cancellation``
    ``sum|terms| / |value|`` at that frequency.  On a closed orbit's line centre
    this is about ``2.5e5`` (section 8.2(f)), which means the answer is a small
    residue of large cancelling terms -- and any partial sum is then dominated
    by quadrature and round-off rather than by physics.  Report it *before*
    interpreting any curve.

The upper-triangle convention (``j >= i``, off-diagonal doubled) encodes
``K(t2,t1) = K(t1,t2)^*``, which holds for *any* trajectory and is exactly what
``integrator._accumulate_row`` does for the numba backend.  Contributions are
binned with the actual trapezoid weights ``w_i w_j`` rather than by index
difference, because the end weights are ``dt/2`` and never repeat.

Limitation: only the fixed-incident-energy phase is implemented.  A trajectory
carrying an ``energy`` history (the local-energy mode) raises, because its phase
and kernel are vertex-dependent -- see ``integrator._accumulate_row_local``.
"""

from __future__ import annotations

import numpy as np

from . import kernel as _kernel
from .integrator import PREFACTOR, cone_directions, trapz_weights
from .types import Parameters, Trajectory

__all__ = ["lag_profile", "truncation_curve", "probe_directions"]


def probe_directions(theta_max=np.pi, n_theta=32, n_phi=1, axis=(0.0, 0.0, 1.0)):
    """Canonical direction grid for the probe.

    Defaults match the closed-orbit measurement in section 8.2 so a new record
    can be compared against the published table with the same quadrature.
    """
    return cone_directions(np.asarray(axis, dtype=np.float64), theta_max,
                           n_theta, n_phi)


def lag_profile(trajectory: Trajectory, params: Parameters, omega: float,
                dirs=None, dom=None, chunk: int = 256, bin_width: float = None):
    """Bin the double-sum contributions by physical lag.

    Returned ``lag`` is the **upper edge** of each bin.  Because a pair whose
    lag equals an edge falls into the *next* bin, ``cumulative[i]`` is the set
    ``lag < lag[i]`` -- one boundary short of the direct mask ``|t_j - t_i| <=
    lag[i]``.  Report the edge (rather than the bin centre) because the curve is
    discontinuous on the bin scale: the cancellation ratio is 1e4-1e7, so
    including one extra diagonal band of pairs moves the ratio by O(1).  For
    the same reason, read the curve off by bin index; interpolating at an
    arbitrary ``W`` compares sets that differ by a whole band.

    Note the default direction grid (``probe_directions``) is a *probe* rule,
    not a converged angular quadrature: ``dE_domega`` from this function is
    0.6x-2.4x the module's converged angle-integrated value on a
    ``gamma=10, chi=0.5`` circle, and can even have the wrong sign where the
    spectrum is a cancellation residual.  Ratios along the curve are the
    meaningful output; the absolute value is not a spectrum.

    Returns
    -------
    dict with
        ``lag``           upper edge of each bin ``(N_bins,)``, trajectory time units
        ``c``             signed contribution per bin ``(N_bins,)``
        ``cumulative``    ``np.cumsum(c)`` -- the truncation curve ``C(W)``
        ``total``         the unbinned sum (equals ``cumulative[-1]``)
        ``cancellation``  ``sum|terms| / |total|``
        ``dE_domega``     angle-integrated ``dE/domega`` from the same sum
    """
    if trajectory.energy is not None:
        raise NotImplementedError(
            "local-energy trajectories are not supported by the probe: their "
            "phase and kernel are vertex-dependent (see "
            "integrator._accumulate_row_local). The closed-orbit control and "
            "the constant-gamma testbed both use the fixed-energy phase."
        )
    if dirs is None and dom is None:
        dirs, dom = probe_directions()
    elif dirs is None or dom is None:
        raise ValueError("pass dirs and dom together (or neither)")
    dirs = np.asarray(dirs, dtype=np.float64)
    dom = np.asarray(dom, dtype=np.float64)

    time = np.asarray(trajectory.time, dtype=np.float64)
    position = np.asarray(trajectory.position, dtype=np.float64)
    beta = trajectory.beta()
    w = trapz_weights(time)
    nt = time.shape[0]

    eps_p = float(params.epsilon_prime(omega))
    fac = float(omega * params.recoil_factor(omega))
    mass = params.mass
    # honour the parameter, do not hardcode: on shell the two kernels coincide,
    # off shell they differ by ~7% on a gamma=10 circle at epsilon=5
    kern = (_kernel.dot_kernel if params.kernel == "dot"
            else _kernel.trace_kernel)

    if bin_width is None:
        bin_width = float(np.min(np.diff(time)))
    n_bins = int(np.floor((time[-1] - time[0]) / bin_width)) + 1
    edges = np.arange(n_bins + 1) * bin_width

    c = np.zeros(n_bins)
    abs_sum = 0.0
    for start in range(0, nt, chunk):
        idx = np.arange(start, min(start + chunk, nt))
        ti = time[idx][:, None]
        dt = time[None, :] - ti
        dr = position[None, :, :] - position[idx][:, None, :]
        upper = np.arange(nt)[None, :] >= idx[:, None]

        ang = np.zeros_like(dt)
        for d in range(dirs.shape[0]):
            ang += dom[d] * np.cos(fac * (dt - dr @ dirs[d]))

        n_mat = kern(beta[idx][:, None, :], beta[None, :, :],
                     params.epsilon, omega, mass=mass, epsilon_prime=eps_p)
        contrib = np.where(upper, w[idx][:, None] * w[None, :] * n_mat * ang, 0.0)
        # off-diagonal pairs are counted twice by the Hermitian form
        contrib = np.where(upper & (dt > 0.0), 2.0 * contrib, contrib)

        lag = np.abs(dt)
        # A pair at lag == s*bin_width must land in bin s, but the computed
        # ratio is s*(1 +/- 1e-16), and truncation would drop a fraction of them
        # into bin s-1.  With the cancellation ratio of order 1e4 measured
        # below, re-assigning even a few pairs moves the *partial* sums
        # appreciably, so the curve would be measuring bin-assignment noise
        # rather than physics (measured: cumulative at 3 tau_f differed by 27%
        # against an exactly-masked sum before this tolerance was added).
        b = np.clip(np.floor(lag / bin_width + 1e-9).astype(np.int64), 0, n_bins - 1)
        c += np.bincount(b.ravel(), weights=contrib.ravel(), minlength=n_bins)[:n_bins]
        abs_sum += float(np.abs(contrib).sum())

    total = float(c.sum())
    cumulative = np.cumsum(c)
    return dict(
        lag=edges[1:],
        c=c,
        cumulative=cumulative,
        total=total,
        cancellation=float(abs_sum / abs(total)) if total != 0.0 else np.inf,
        dE_domega=PREFACTOR * omega ** 2 * params.charge ** 2 * total,
        bin_width=bin_width,
    )


def truncation_curve(trajectory: Trajectory, params: Parameters, omegas,
                     dirs=None, dom=None, chunk: int = 256, bin_width: float = None):
    """``C(W)/C(inf)`` for several frequencies at once.

    Returns a dict with ``lag`` ``(N_bins,)`` shared by all frequencies,
    ``omega`` ``(N_omega,)``, ``ratio`` ``(N_omega, N_bins)`` and
    ``cancellation`` ``(N_omega,)``.
    """
    omegas = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
    ratios, cancels, lags = [], [], None
    for omega in omegas:
        prof = lag_profile(trajectory, params, float(omega), dirs, dom,
                           chunk=chunk, bin_width=bin_width)
        total = prof["total"]
        ratios.append(prof["cumulative"] / total if total != 0.0
                      else np.full_like(prof["cumulative"], np.nan))
        cancels.append(prof["cancellation"])
        lags = prof["lag"]
    return dict(lag=lags, omega=omegas, ratio=np.asarray(ratios),
                cancellation=np.asarray(cancels))
