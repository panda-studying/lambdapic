"""Baier--Katkov single-particle radiation spectrum (reference implementation).

This package is a **minimal, physics-first, single-particle** implementation of
the quasi-classical BK radiation spectrum, kept fully decoupled from the online
LCFA Monte Carlo pipeline in ``core.qed``.

Scope of this first version (phase 1):

* single particle, FP64, direct double-time integration;
* natural units ``c = hbar = 1``, ``m_e = 1`` (Compton units; SI mapping in
  :mod:`.units`);
* spin-averaged, polarization-summed kernels (:mod:`.kernel`);
* angle-integrated ``dW/domega`` output (:class:`.Spectrum`);
* two adequacy guards on every result (:class:`.SamplingWarning` -- the time
  grid under-resolves the double-time phase and the trapezoid sum is aliased;
  :class:`.RecordLengthWarning` -- the record is short compared with the
  formation time, so the integral is endpoint-dominated and under-reports).
  ``checks="warn"`` by default on every entry point; see
  :func:`integrator.sampling_margin` / :func:`integrator.record_adequacy`.
* an angular guard on every spectrum (:class:`.AngularConvergenceWarning` --
  the direction grid under-resolves the angular integral; probed by grid
  refinement at a few frequencies), plus the free
  ``metadata["angular_edge_fraction"]`` diagnostic.  The default cone is
  adapted to the record by :func:`integrator.default_theta_max`
  (``velocity_swing + 3 max_omega theta_c``) instead of the old fixed
  ``5/gamma``, and the direction grid splits the ``1/gamma`` core off with
  :func:`integrator.cone_directions`.

Explicitly **not** included: multi-particle summation, MPI/GPU, NUFFT/FFT,
formation-length windowing, importance sampling, LCFA/BK hybrid models, or any
performance claim.  This is a reference implementation to pin down units,
phases, recoil, positivity and the classical limit.
"""

from .types import Parameters, Spectrum, Trajectory
from .trajectory import as_trajectory, as_trajectory_si, uniform_trajectory
from .integrator import BKIntegrator, compute_spectrum
from . import kernel, phase, units, result, reference

__all__ = [
    "Trajectory",
    "Parameters",
    "Spectrum",
    "as_trajectory",
    "as_trajectory_si",
    "uniform_trajectory",
    "BKIntegrator",
    "compute_spectrum",
    "kernel",
    "phase",
    "units",
    "result",
    "reference",
]