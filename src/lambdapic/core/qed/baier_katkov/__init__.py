"""Baier--Katkov single-particle radiation spectrum (reference implementation).

This package is a **minimal, physics-first, single-particle** implementation of
the quasi-classical BK radiation spectrum, kept fully decoupled from the online
LCFA Monte Carlo pipeline in ``core.qed``.

Scope of this first version (phase 1):

* single particle, FP64, direct double-time integration;
* natural units ``c = hbar = 1``, ``m_e = 1`` (Compton units; SI mapping in
  :mod:`.units`);
* spin-averaged, polarization-summed kernels (:mod:`.kernel`);
* angle-integrated ``dW/domega`` output (:class:`.Spectrum`).

Explicitly **not** included: multi-particle summation, MPI/GPU, NUFFT/FFT,
formation-length windowing, importance sampling, LCFA/BK hybrid models, or any
performance claim.  This is a reference implementation to pin down units,
phases, recoil, positivity and the classical limit.
"""

from .types import Parameters, Spectrum, Trajectory
from .trajectory import as_trajectory, uniform_trajectory
from .integrator import BKIntegrator, compute_spectrum
from . import kernel, phase, units, result

__all__ = [
    "Trajectory",
    "Parameters",
    "Spectrum",
    "as_trajectory",
    "uniform_trajectory",
    "BKIntegrator",
    "compute_spectrum",
    "kernel",
    "phase",
    "units",
    "result",
]