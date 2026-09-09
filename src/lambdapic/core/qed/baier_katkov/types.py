"""Data structures for the Baier--Katkov reference module.

Everything in this module lives in **natural units** with ``c = hbar = 1`` and
electron mass ``m_e = 1`` (Compton units); see :mod:`.units` for the SI
mapping.  The metric is ``(+---)`` and a vacuum photon satisfies
``k = (omega, omega n)`` with ``|n| = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    """A single-particle classical trajectory sampled on a time grid.

    Attributes
    ----------
    time:
        1-D array of monotonically increasing sample times (natural units).
    position:
        ``(Nt, 3)`` array of positions (natural units).
    momentum:
        Optional ``(Nt, 3)`` array of normalized momentum ``u = gamma beta``.
        When ``None``, ``beta`` is reconstructed by finite-differencing
        ``position`` (with the caveats documented in :mod:`.trajectory`).
    energy:
        Optional ``(Nt,)`` history of the local electron energy ``eps(t_i)``
        (natural units).  When set, the BK integrator switches from the fixed
        incident energy of :class:`Parameters` to the sec. 4.2 local-energy
        generalization: the phase and kernel use ``eps(t1), eps(t2)`` with
        per-vertex recoil ``eps'_i = eps_i - omega``.  On the mass shell the
        default local energy is ``mass * gamma``; supplying ``energy`` is how
        one feeds a PIC energy history (accelerating electrons) into the
        module.  Values must be positive; consistency with ``momentum``
        (``eps_i = mass * sqrt(1 + |u_i|^2)``) is the caller's responsibility
        -- the kernel assumes ``|beta_i|^2 = 1 - m^2/eps_i^2``.
    mass:
        Particle mass in natural units (``m_e = 1`` for electrons).
    charge:
        Charge multiplicity ``|q| / e`` (magnitude only; the spectrum scales
        with ``charge^2``).  Default 1 for a single electron.
    """

    time: np.ndarray
    position: np.ndarray
    momentum: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    mass: float = 1.0
    charge: float = 1.0

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.time.ndim != 1:
            raise ValueError("time must be 1-D")
        if self.time.shape[0] < 2:
            raise ValueError("at least two time samples are required")
        if self.position.ndim != 2 or self.position.shape[1] != 3:
            raise ValueError("position must have shape (Nt, 3)")
        if self.time.shape[0] != self.position.shape[0]:
            raise ValueError("time and position must have the same length")
        if self.momentum is not None:
            self.momentum = np.asarray(self.momentum, dtype=np.float64)
            if self.momentum.shape != self.position.shape:
                raise ValueError("momentum must have shape (Nt, 3)")
        if self.energy is not None:
            self.energy = np.asarray(self.energy, dtype=np.float64)
            if self.energy.shape != (self.time.shape[0],):
                raise ValueError("energy must have shape (Nt,)")
            if not np.all(np.isfinite(self.energy)) or np.any(self.energy <= 0.0):
                raise ValueError("energy values must be finite and positive")

    @property
    def n_samples(self) -> int:
        return int(self.time.shape[0])

    def beta(self) -> np.ndarray:
        """Velocity ``beta = v/c`` at each sample (dimensionless, ``(Nt, 3)``)."""
        if self.momentum is not None:
            u = self.momentum
            gamma = np.sqrt(1.0 + np.einsum("ij,ij->i", u, u))
            return u / gamma[:, None]
        return _beta_from_position(self.time, self.position)

    def gamma(self) -> np.ndarray:
        """Lorentz factor ``gamma`` at each sample."""
        if self.momentum is not None:
            u = self.momentum
            return np.sqrt(1.0 + np.einsum("ij,ij->i", u, u))
        beta = self.beta()
        return 1.0 / np.sqrt(1.0 - np.einsum("ij,ij->i", beta, beta))

    def local_energy(self) -> np.ndarray:
        """Local electron energy ``eps(t_i)`` at each sample.

        Returns the explicit ``energy`` history when one was supplied;
        otherwise falls back to the on-shell value ``mass * gamma`` (exact
        for a trajectory carrying normalized momentum ``u``).
        """
        if self.energy is not None:
            return self.energy
        return self.mass * self.gamma()


def _beta_from_position(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Central-difference velocity with forward/backward endpoints.

    This is only used when no momentum history is supplied.  It amplifies
    sampling noise; supplying ``momentum`` (normalized momentum ``u``) is
    strongly preferred, matching how λPIC stores ``ux, uy, uz``.
    """
    beta = np.empty_like(position)
    dt = np.diff(time)
    beta[1:-1] = (position[2:] - position[:-2]) / (dt[1:] + dt[:-1])[:, None]
    beta[0] = (position[1] - position[0]) / dt[0]
    beta[-1] = (position[-1] - position[-2]) / dt[-1]
    return beta


@dataclass
class Parameters:
    """Physical and bookkeeping conventions of a BK calculation.

    Attributes
    ----------
    epsilon:
        Incident electron energy (natural units, ``gamma m``).  Used for the
        recoil ``epsilon' = epsilon - omega`` and the recoil phase factor
        ``epsilon / epsilon'``.  This fixed value applies while the
        :class:`Trajectory` carries no ``energy`` history; for time-dependent
        external fields (accelerating electrons) supply
        ``Trajectory.energy`` and the integrator switches to the sec. 4.2
        local-energy generalization ``eps(t1), eps(t2)``, at which point
        ``epsilon`` is no longer used by the kernel or phase.
    mass:
        Electron mass in natural units (1).
    charge:
        Charge multiplicity ``|q|/e``; the spectrum prefactor carries
        ``charge^2``.
    spin_averaged:
        Initial spin averaged (True) and final spin summed; the kernels in
        :mod:`.kernel` are built under this convention.
    polarization_summed:
        Photon polarization summed (True).
    recoil:
        ``"baier_katkov"`` uses ``epsilon' = epsilon - omega`` and the recoil
        phase factor ``epsilon/epsilon'`` (default); ``"classical"`` sets
        ``epsilon' = epsilon`` and a unit recoil factor, i.e. the classical
        phase (useful for cross-checking the soft-photon limit).
    kernel:
        ``"dot"`` (velocity form, eq. 7.3 of `Baier-Katkov.md`, the default)
        or ``"trace"`` (trace form, eq. 7.1).  The two are pointwise identical
        on shell (validation V7) and both reproduce the exact quantum
        synchrotron spectrum (validation V8); see :mod:`.kernel`.
    """

    epsilon: float
    mass: float = 1.0
    charge: float = 1.0
    spin_averaged: bool = True
    polarization_summed: bool = True
    recoil: str = "baier_katkov"
    kernel: str = "dot"

    def __post_init__(self) -> None:
        if self.recoil not in ("baier_katkov", "classical"):
            raise ValueError("recoil must be 'baier_katkov' or 'classical'")
        if self.kernel not in ("dot", "trace"):
            raise ValueError("kernel must be 'dot' or 'trace'")
        if not self.spin_averaged or not self.polarization_summed:
            raise NotImplementedError(
                "only spin-averaged, polarization-summed kernels are implemented"
            )

    def epsilon_prime(self, omega) -> np.ndarray:
        """Final-state energy label ``epsilon'`` for photon energy ``omega``."""
        omega = np.asarray(omega, dtype=np.float64)
        if self.recoil == "classical":
            return np.full_like(omega, self.epsilon, dtype=np.float64)
        if np.any(omega >= self.epsilon):
            raise ValueError(
                "omega must be < epsilon "
                "(epsilon' = epsilon - omega must be positive)"
            )
        return self.epsilon - omega

    def recoil_factor(self, omega) -> np.ndarray:
        """Recoil phase factor ``epsilon / epsilon'`` for photon energy ``omega``."""
        omega = np.asarray(omega, dtype=np.float64)
        return self.epsilon / self.epsilon_prime(omega)


@dataclass
class Spectrum:
    """Single-particle differential spectrum.

    Attributes
    ----------
    omega:
        1-D photon angular-frequency grid (natural units, ``m_e``).
    dW_domega:
        Differential *probability* per unit photon energy, ``dW/domega``,
        angle-integrated (dimensionless, natural units).  This is the primary
        output.  For an ensemble it is summed as ``S = sum_i w_i S_i`` with
        ``w_i`` the macro-particle weight (not implemented here).
    dE_domega:
        Optional differential radiated *energy* per unit photon energy,
        ``dE/domega = omega * dW/domega`` (natural energy units).
    metadata:
        Dictionary of provenance: trajectory, parameters, grids, conventions.
    """

    omega: np.ndarray
    dW_domega: np.ndarray
    dE_domega: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.omega = np.asarray(self.omega, dtype=np.float64)
        self.dW_domega = np.asarray(self.dW_domega, dtype=np.float64)
        if self.dE_domega is not None:
            self.dE_domega = np.asarray(self.dE_domega, dtype=np.float64)
        if self.omega.shape != self.dW_domega.shape:
            raise ValueError("omega and dW_domega must have the same shape")
        if self.dE_domega is not None and self.dE_domega.shape != self.omega.shape:
            raise ValueError("dE_domega must have the same shape as omega")

    def total_probability(self) -> float:
        """Integrated emission probability ``W = int domega dW/domega``."""
        return float(np.trapezoid(self.dW_domega, self.omega))

    def total_energy(self) -> float:
        """Integrated radiated energy ``E = int domega dE/domega``."""
        dE = self.dE_domega
        if dE is None:
            dE = self.omega * self.dW_domega
        return float(np.trapezoid(dE, self.omega))
