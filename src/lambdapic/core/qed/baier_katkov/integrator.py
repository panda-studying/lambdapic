"""Direct double-time integration of the Baier--Katkov kernel.

Reference implementation (phase 1 of `Baier-Katkov-multiparticle.md` sec. 15):
single particle, FP64, plain trapezoid over the full recorded record, no GPU,
no formation-length windowing, no importance sampling.  It is intentionally
slow and explicit; it exists to fix conventions, not to be fast.

Primary output quantities (natural units, ``m_e = 1``)::

    d2E/dw dO = (alpha/pi) * w^2 * q^2 * Re [ int dt1 dt2 N e^{-i Phi} ]
    d2W/dw dO = d2E/dw dO / w

with ``N`` the kernel (eq. 7.3 / 7.1) and ``Phi`` the recoil phase
(eq. 5.3), both from :mod:`.kernel` / :mod:`.phase`.  ``q`` is the charge
multiplicity (``= 1`` for a single electron).  The angle-integrated spectrum
``dW/dw`` is obtained by quadrature over a cone of directions.

Because the double-time kernel obeys ``K(t2, t1) = K(t1, t2)^*`` the integral
is real; the imaginary part is kept as a numerical diagnostic, never dropped
silently.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from . import kernel as _kernel
from . import phase as _phase
from .types import Parameters, Spectrum, Trajectory
from .units import fine_structure

__all__ = [
    "trapz_weights",
    "double_time_integral",
    "d2_probability",
    "d2_energy",
    "classical_d2_energy",
    "cone_directions",
    "BKIntegrator",
    "compute_spectrum",
]


def trapz_weights(time):
    """1-D trapezoid weights for a sorted (not necessarily uniform) time grid."""
    time = np.asarray(time, dtype=np.float64)
    w = np.empty_like(time)
    dt = np.diff(time)
    w[0] = dt[0] / 2.0
    w[-1] = dt[-1] / 2.0
    if len(time) > 2:
        w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
    elif len(time) == 2:
        w[0] = w[-1] = (time[1] - time[0]) / 2.0
    return w


def double_time_integral(time, position, beta, omega, n, epsilon, mass=1.0,
                         kernel="dot", epsilon_prime=None, phase="recoil",
                         chunk=256):
    """Return ``I = int dt1 dt2 N(t1,t2) exp(-i Phi(t1,t2))`` (complex).

    Parameters
    ----------
    time, position, beta : arrays of shape (Nt,) / (Nt, 3)
    omega : float
        photon angular frequency (natural units).
    n : (3,) unit vector.
    epsilon : float
        incident electron energy.
    kernel : ``"dot"`` (default) or ``"trace"``.
    epsilon_prime : float, optional
        Final-state energy label.  Default ``epsilon - omega`` (BK recoil);
        pass ``epsilon`` for the recoilless / classical comparison.
    phase : ``"recoil"`` (default) or ``"classical"``.
        ``"recoil"`` uses the recoil-scaled frequency ``omega*epsilon/epsilon'``
        (eq. 5.3); ``"classical"`` uses the bare ``omega`` (eq. 7.4 reference).
    chunk : int
        row-block size for the memory-bounded outer product.
    """
    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    nt = time.shape[0]

    w = trapz_weights(time)
    if epsilon_prime is None:
        epsilon_prime = epsilon - omega
    if phase == "recoil":
        factor = _phase.recoil_frequency(omega, epsilon, epsilon_prime)
    elif phase == "classical":
        factor = float(omega)
    else:
        raise ValueError("phase must be 'recoil' or 'classical'")

    kern = _kernel.dot_kernel if kernel == "dot" else _kernel.trace_kernel

    total = 0.0 + 0.0j
    for start in range(0, nt, chunk):
        idx = np.arange(start, min(start + chunk, nt))
        ti = time[idx]                       # (nchunk,)
        ri = position[idx]                   # (nchunk, 3)
        bi = beta[idx]                       # (nchunk, 3)

        dt = time[None, :] - ti[:, None]                 # (nchunk, Nt)
        dr = position[None, :, :] - ri[:, None, :]       # (nchunk, Nt, 3)
        n_dot_dr = np.tensordot(dr, n, axes=([2], [0]))  # (nchunk, Nt)
        Phi = factor * (dt - n_dot_dr)

        if kernel == "velocity":
            N = _kernel.classical_velocity_kernel(bi[:, None, :], beta[None, :, :], n)
        else:
            N = kern(bi[:, None, :], beta[None, :, :], epsilon, omega, mass=mass,
                     epsilon_prime=epsilon_prime)
        K = N * np.exp(-1j * Phi)

        total += np.einsum("i,ij,j->", w[idx], K, w)

    return total


def d2_energy(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
              kernel="dot", epsilon_prime=None, phase="recoil", chunk=256):
    """Differential energy spectrum ``d2E/(dw dO)`` for one (omega, n)."""
    I = double_time_integral(time, position, beta, omega, n, epsilon,
                             mass=mass, kernel=kernel, epsilon_prime=epsilon_prime,
                             phase=phase, chunk=chunk)
    return (fine_structure / np.pi) * omega ** 2 * charge ** 2 * I.real


def d2_probability(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
                   kernel="dot", epsilon_prime=None, phase="recoil", chunk=256):
    """Differential probability ``d2W/(dw dO)`` for one (omega, n)."""
    return d2_energy(time, position, beta, omega, n, epsilon, mass=mass,
                     charge=charge, kernel=kernel, epsilon_prime=epsilon_prime,
                     phase=phase, chunk=chunk) / omega


def classical_d2_energy(time, position, beta, omega, n, charge=1.0):
    """Classical LW reference ``d2E/(dw dO) = (alpha/pi) w^2 |A|^2`` (eq. 7.4).

    ``A`` is the vector amplitude ``int dt v e^{i omega [t - n.r]}`` with
    ``v = n x (n x beta)``; ``|A|^2`` is the sum over its three components.
    """
    A = _kernel.classical_amplitude(time, position, beta, omega, n)
    return (fine_structure / np.pi) * omega ** 2 * charge ** 2 * np.sum(np.abs(A) ** 2)


def cone_directions(axis, theta_max, n_theta, n_phi):
    """Directions and solid-angle weights on a cone around ``axis``.

    ``cos(theta)`` is sampled with Gauss--Legendre over ``[cos(theta_max), 1]``
    and ``phi`` uniformly over ``[0, 2 pi)``.  Returns ``(dirs, dOmega)`` with
    ``dirs`` of shape ``(n_theta * n_phi, 3)`` and ``dOmega`` the per-sample
    solid angle (summing to the full cone solid angle ``2 pi (1 - cos(theta_max))``).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    x, wx = roots_legendre(n_theta)             # x in (-1, 1)
    cos_min = np.cos(theta_max)
    cos_theta = 0.5 * (1.0 - cos_min) * x + 0.5 * (1.0 + cos_min)
    dcos = 0.5 * (1.0 - cos_min) * wx           # d(cos theta) weight
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    # orthonormal frame with e3 = axis
    if abs(axis[2]) < 0.999:
        e1 = np.cross([0.0, 0.0, 1.0], axis)
    else:
        e1 = np.cross([1.0, 0.0, 0.0], axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    dirs = []
    dom = []
    for ct, th, dct in zip(cos_theta, theta, dcos):
        st = np.sin(th)
        for p in phi:
            d = st * np.cos(p) * e1 + st * np.sin(p) * e2 + ct * axis
            dirs.append(d)
            dom.append(dct * dphi)
    return np.asarray(dirs), np.asarray(dom)


class BKIntegrator:
    """Convenience wrapper bundling a trajectory and parameters."""

    def __init__(self, trajectory: Trajectory, params: Parameters) -> None:
        self.trajectory = trajectory
        self.params = params
        self.time = trajectory.time
        self.position = trajectory.position
        self.beta = trajectory.beta()

    # -- single (omega, n) -------------------------------------------------
    def d2_probability(self, omega, n):
        return d2_probability(
            self.time, self.position, self.beta, omega, n,
            self.params.epsilon, mass=self.params.mass, charge=self.params.charge,
            kernel=self.params.kernel,
        )

    def d2_energy(self, omega, n):
        return d2_energy(
            self.time, self.position, self.beta, omega, n,
            self.params.epsilon, mass=self.params.mass, charge=self.params.charge,
            kernel=self.params.kernel,
        )

    # -- spectrum ----------------------------------------------------------
    def compute_spectrum(self, omega_grid, theta_max=None, n_theta=16, n_phi=8,
                         axis=None) -> Spectrum:
        """Angle-integrate ``dW/dw`` and ``dE/dw`` over a cone of directions.

        The cone axis defaults to the initial velocity direction (beam axis);
        ``theta_max`` defaults to ``5 / gamma``.
        """
        omega_grid = np.asarray(omega_grid, dtype=np.float64)
        if axis is None:
            beta0 = self.beta[0]
            if np.linalg.norm(beta0) < 1e-12:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = beta0 / np.linalg.norm(beta0)
        if theta_max is None:
            gamma0 = float(self.trajectory.gamma()[0])
            theta_max = 5.0 / max(gamma0, 1e-6)

        dirs, dom = cone_directions(axis, theta_max, n_theta, n_phi)

        dW = np.zeros_like(omega_grid)
        dE = np.zeros_like(omega_grid)
        for k, om in enumerate(omega_grid):
            acc_W = 0.0
            acc_E = 0.0
            for nd, dO in zip(dirs, dom):
                acc_W += self.d2_probability(om, nd) * dO
                acc_E += self.d2_energy(om, nd) * dO
            dW[k] = acc_W
            dE[k] = acc_E

        meta = dict(
            epsilon=self.params.epsilon,
            mass=self.params.mass,
            charge=self.params.charge,
            recoil=self.params.recoil,
            kernel=self.params.kernel,
            spin_averaged=self.params.spin_averaged,
            polarization_summed=self.params.polarization_summed,
            n_samples=int(self.time.shape[0]),
            t_span=(float(self.time[0]), float(self.time[-1])),
            theta_max=float(theta_max),
            n_theta=int(n_theta),
            n_phi=int(n_phi),
            units="natural (c=hbar=1, m_e=1)",
        )
        return Spectrum(omega=omega_grid, dW_domega=dW, dE_domega=dE, metadata=meta)


def compute_spectrum(trajectory: Trajectory, params: Parameters, omega_grid,
                     theta_max=None, n_theta=16, n_phi=8, axis=None) -> Spectrum:
    """Convenience function: build an integrator and return the spectrum."""
    return BKIntegrator(trajectory, params).compute_spectrum(
        omega_grid, theta_max=theta_max, n_theta=n_theta, n_phi=n_phi, axis=axis
    )
