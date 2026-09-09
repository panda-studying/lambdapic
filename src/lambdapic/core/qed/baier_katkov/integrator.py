"""Direct double-time integration of the Baier--Katkov kernel.

Reference implementation (phase 1 of `Baier-Katkov-multiparticle.md` sec. 15):
single particle, FP64, plain trapezoid over the full recorded record, no
formation-length windowing, no importance sampling.  Two backends evaluate
the *same* double sum:

* ``"numpy"`` -- chunked outer-product reference path.  Slow; kept as the
  ground truth for cross-checks.
* ``"numba"`` -- parallel, cache-compiled direct summation that exploits the
  Hermitian symmetry ``K(t2, t1) = K(t1, t2)^*`` (upper triangle only) and
  batches every ``(omega, n)`` pair of a spectrum into one pass over the
  ``(t1, t2)`` pairs (kernel shared between directions, phase argument shared
  between frequencies).  Still ``O(Nt^2)`` per record and
  ``O(Nt^2 N_omega N_dir)`` per spectrum -- it is a fast evaluation of the
  reference method, not a different method.

Primary output quantities (natural units, ``m_e = 1``)::

    d2E/dw dO = (alpha/(4 pi^2)) * w^2 * q^2 * Re [ int dt1 dt2 N e^{-i Phi} ]
    d2W/dw dO = d2E/dw dO / w

with ``N`` the kernel (eq. 7.3 / 7.1) and ``Phi`` the recoil phase
(eq. 5.3), both from :mod:`.kernel` / :mod:`.phase`.  ``q`` is the charge
multiplicity (``= 1`` for a single electron).  The prefactor is the
``e^2/(4 pi^2)`` of the Liénard--Wiechert spectrum (Jackson eq. 14.67,
Gaussian units) with ``e^2 = alpha`` for ``hbar = c = 1``; it reproduces the
Schwinger synchrotron spectrum and the Larmor power (validation V4) and is
the same normalization as the LCFA rates in ``core/qed/optical_depth_tables``.
The angle-integrated spectrum ``dW/dw`` is obtained by quadrature over a cone
of directions.

Because the double-time kernel obeys ``K(t2, t1) = K(t1, t2)^*`` the integral
is real.  The numpy path evaluates the full square and keeps the imaginary
part as a round-off diagnostic; the numba path uses the symmetry explicitly,
so its imaginary part is zero by construction.

Local energies (sec. 4.2): passing an ``energy`` history ``eps(t)`` to any
public entry point switches the kernel and phase to the per-vertex forms
``eps(t1), eps(t2)`` with ``eps'_i = eps_i - omega``.  With this per-vertex
recoil the on-shell identity ``(eps_i - eps'_i)^2 = omega^2`` holds vertex
by vertex, so the symmetrized local dot and trace forms still coincide
exactly (verified to round-off by the regression tests); in the classical
``eps'_i = eps_i`` mode the trace form drops the ``omega`` contact term and
is the fundamental one there.  The pair kernel remains symmetric and the
pair phase antisymmetric under ``1 <-> 2``, so Hermitian symmetry
(upper-triangle summation) and reality still hold.  The fixed-``eps`` path
is unchanged and stays the default.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from . import kernel as _kernel
from . import phase as _phase
from .types import Parameters, Spectrum, Trajectory
from .units import fine_structure

try:
    import numba as _numba
except ImportError:  # pragma: no cover - numba is a lambdapic dependency
    _numba = None

__all__ = [
    "PREFACTOR",
    "available_backends",
    "trapz_weights",
    "double_time_integral",
    "double_time_integral_batch",
    "d2_probability",
    "d2_energy",
    "classical_d2_energy",
    "cone_directions",
    "BKIntegrator",
    "compute_spectrum",
]

#: ``e^2 / (4 pi^2)`` in Gaussian natural units (``e^2 = alpha``, ``hbar = c = 1``).
PREFACTOR = fine_structure / (4.0 * np.pi ** 2)

_KERNELS = ("dot", "trace", "velocity")
_KERNEL_ID = {"dot": 0, "trace": 1, "velocity": 2}
_PHASES = ("recoil", "classical")


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------
def available_backends():
    """Backends usable on this installation (``"numpy"`` is always present)."""
    return ("numpy", "numba") if _numba is not None else ("numpy",)


def _resolve_backend(backend):
    if backend == "auto":
        return "numba" if _numba is not None else "numpy"
    if backend == "numba" and _numba is None:
        raise RuntimeError("backend='numba' requested but numba is not importable")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'auto', 'numpy' or 'numba'")
    return backend


# --------------------------------------------------------------------------
# quadrature weights and per-frequency bookkeeping
# --------------------------------------------------------------------------
def trapz_weights(time):
    """1-D trapezoid weights for a sorted (not necessarily uniform) time grid.

    A single-sample grid has no interval to integrate over and returns a
    zero weight; callers should normally reject ``Nt < 2`` trajectories
    earlier (see :class:`.Trajectory`).
    """
    time = np.asarray(time, dtype=np.float64)
    w = np.empty_like(time)
    if len(time) == 1:
        w[0] = 0.0
        return w
    dt = np.diff(time)
    w[0] = dt[0] / 2.0
    w[-1] = dt[-1] / 2.0
    if len(time) > 2:
        w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
    return w


def _final_energy(omega, epsilon, epsilon_prime):
    """``epsilon'`` broadcast to the shape of ``omega``.

    ``None`` means the BK recoil label ``epsilon - omega`` and requires
    ``omega < epsilon``; a scalar or array is broadcast as given (e.g. the
    classical ``epsilon' = epsilon``).
    """
    omega = np.asarray(omega, dtype=np.float64)
    if epsilon_prime is None:
        if np.any(omega >= epsilon):
            raise ValueError(
                f"omega = {omega} must be < epsilon = {epsilon} "
                "(the recoil final-state energy epsilon' = epsilon - omega "
                "must be positive)"
            )
        return epsilon - omega
    eps_p = np.asarray(epsilon_prime, dtype=np.float64)
    return np.array(np.broadcast_to(eps_p, omega.shape), dtype=np.float64)


def _phase_factor(phase, omega, epsilon, eps_p):
    """Frequency multiplying ``(t2 - t1) - n.(r2 - r1)`` in the phase."""
    omega = np.asarray(omega, dtype=np.float64)
    if phase == "recoil":
        return np.asarray(_phase.recoil_frequency(omega, epsilon, eps_p), dtype=np.float64)
    if phase == "classical":
        return omega
    raise ValueError("phase must be 'recoil' or 'classical'")


def _kernel_coefficients(kernel, omega, epsilon, eps_p, mass):
    """Per-frequency ``(a, b)`` such that ``N = a + b * x``.

    ``x`` is the pair-dependent scalar: ``b1.b2 - 1`` for ``"dot"``
    (eq. 7.3), ``(b1 - b2)^2`` for ``"trace"`` (eq. 7.1) and
    ``b1.b2 - (n.b1)(n.b2)`` for ``"velocity"`` (direction dependent).
    """
    omega = np.asarray(omega, dtype=np.float64)
    eps_p = np.asarray(eps_p, dtype=np.float64)
    if kernel == "dot":
        gamma2 = (epsilon / mass) ** 2
        a = (omega ** 2 / gamma2) / (2.0 * eps_p ** 2)
        b = (epsilon ** 2 + eps_p ** 2) / (2.0 * eps_p ** 2)
    elif kernel == "trace":
        a = -(mass ** 2 / (epsilon * eps_p)) * np.ones_like(omega)
        b = -(epsilon ** 2 + eps_p ** 2) / (4.0 * eps_p ** 2)
    else:
        a = np.zeros_like(omega)
        b = np.ones_like(omega)
    return a, b


def _local_vertex_arrays(kernel, omega, energy, phase, mass):
    """Per-(frequency, vertex) arrays ``A, B, f`` of the sec. 4.2 local form.

    The pair kernel is ``N_ij = (A_ki + A_kj) + (B_ki + B_kj) (b_i . b_j - 1)``
    and the pair phase ``Phi_ij = omega_k [ fbar (x_j - x_i) + df (x_j + x_i)/2 ]``
    with ``x = t - n.r``, ``fbar = (f_ki + f_kj)/2``, ``df = f_kj - f_ki``.

    ``phase`` selects the per-vertex final-state energy: ``"recoil"`` ->
    ``eps'_i = eps_i - omega_k`` (BK); ``"classical"`` -> ``eps'_i = eps_i``
    (recoilless, ``f = 1``).  Per-vertex coefficients (see :mod:`.kernel`):

    * trace: ``A_i = -m^2/(2 eps_i eps'_i) + m^2 c_i/(4 eps_i^2)``,
      ``B_i = c_i/4``;
    * dot: ``A_i = m^2 omega_k^2/(4 eps_i^2 eps'_i^2)``, ``B_i = c_i/4``;

    with ``c_i = (eps_i^2 + eps'_i^2)/eps'_i^2``.  Returns arrays of shape
    ``(N_omega, Nt)``.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    energy = np.asarray(energy, dtype=np.float64)
    if phase == "recoil":
        if np.any(omega[:, None] >= energy[None, :]):
            raise ValueError(
                "omega must be < eps(t) at every sample (the per-vertex "
                "final-state energy eps' = eps(t) - omega must be positive)"
            )
        epsp = energy[None, :] - omega[:, None]
        f = energy[None, :] / epsp
    elif phase == "classical":
        epsp = np.broadcast_to(energy, (omega.shape[0], energy.shape[0]))
        f = np.ones_like(epsp)
    else:
        raise ValueError("phase must be 'recoil' or 'classical'")
    c = (energy[None, :] ** 2 + epsp ** 2) / epsp ** 2
    if kernel == "trace":
        A = -mass ** 2 / (2.0 * energy[None, :] * epsp) + mass ** 2 * c / (4.0 * energy[None, :] ** 2)
    elif kernel == "dot":
        A = mass ** 2 * omega[:, None] ** 2 / (4.0 * energy[None, :] ** 2 * epsp ** 2)
    else:
        raise ValueError("local-energy mode supports only the 'dot' and 'trace' kernels")
    return A, 0.25 * c, f


# --------------------------------------------------------------------------
# numpy reference path (full square, complex)
# --------------------------------------------------------------------------
def _double_sum_numpy(time, position, beta, n, factor, kernel, epsilon, omega,
                      eps_p, mass, chunk):
    nt = time.shape[0]
    w = trapz_weights(time)
    if kernel == "dot":
        kern = _kernel.dot_kernel
    elif kernel == "trace":
        kern = _kernel.trace_kernel
    else:
        kern = None  # classical transverse-velocity kernel, no energy dependence

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

        if kern is None:
            N = _kernel.classical_velocity_kernel(bi[:, None, :], beta[None, :, :], n)
        else:
            N = kern(bi[:, None, :], beta[None, :, :], epsilon, omega, mass=mass,
                     epsilon_prime=eps_p)
        K = N * np.exp(-1j * Phi)

        total += np.einsum("i,ij,j->", w[idx], K, w)

    return total


def _double_sum_numpy_local(time, position, beta, n, omega, A, B, f, chunk):
    """Full-square numpy double sum for the local-energy kernel (one omega).

    ``A, B, f`` are the per-vertex ``(Nt,)`` arrays of
    :func:`_local_vertex_arrays` for this frequency.  The phase is evaluated
    in the difference-stable form
    ``Phi = omega [ fbar (dx) + df (x1 + x2)/2 ]`` with ``x = t - n.r``.
    """
    nt = time.shape[0]
    w = trapz_weights(time)
    x = time - position @ n          # (Nt,)  light-cone coordinate t - n.r
    dot = beta @ beta.T              # (Nt, Nt)  b_i . b_j

    total = 0.0 + 0.0j
    for start in range(0, nt, chunk):
        idx = np.arange(start, min(start + chunk, nt))
        Aij = A[idx][:, None] + A[None, :]           # (nchunk, Nt)
        Bij = B[idx][:, None] + B[None, :]
        N = Aij + Bij * (dot[idx] - 1.0)

        dx = x[None, :] - x[idx][:, None]
        xavg = 0.5 * (x[None, :] + x[idx][:, None])
        fbar = 0.5 * (f[idx][:, None] + f[None, :])
        df = f[None, :] - f[idx][:, None]
        Phi = omega * (fbar * dx + df * xavg)

        K = N * np.exp(-1j * Phi)
        total += np.einsum("i,ij,j->", w[idx], K, w)

    return total


# --------------------------------------------------------------------------
# numba path (upper triangle, real, batched over omega and directions)
# --------------------------------------------------------------------------
if _numba is not None:
    from numba import njit, prange

    @njit(cache=True)
    def _accumulate_row(i, time, w, beta, nr, nb, fac, a, b, kernel_id, acc, psi):
        """Add row ``i`` (``j >= i``) of the double sum to ``acc[k, d]``.

        ``Re[N e^{-i Phi}] = N cos(Phi)`` for the real kernels; off-diagonal
        pairs are counted twice (Hermitian symmetry).  ``nr[j, d] = n_d.r_j``
        and ``nb[j, d] = n_d.b_j`` are precomputed projections.
        """
        nt = time.shape[0]
        nk = fac.shape[0]
        nd = psi.shape[0]
        ti = time[i]
        wi = w[i]
        bix = beta[i, 0]
        biy = beta[i, 1]
        biz = beta[i, 2]
        for j in range(i, nt):
            dt = time[j] - ti
            wij = wi * w[j]
            if j > i:
                wij *= 2.0
            dot = bix * beta[j, 0] + biy * beta[j, 1] + biz * beta[j, 2]
            for d in range(nd):
                psi[d] = dt - (nr[j, d] - nr[i, d])
            if kernel_id == 2:
                # classical transverse kernel: x depends on the direction
                for d in range(nd):
                    x = dot - nb[i, d] * nb[j, d]
                    for k in range(nk):
                        acc[k, d] += wij * (a[k] + b[k] * x) * np.cos(fac[k] * psi[d])
            else:
                if kernel_id == 0:
                    x = dot - 1.0
                else:
                    dbx = bix - beta[j, 0]
                    dby = biy - beta[j, 1]
                    dbz = biz - beta[j, 2]
                    x = dbx * dbx + dby * dby + dbz * dbz
                for k in range(nk):
                    wN = wij * (a[k] + b[k] * x)
                    f = fac[k]
                    for d in range(nd):
                        acc[k, d] += wN * np.cos(f * psi[d])

    @njit(parallel=True, cache=True)
    def _double_sum_batch(time, w, beta, nr, nb, fac, a, b, kernel_id, block):
        nt = time.shape[0]
        nk = fac.shape[0]
        nd = nr.shape[1]
        half = (nt + 1) // 2
        nblocks = (half + block - 1) // block
        partial = np.zeros((nblocks, nk, nd))
        for ib in prange(nblocks):
            acc = np.zeros((nk, nd))
            psi = np.empty(nd)
            i0 = ib * block
            i1 = min(i0 + block, half)
            for ii in range(i0, i1):
                # rows ii and nt-1-ii together hold nt-1 off-diagonal pairs,
                # so every iteration carries the same amount of work
                _accumulate_row(ii, time, w, beta, nr, nb, fac, a, b, kernel_id, acc, psi)
                jj = nt - 1 - ii
                if jj != ii:
                    _accumulate_row(jj, time, w, beta, nr, nb, fac, a, b, kernel_id, acc, psi)
            partial[ib, :, :] = acc
        out = np.zeros((nk, nd))
        for ib in range(nblocks):
            out += partial[ib]
        return out


if _numba is not None:
    from numba import njit, prange

    @njit(cache=True)
    def _accumulate_row_local(i, time, w, beta, nr, nb, omega, A, B, f,
                              acc, psi, chi):
        """Add row ``i`` (``j >= i``) of the local-energy double sum.

        Pair kernel ``N_ij = (A[k,i]+A[k,j]) + (B[k,i]+B[k,j]) (b_i.b_j - 1)``
        and pair phase ``Phi = omega_k [ fbar psi_d + df chi_d ]`` with
        ``psi_d = dt - (nr[j,d] - nr[i,d])`` (difference form) and
        ``chi_d = tavg - (nr[j,d] + nr[i,d])/2`` (energy-variation term,
        vanishes for a constant recoil factor).  Off-diagonal pairs are
        counted twice (Hermitian symmetry, preserved because ``N`` is
        symmetric and ``Phi`` antisymmetric under ``i <-> j``).
        """
        nt = time.shape[0]
        nk = omega.shape[0]
        nd = psi.shape[0]
        ti = time[i]
        wi = w[i]
        bix = beta[i, 0]
        biy = beta[i, 1]
        biz = beta[i, 2]
        for j in range(i, nt):
            tj = time[j]
            dt = tj - ti
            tavg = 0.5 * (tj + ti)
            wij = wi * w[j]
            if j > i:
                wij *= 2.0
            dot = bix * beta[j, 0] + biy * beta[j, 1] + biz * beta[j, 2]
            for d in range(nd):
                psi[d] = dt - (nr[j, d] - nr[i, d])
                chi[d] = tavg - 0.5 * (nr[j, d] + nr[i, d])
            x = dot - 1.0
            for k in range(nk):
                wN = wij * ((A[k, i] + A[k, j]) + (B[k, i] + B[k, j]) * x)
                fbar = 0.5 * (f[k, i] + f[k, j])
                df = f[k, j] - f[k, i]
                om = omega[k]
                for d in range(nd):
                    acc[k, d] += wN * np.cos(om * (fbar * psi[d] + df * chi[d]))

    @njit(parallel=True, cache=True)
    def _double_sum_batch_local(time, w, beta, nr, nb, omega, A, B, f, block):
        nt = time.shape[0]
        nk = omega.shape[0]
        nd = nr.shape[1]
        half = (nt + 1) // 2
        nblocks = (half + block - 1) // block
        partial = np.zeros((nblocks, nk, nd))
        for ib in prange(nblocks):
            acc = np.zeros((nk, nd))
            psi = np.empty(nd)
            chi = np.empty(nd)
            i0 = ib * block
            i1 = min(i0 + block, half)
            for ii in range(i0, i1):
                # rows ii and nt-1-ii together hold nt-1 off-diagonal pairs,
                # so every iteration carries the same amount of work
                _accumulate_row_local(ii, time, w, beta, nr, nb, omega, A, B, f,
                                      acc, psi, chi)
                jj = nt - 1 - ii
                if jj != ii:
                    _accumulate_row_local(jj, time, w, beta, nr, nb, omega, A, B, f,
                                          acc, psi, chi)
            partial[ib, :, :] = acc
        out = np.zeros((nk, nd))
        for ib in range(nblocks):
            out += partial[ib]
        return out


# --------------------------------------------------------------------------
# public integrals
# --------------------------------------------------------------------------
def _check_local_energy(energy, nt, kernel, epsilon_prime):
    """Validate an explicit energy history and return it as a C-ordered array."""
    energy = np.ascontiguousarray(np.asarray(energy, dtype=np.float64))
    if kernel == "velocity":
        raise ValueError(
            "local-energy mode applies to the 'dot' and 'trace' kernels, "
            "not to the classical 'velocity' kernel"
        )
    if epsilon_prime is not None:
        raise ValueError(
            "epsilon_prime must be None in local-energy mode "
            "(eps'_i = eps(t_i) - omega, or eps(t_i) for the classical phase)"
        )
    if energy.ndim != 1 or energy.shape[0] != nt:
        raise ValueError("energy must have shape (Nt,)")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy values must be finite and positive")
    return energy


def double_time_integral_batch(time, position, beta, omega, dirs, epsilon, mass=1.0,
                               kernel="dot", epsilon_prime=None, phase="recoil",
                               backend="auto", block=None, chunk=256, energy=None):
    """``Re int dt1 dt2 N e^{-i Phi}`` for every ``(omega[k], dirs[d])`` pair.

    Parameters
    ----------
    time, position, beta : arrays of shape (Nt,) / (Nt, 3)
    omega : (N_omega,) photon angular frequencies (natural units).
    dirs : (N_dir, 3) unit vectors.
    epsilon : float
        incident electron energy.
    epsilon_prime : None, float or (N_omega,) array
        Final-state energy label.  ``None`` -> ``epsilon - omega`` (BK
        recoil, requires ``omega < epsilon``); pass ``epsilon`` for the
        recoilless / classical comparison.  Must be ``None`` in
        local-energy mode.
    kernel : ``"dot"`` (default), ``"trace"`` or ``"velocity"``.
        ``"dot"`` / ``"trace"`` are the two BK kernels (eq. 7.3 / 7.1);
        ``"velocity"`` is the classical transverse-velocity kernel used for
        internal consistency checks (validation V2).
    phase : ``"recoil"`` (default) or ``"classical"``.
        ``"recoil"`` uses the recoil-scaled frequency ``omega*epsilon/epsilon'``
        (eq. 5.3); ``"classical"`` uses the bare ``omega`` (eq. 7.4 reference).
    backend : ``"auto"`` (numba if importable), ``"numba"`` or ``"numpy"``.
    block : int, optional
        numba only: folded rows per parallel task (default adapts to the
        thread count).
    chunk : int
        numpy only: row-block size of the memory-bounded outer product.
    energy : (Nt,) array, optional
        Local electron energy history ``eps(t)`` (sec. 4.2 generalization).
        When given, the kernel and phase use the per-vertex energies
        ``eps(t1), eps(t2)`` with ``eps'_i = eps_i - omega`` (recoil phase)
        or ``eps'_i = eps_i`` (classical phase), and ``epsilon`` /
        ``epsilon_prime`` are no longer used.  Every value must be positive
        and, for the recoil phase, above every ``omega[k]``.  On the mass
        shell ``eps(t) = mass * gamma(t)``.

    Returns
    -------
    (N_omega, N_dir) float array.
    """
    if kernel not in _KERNELS:
        raise ValueError("kernel must be 'dot', 'trace' or 'velocity'")
    if phase not in _PHASES:
        raise ValueError("phase must be 'recoil' or 'classical'")
    backend = _resolve_backend(backend)

    time = np.ascontiguousarray(time, dtype=np.float64)
    position = np.ascontiguousarray(position, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))

    if energy is not None:
        energy = _check_local_energy(energy, time.shape[0], kernel, epsilon_prime)
        A, B, f = _local_vertex_arrays(kernel, omega, energy, phase, mass)
        if backend == "numpy":
            out = np.empty((omega.shape[0], dirs.shape[0]))
            for k in range(omega.shape[0]):
                for d in range(dirs.shape[0]):
                    out[k, d] = _double_sum_numpy_local(
                        time, position, beta, dirs[d], float(omega[k]),
                        A[k], B[k], f[k], chunk,
                    ).real
            return out
        w = trapz_weights(time)
        nr = np.ascontiguousarray(position @ dirs.T)   # (Nt, N_dir)  n_d . r_j
        nb = np.ascontiguousarray(beta @ dirs.T)       # (Nt, N_dir)  n_d . b_j
        if block is None:
            half = (time.shape[0] + 1) // 2
            block = max(1, min(32, half // (4 * _numba.get_num_threads())))
        return _double_sum_batch_local(
            time, w, beta, nr, nb, np.ascontiguousarray(omega),
            np.ascontiguousarray(A), np.ascontiguousarray(B),
            np.ascontiguousarray(f), int(block),
        )

    eps_p = _final_energy(omega, epsilon, epsilon_prime)
    fac = _phase_factor(phase, omega, epsilon, eps_p)

    if backend == "numpy":
        out = np.empty((omega.shape[0], dirs.shape[0]))
        for k in range(omega.shape[0]):
            for d in range(dirs.shape[0]):
                out[k, d] = _double_sum_numpy(
                    time, position, beta, dirs[d], float(fac[k]), kernel, epsilon,
                    float(omega[k]), float(eps_p[k]), mass, chunk,
                ).real
        return out

    a, b = _kernel_coefficients(kernel, omega, epsilon, eps_p, mass)
    w = trapz_weights(time)
    nr = np.ascontiguousarray(position @ dirs.T)   # (Nt, N_dir)  n_d . r_j
    nb = np.ascontiguousarray(beta @ dirs.T)       # (Nt, N_dir)  n_d . b_j
    if block is None:
        half = (time.shape[0] + 1) // 2
        block = max(1, min(32, half // (4 * _numba.get_num_threads())))
    return _double_sum_batch(
        time, w, beta, nr, nb,
        np.ascontiguousarray(fac), np.ascontiguousarray(a), np.ascontiguousarray(b),
        _KERNEL_ID[kernel], int(block),
    )


def double_time_integral(time, position, beta, omega, n, epsilon, mass=1.0,
                         kernel="dot", epsilon_prime=None, phase="recoil",
                         chunk=256, backend="auto", energy=None):
    """Return ``I = int dt1 dt2 N(t1,t2) exp(-i Phi(t1,t2))`` (complex) for
    one ``(omega, n)``.

    See :func:`double_time_integral_batch` for the parameters (``energy`` is
    the optional local energy history).  With the numpy backend the full
    square is summed and the imaginary part is a round-off diagnostic; with
    the numba backend the Hermitian symmetry is used explicitly and the
    imaginary part is exactly zero.
    """
    if kernel not in _KERNELS:
        raise ValueError("kernel must be 'dot', 'trace' or 'velocity'")
    if phase not in _PHASES:
        raise ValueError("phase must be 'recoil' or 'classical'")
    backend = _resolve_backend(backend)
    omega = float(omega)

    if energy is not None:
        time_arr = np.asarray(time, dtype=np.float64)
        energy = _check_local_energy(energy, time_arr.shape[0], kernel, epsilon_prime)
        A, B, f = _local_vertex_arrays(kernel, np.array([omega]), energy, phase, mass)
        if backend == "numpy":
            position = np.asarray(position, dtype=np.float64)
            beta = np.asarray(beta, dtype=np.float64)
            n = np.asarray(n, dtype=np.float64)
            return _double_sum_numpy_local(time_arr, position, beta, n, omega,
                                           A[0], B[0], f[0], chunk)
        out = double_time_integral_batch(time, position, beta, [omega], [n], epsilon,
                                         mass=mass, kernel=kernel, phase=phase,
                                         backend="numba", energy=energy)
        return complex(out[0, 0], 0.0)

    eps_p = float(np.asarray(_final_energy(omega, epsilon, epsilon_prime)))
    factor = float(np.asarray(_phase_factor(phase, omega, epsilon, eps_p)))

    if backend == "numpy":
        time = np.asarray(time, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        return _double_sum_numpy(time, position, beta, n, factor, kernel, epsilon,
                                 omega, eps_p, mass, chunk)

    out = double_time_integral_batch(time, position, beta, [omega], [n], epsilon,
                                     mass=mass, kernel=kernel, epsilon_prime=eps_p,
                                     phase=phase, backend="numba")
    return complex(out[0, 0], 0.0)


def d2_energy(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
              kernel="dot", epsilon_prime=None, phase="recoil", chunk=256,
              diagnostics=False, backend="auto", energy=None):
    """Differential energy spectrum ``d2E/(dw dO)`` for one (omega, n).

    ``energy`` optionally switches to the local-energy (sec. 4.2) form, see
    :func:`double_time_integral_batch`.  With ``diagnostics=True`` returns
    ``(d2E, d2E_imag)`` where ``d2E_imag`` is the imaginary part scaled by
    the same prefactor (round-off level with the numpy backend, identically
    zero with numba); the real part is the physical spectrum.
    """
    I = double_time_integral(time, position, beta, omega, n, epsilon,
                             mass=mass, kernel=kernel, epsilon_prime=epsilon_prime,
                             phase=phase, chunk=chunk, backend=backend,
                             energy=energy)
    pref = PREFACTOR * omega ** 2 * charge ** 2
    d2E = pref * I.real
    if diagnostics:
        return d2E, pref * I.imag
    return d2E


def d2_probability(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
                   kernel="dot", epsilon_prime=None, phase="recoil", chunk=256,
                   diagnostics=False, backend="auto", energy=None):
    """Differential probability ``d2W/(dw dO)`` for one (omega, n).

    With ``diagnostics=True`` returns ``(d2W, d2W_imag)`` (see
    :func:`d2_energy`).
    """
    if diagnostics:
        d2E, d2E_imag = d2_energy(time, position, beta, omega, n, epsilon,
                                   mass=mass, charge=charge, kernel=kernel,
                                   epsilon_prime=epsilon_prime, phase=phase,
                                   chunk=chunk, diagnostics=True, backend=backend,
                                   energy=energy)
        return d2E / omega, d2E_imag / omega
    return d2_energy(time, position, beta, omega, n, epsilon, mass=mass,
                     charge=charge, kernel=kernel, epsilon_prime=epsilon_prime,
                     phase=phase, chunk=chunk, backend=backend,
                     energy=energy) / omega


def classical_d2_energy(time, position, beta, omega, n, charge=1.0):
    """Classical LW reference ``d2E/(dw dO) = (alpha/(4 pi^2)) w^2 |A|^2`` (eq. 7.4).

    ``A`` is the vector amplitude ``int dt v e^{i omega [t - n.r]}`` with
    ``v = n x (n x beta)``; ``|A|^2`` is the sum over its three components.
    """
    A = _kernel.classical_amplitude(time, position, beta, omega, n)
    return PREFACTOR * omega ** 2 * charge ** 2 * np.sum(np.abs(A) ** 2)


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


def _recoil_args(params):
    """Map ``Parameters.recoil`` onto the integrator's ``(phase, epsilon_prime)``.

    ``"baier_katkov"`` -> recoil phase with ``epsilon' = epsilon - omega``
    (default); ``"classical"`` -> classical phase with ``epsilon' = epsilon``.
    """
    if params.recoil == "baier_katkov":
        return "recoil", None
    if params.recoil == "classical":
        return "classical", params.epsilon
    raise ValueError("recoil must be 'baier_katkov' or 'classical'")


class BKIntegrator:
    """Convenience wrapper bundling a trajectory, parameters and a backend.

    When the trajectory carries an ``energy`` history, every spectrum is
    evaluated with the sec. 4.2 local-energy generalization
    (``eps(t1), eps(t2)`` with per-vertex recoil); otherwise the fixed
    incident energy ``Parameters.epsilon`` is used.
    """

    def __init__(self, trajectory: Trajectory, params: Parameters,
                 backend: str = "auto") -> None:
        self.trajectory = trajectory
        self.params = params
        self.backend = _resolve_backend(backend)
        self.time = trajectory.time
        self.position = trajectory.position
        self.beta = trajectory.beta()
        self.energy = trajectory.energy

    # -- grids of (omega, n) ------------------------------------------------
    def d2_energy_batch(self, omega_grid, dirs):
        """``d2E/(dw dO)`` on the full ``(N_omega, N_dir)`` grid."""
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
        phase, eps_p = _recoil_args(self.params)
        if self.energy is not None:
            eps_p = None
        I = double_time_integral_batch(
            self.time, self.position, self.beta, omega_grid, dirs,
            self.params.epsilon, mass=self.params.mass, kernel=self.params.kernel,
            epsilon_prime=eps_p, phase=phase, backend=self.backend,
            energy=self.energy,
        )
        return PREFACTOR * omega_grid[:, None] ** 2 * self.params.charge ** 2 * I

    def d2_probability_batch(self, omega_grid, dirs):
        """``d2W/(dw dO)`` on the full ``(N_omega, N_dir)`` grid."""
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
        return self.d2_energy_batch(omega_grid, dirs) / omega_grid[:, None]

    # -- single (omega, n) -------------------------------------------------
    def d2_energy(self, omega, n):
        return float(self.d2_energy_batch([omega], [n])[0, 0])

    def d2_probability(self, omega, n):
        return self.d2_energy(omega, n) / omega

    # -- spectrum ----------------------------------------------------------
    def compute_spectrum(self, omega_grid, theta_max=None, n_theta=16, n_phi=8,
                         axis=None) -> Spectrum:
        """Angle-integrate ``dW/dw`` and ``dE/dw`` over a cone of directions.

        The cone axis defaults to the initial velocity direction (beam axis);
        ``theta_max`` defaults to ``5 / gamma``.
        """
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
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

        d2E = self.d2_energy_batch(omega_grid, dirs)   # (N_omega, N_dir)
        dE = d2E @ dom
        dW = dE / omega_grid

        meta = dict(
            epsilon=self.params.epsilon,
            local_energy=self.energy is not None,
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
            backend=self.backend,
            units="natural (c=hbar=1, m_e=1)",
        )
        return Spectrum(omega=omega_grid, dW_domega=dW, dE_domega=dE, metadata=meta)


def compute_spectrum(trajectory: Trajectory, params: Parameters, omega_grid,
                     theta_max=None, n_theta=16, n_phi=8, axis=None,
                     backend="auto") -> Spectrum:
    """Convenience function: build an integrator and return the spectrum."""
    return BKIntegrator(trajectory, params, backend=backend).compute_spectrum(
        omega_grid, theta_max=theta_max, n_theta=n_theta, n_phi=n_phi, axis=axis
    )
