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

Two adequacy guards protect the result from silently returning non-physics
(``checks="warn"`` by default, ``"raise"`` / ``"ignore"`` available; see
:func:`sampling_margin`, :func:`record_adequacy`):

* **Trapezoid aliasing.**  The integrand oscillates at up to
  ``omega (eps/eps') max|1 - n.v|``, so the step must satisfy the Nyquist
  bound ``dt * that < pi``.  Beyond it the double sum returns an artifact --
  arbitrarily large and of either sign -- not a spectrum.  On a circle this
  is the familiar ``Nt > 4m`` (the ``Nt > 2m`` often quoted permits one
  sample per oscillation and is measurably too lax).
* **Record shorter than the formation time.**  For ``L <~ tau_f(omega)``
  the integral is dominated by the hard record endpoints and under-reports,
  going negative in the worst case.  ``tau_f`` grows towards low ``omega``,
  so the soft-photon region is the first to suffer.

Both are *diagnostics*, not corrections: they never change the computed
value, only say whether it can be trusted.

A third guard covers the angular integral
(:class:`AngularConvergenceWarning`, raised by
:meth:`BKIntegrator.compute_spectrum`): doubling ``n_theta``/``n_phi`` at a few
probe frequencies must not move the angle-integrated ``dE/domega`` by more than
``ANGULAR_RTOL``.  This one is not free -- it costs about ``4 n_probe /
N_omega`` of the base angular integral -- but it catches exactly the failure
mode the historical fixed ``5/gamma`` cone had, where a single wide
Gauss--Legendre panel silently under-reported the soft-photon end.  The cone
itself is now adapted to the record (:func:`default_theta_max`) and the
``1/gamma`` core is split off as its own panel (:func:`cone_directions`).
"""

from __future__ import annotations

import warnings

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
    "cone_bands",
    "sampling_margin",
    "formation_time",
    "record_adequacy",
    "velocity_swing",
    "emission_half_angle",
    "default_theta_max",
    "angular_edge_fraction",
    "ANGULAR_RTOL",
    "SamplingWarning",
    "RecordLengthWarning",
    "AngularConvergenceWarning",
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


def _double_sum_numpy_local(time, position, beta, n, omega, A, B, T, R, chunk):
    """Full-square numpy double sum for the local-energy kernel (one omega).

    ``A, B`` are the per-vertex ``(Nt,)`` kernel coefficients for this frequency
    and ``T, R`` the cumulative phase tables of :func:`phase.local_phase_tables`
    (``(Nt,)`` and ``(Nt, 3)`` here).  The phase is the accumulated one,
    ``Phi_ij = omega [(T_j - T_i) - n.(R_j - R_i)]`` -- see that function for
    why an endpoint expression is not translation invariant.
    """
    nt = time.shape[0]
    w = trapz_weights(time)
    dot = beta @ beta.T              # (Nt, Nt)  b_i . b_j
    nR = R @ n                       # (Nt,)  n . R_i

    total = 0.0 + 0.0j
    for start in range(0, nt, chunk):
        idx = np.arange(start, min(start + chunk, nt))
        Aij = A[idx][:, None] + A[None, :]           # (nchunk, Nt)
        Bij = B[idx][:, None] + B[None, :]
        N = Aij + Bij * (dot[idx] - 1.0)

        Phi = omega * ((T[None, :] - T[idx][:, None])
                       - (nR[None, :] - nR[idx][:, None]))

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
    def _accumulate_row_local(i, time, w, beta, nr, nb, omega, A, B, T, R, dirs,
                              acc):
        """Add row ``i`` (``j >= i``) of the local-energy double sum.

        Pair kernel ``N_ij = (A[k,i]+A[k,j]) + (B[k,i]+B[k,j]) (b_i.b_j - 1)``
        and pair phase ``Phi_ij = omega_k [(T[k,j]-T[k,i]) - n_d.(R[k,j]-R[k,i])]``
        -- the accumulated form of :func:`phase.local_phase_tables`, which is
        translation invariant (an endpoint expression ``omega[f_j x_j - f_i x_i]``
        is not, when the recoil factor varies).  Off-diagonal pairs are counted
        twice (Hermitian symmetry: ``N`` is symmetric and ``Phi`` antisymmetric
        under ``i <-> j``).
        """
        nt = time.shape[0]
        nk = omega.shape[0]
        nd = nr.shape[1]
        ti = time[i]
        wi = w[i]
        bix = beta[i, 0]
        biy = beta[i, 1]
        biz = beta[i, 2]
        for j in range(i, nt):
            tj = time[j]
            wij = wi * w[j]
            if j > i:
                wij *= 2.0
            dot = bix * beta[j, 0] + biy * beta[j, 1] + biz * beta[j, 2]
            x = dot - 1.0
            for k in range(nk):
                wN = wij * ((A[k, i] + A[k, j]) + (B[k, i] + B[k, j]) * x)
                om = omega[k]
                dT = T[k, j] - T[k, i]
                r0 = R[k, j, 0] - R[k, i, 0]
                r1 = R[k, j, 1] - R[k, i, 1]
                r2 = R[k, j, 2] - R[k, i, 2]
                for d in range(nd):
                    dR = (r0 * dirs[d, 0] + r1 * dirs[d, 1] + r2 * dirs[d, 2])
                    acc[k, d] += wN * np.cos(om * (dT - dR))
        _ = tj

    @njit(parallel=True, cache=True)
    def _double_sum_batch_local(time, w, beta, nr, nb, omega, A, B, T, R, dirs,
                                block):
        nt = time.shape[0]
        nk = omega.shape[0]
        nd = nr.shape[1]
        half = (nt + 1) // 2
        nblocks = (half + block - 1) // block
        partial = np.zeros((nblocks, nk, nd))
        for ib in prange(nblocks):
            acc = np.zeros((nk, nd))
            i0 = ib * block
            i1 = min(i0 + block, half)
            for ii in range(i0, i1):
                # rows ii and nt-1-ii together hold nt-1 off-diagonal pairs,
                # so every iteration carries the same amount of work
                _accumulate_row_local(ii, time, w, beta, nr, nb, omega, A, B,
                                      T, R, dirs, acc)
                jj = nt - 1 - ii
                if jj != ii:
                    _accumulate_row_local(jj, time, w, beta, nr, nb, omega, A, B,
                                          T, R, dirs, acc)
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
                               backend="auto", block=None, chunk=256, energy=None,
                               checks="warn"):
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
    checks : ``"warn"`` (default), ``"raise"`` or ``"ignore"``
        Adequacy guards on the result.  ``"warn"`` emits
        :class:`SamplingWarning` when the time grid under-resolves the
        double-time phase -- the trapezoid sum is then aliased and the value
        is a sampling artifact, arbitrarily large and of either sign, not a
        spectrum -- and :class:`RecordLengthWarning` when the record is short
        compared with the formation time, where the integral is
        endpoint-dominated and under-reports.  ``"raise"`` turns either into
        an exception; ``"ignore"`` silences both.  Use
        :func:`sampling_margin` / :func:`record_adequacy` (or
        :meth:`BKIntegrator.adequacy`) to inspect the margins directly.

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
        T, R = _phase.local_phase_tables(time, position, f)
        # the guard's ``factor`` is the phase rate, omega * eps/eps' -- the same
        # quantity the fixed-energy path passes via _phase_factor.  Passing the
        # dimensionless eps/eps' alone is off by exactly 1/omega (measured: it
        # reported 2.15 where the true margin is 1.08 at omega = 0.5, and 42459
        # instead of 0.129 at LWFA scales).
        _run_adequacy_checks(time, beta, dirs, omega,
                             omega * f.max(axis=1) if f.ndim > 1 else omega * f,
                             epsilon, epsilon_prime, phase, energy, checks)
        if backend == "numpy":
            out = np.empty((omega.shape[0], dirs.shape[0]))
            for k in range(omega.shape[0]):
                for d in range(dirs.shape[0]):
                    out[k, d] = _double_sum_numpy_local(
                        time, position, beta, dirs[d], float(omega[k]),
                        A[k], B[k], T[k], R[k], chunk,
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
            np.ascontiguousarray(T), np.ascontiguousarray(R),
            np.ascontiguousarray(dirs), int(block),
        )

    eps_p = _final_energy(omega, epsilon, epsilon_prime)
    fac = _phase_factor(phase, omega, epsilon, eps_p)
    _run_adequacy_checks(time, beta, dirs, omega, fac, epsilon, epsilon_prime,
                         phase, None, checks)

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


def _as_scalar_omega(omega) -> float:
    """Coerce a single photon energy to ``float``.

    Accepts a python scalar, a 0-d array or a size-1 array -- the single-point
    entries are routinely called as ``grid[i:i + 1]``, and ``float(np.array([w]))``
    raises ``TypeError`` on NumPy >= 2.
    """
    arr = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if arr.size != 1:
        raise ValueError(f"expected a single omega, got array of size {arr.size}")
    return float(arr[0])


def double_time_integral(time, position, beta, omega, n, epsilon, mass=1.0,
                         kernel="dot", epsilon_prime=None, phase="recoil",
                         chunk=256, backend="auto", energy=None, checks="warn"):
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
    omega = _as_scalar_omega(omega)

    if energy is not None:
        time_arr = np.asarray(time, dtype=np.float64)
        energy = _check_local_energy(energy, time_arr.shape[0], kernel, epsilon_prime)
        A, B, f = _local_vertex_arrays(kernel, np.array([omega]), energy, phase, mass)
        T, R = _phase.local_phase_tables(time_arr, position, f)
        if backend == "numpy":
            _run_adequacy_checks(time_arr, np.asarray(beta, dtype=np.float64),
                                 np.atleast_2d(np.asarray(n, dtype=np.float64)),
                                 np.array([omega]),
                                 np.array([omega]) * f.max(axis=1), epsilon,
                                 epsilon_prime, phase, energy, checks)
            position = np.asarray(position, dtype=np.float64)
            beta = np.asarray(beta, dtype=np.float64)
            n = np.asarray(n, dtype=np.float64)
            return _double_sum_numpy_local(time_arr, position, beta, n, omega,
                                           A[0], B[0], T[0], R[0], chunk)
        out = double_time_integral_batch(time, position, beta, [omega], [n], epsilon,
                                         mass=mass, kernel=kernel, phase=phase,
                                         backend="numba", energy=energy, checks=checks)
        return complex(out[0, 0], 0.0)

    eps_p = float(np.asarray(_final_energy(omega, epsilon, epsilon_prime)))
    factor = float(np.asarray(_phase_factor(phase, omega, epsilon, eps_p)))

    if backend == "numpy":
        time_arr = np.asarray(time, dtype=np.float64)
        _run_adequacy_checks(time_arr, np.asarray(beta, dtype=np.float64),
                             np.atleast_2d(np.asarray(n, dtype=np.float64)),
                             np.array([omega]), np.array([factor]), epsilon,
                             epsilon_prime, phase, None, checks)
        position = np.asarray(position, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        return _double_sum_numpy(time_arr, position, beta, n, factor, kernel, epsilon,
                                 omega, eps_p, mass, chunk)

    out = double_time_integral_batch(time, position, beta, [omega], [n], epsilon,
                                     mass=mass, kernel=kernel, epsilon_prime=eps_p,
                                     phase=phase, backend="numba", checks=checks)
    return complex(out[0, 0], 0.0)


def d2_energy(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
              kernel="dot", epsilon_prime=None, phase="recoil", chunk=256,
              diagnostics=False, backend="auto", energy=None, checks="warn"):
    """Differential energy spectrum ``d2E/(dw dO)`` for one (omega, n).

    ``energy`` optionally switches to the local-energy (sec. 4.2) form, see
    :func:`double_time_integral_batch`.  With ``diagnostics=True`` returns
    ``(d2E, d2E_imag)`` where ``d2E_imag`` is the imaginary part scaled by
    the same prefactor (round-off level with the numpy backend, identically
    zero with numba); the real part is the physical spectrum.  ``checks``
    selects the aliasing / record-length guards (``"warn"`` by default).
    """
    I = double_time_integral(time, position, beta, omega, n, epsilon,
                             mass=mass, kernel=kernel, epsilon_prime=epsilon_prime,
                             phase=phase, chunk=chunk, backend=backend,
                             energy=energy, checks=checks)
    pref = PREFACTOR * _as_scalar_omega(omega) ** 2 * charge ** 2
    d2E = pref * I.real
    if diagnostics:
        return d2E, pref * I.imag
    return d2E


def d2_probability(time, position, beta, omega, n, epsilon, mass=1.0, charge=1.0,
                   kernel="dot", epsilon_prime=None, phase="recoil", chunk=256,
                   diagnostics=False, backend="auto", energy=None, checks="warn"):
    """Differential probability ``d2W/(dw dO)`` for one (omega, n).

    With ``diagnostics=True`` returns ``(d2W, d2W_imag)`` (see
    :func:`d2_energy`).  ``dE/domega`` carries ``omega^2``, so the quotient
    ``dW/domega = dE/domega / omega`` is ``0/0`` at ``omega = 0``; zero (and
    negative) photon energies are rejected here rather than returning a NaN.
    """
    omega = _as_scalar_omega(omega)
    if omega <= 0.0:
        raise ValueError(
            "omega must be strictly positive: dW/domega = dE/domega / omega is "
            f"undefined at omega = 0 (got {omega!r}); use d2_energy for omega = 0"
        )
    if diagnostics:
        d2E, d2E_imag = d2_energy(time, position, beta, omega, n, epsilon,
                                   mass=mass, charge=charge, kernel=kernel,
                                   epsilon_prime=epsilon_prime, phase=phase,
                                   chunk=chunk, diagnostics=True, backend=backend,
                                   energy=energy, checks=checks)
        return d2E / omega, d2E_imag / omega
    return d2_energy(time, position, beta, omega, n, epsilon, mass=mass,
                     charge=charge, kernel=kernel, epsilon_prime=epsilon_prime,
                     phase=phase, chunk=chunk, backend=backend,
                     energy=energy, checks=checks) / omega


def classical_d2_energy(time, position, beta, omega, n, charge=1.0):
    """Classical LW reference ``d2E/(dw dO) = (alpha/(4 pi^2)) w^2 |A|^2`` (eq. 7.4).

    ``A`` is the vector amplitude ``int dt v e^{i omega [t - n.r]}`` with
    ``v = n x (n x beta)``; ``|A|^2`` is the sum over its three components.
    """
    A = _kernel.classical_amplitude(time, position, beta, omega, n)
    return PREFACTOR * omega ** 2 * charge ** 2 * np.sum(np.abs(A) ** 2)


def cone_bands(theta_max, n_theta, split=None, n_inner=None):
    """The ``(bound, n_nodes)`` panels :func:`cone_directions` will actually use.

    Single source of truth for the two-panel rule.  Exposing the effective
    panels is what lets callers report the grid they *got* rather than the one
    they *asked for* (the metadata used to record an ignored ``split``).

    ``split`` and ``n_inner`` must be supplied together; a ``split`` outside
    ``(0, theta_max)`` raises rather than silently falling back to a single
    panel.  Returns a one-element list ``[(theta_max, n_theta)]``, or two
    elements ``[(split, n_inner), (theta_max, n_theta)]`` (inner panel first).
    """
    theta_max = float(theta_max)
    if split is None and n_inner is None:
        return [(theta_max, int(n_theta))]
    if split is None or n_inner is None:
        raise ValueError(
            "split and n_inner must be given together: the two-panel rule needs "
            "both the split angle and the number of inner nodes"
        )
    split = float(split)
    if not 0.0 < split < theta_max:
        raise ValueError(
            f"split must satisfy 0 < split < theta_max, got split={split!r} "
            f"with theta_max={theta_max!r}"
        )
    return [(split, int(n_inner)), (theta_max, int(n_theta))]


def cone_directions(axis, theta_max, n_theta, n_phi, split=None, n_inner=None):
    """Directions and solid-angle weights on a cone around ``axis``.

    ``cos(theta)`` is sampled with Gauss--Legendre over ``[cos(theta_max), 1]``
    and ``phi`` uniformly over ``[0, 2 pi)``.  Returns ``(dirs, dOmega)`` with
    ``dirs`` of shape ``(n_theta * n_phi, 3)`` and ``dOmega`` the per-sample
    solid angle (summing to the full cone solid angle ``2 pi (1 - cos(theta_max))``).

    ``split`` (radians), together with ``n_inner``, enables a **two-panel**
    rule: Gauss--Legendre on ``[0, split]`` with ``n_inner`` nodes plus
    ``[split, theta_max]`` with ``n_theta``.  A single wide panel puts its
    nodes where ``cos(theta)`` is smooth, not where the emission is.  Measured
    on a ``gamma = 10``, ``chi = 0.5`` circle at integer recoil-shifted
    harmonics, against a converged reference: a *single* panel spanning
    ``[0, pi]`` with ``n_theta = 16`` under-reports the angle-integrated
    ``dE/domega`` by 20% at ``delta = 0.3`` and by 97% at ``delta = 0.5``, and
    needs ``n_theta ~ 64`` to converge; splitting the ``1/gamma`` core off with
    ``split = 8/gamma`` and ``n_inner ~ 24`` reproduces the converged value to
    the printed digits.  Directions are ordered inner panel first, then outer,
    each grouped by ``theta`` with the ``n_phi`` azimuths innermost.  **Within
    each panel ``theta`` runs descending** (the Gauss--Legendre nodes are stored
    ascending in ``cos(theta)``), so the node nearest ``theta_max`` is the
    *first* ``n_phi`` entries of the *last* panel -- see
    :func:`angular_edge_fraction`, which relies on this.
    """
    bands = cone_bands(theta_max, n_theta, split, n_inner)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    cos_theta = []
    dcos = []
    upper = 1.0
    for bound, n in bands:
        lower = np.cos(bound)
        x, wx = roots_legendre(n)               # x in (-1, 1)
        cos_theta.append(0.5 * (upper - lower) * x + 0.5 * (upper + lower))
        dcos.append(0.5 * (upper - lower) * wx)  # d(cos theta) weight
        upper = lower
    cos_theta = np.concatenate(cos_theta)
    dcos = np.concatenate(dcos)
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


# --------------------------------------------------------------------------
# adequacy diagnostics: trapezoid aliasing and record-length truncation
# --------------------------------------------------------------------------
class SamplingWarning(UserWarning):
    """The time grid under-resolves the double-time phase (trapezoid aliasing).

    Raised when ``dt * max|dPhi/dt| >= pi``.  The double sum is then a
    sampling artifact -- it can be arbitrarily large and of either sign -- and
    not physics.  Remedy: refine ``time`` (or evaluate at smaller ``omega``).
    """


class RecordLengthWarning(UserWarning):
    """The record is short compared with the formation time (truncation deficit).

    Raised when ``L < 1.5 tau_f(omega)``.  The double integral is then
    dominated by the hard record endpoints rather than by the bulk, so the
    spectrum is under-reported and can even go negative.  Remedy: record
    longer, or evaluate at larger ``omega`` (``tau_f`` grows towards lower
    ``omega``).
    """


class AngularConvergenceWarning(UserWarning):
    """The direction grid under-resolves the angular integral.

    Raised by :meth:`BKIntegrator.compute_spectrum` when doubling
    ``n_theta``/``n_phi`` (and the inner-panel node count) moves the
    angle-integrated ``dE/domega`` by more than ``rtol`` **of its own peak** at
    the frequencies carrying the most spectrum.  The quoted spectrum is then a
    quadrature artifact of the grid, not a converged angular integral.  Remedy:
    raise ``n_theta``/``n_phi``, or pass an explicit ``theta_max``/``n_inner``
    matched to the emission cone.

    Both the probe choice and the peak reference exist because a spectral
    density has nulls: between the harmonics of a structured orbit the value is
    a cancellation residual, where a pointwise relative change is meaningless
    (measured to exceed 100% on a closed orbit while the line centres move by
    ``1e-5``).
    """


def _max_phase_rate(beta, dirs):
    """``max_{sample, direction} |1 - n.v|`` -- the phase-rate shape factor.

    ``dPhi/dt2 = f (omega) [1 - n.v(t2)]``, so the fastest rate realised by
    the supplied directions and samples is this factor times ``f``.
    """
    nb = np.asarray(beta, dtype=np.float64) @ np.asarray(dirs, dtype=np.float64).T
    return float(np.max(np.abs(1.0 - nb)))


def _curvature_rate(time, beta):
    """Median ``|dv/dt|`` over the *active* part of the record (velocity-rotation rate).

    Equals ``chi/gamma^2`` for uniform circular motion.  ``0`` for a straight
    record, for which the formation-time diagnostic does not apply (there the
    kernel's vacuum subtraction already removes the straight-line part).

    A PIC record fed in from rest -- the drive has not reached the particle yet
    -- is *mostly motionless*, and a plain median over the whole record is then
    dominated by the numerical jitter of those samples rather than by the
    motion that radiates.  Measured on a ``gamma = 10``, ``chi = 0.5`` circle
    preceded by 65% of samples at rest (``|u| ~ 1e-31``): the median collapses
    to ``2e-31`` against the true ``chi/gamma^2 = 5e-3``, so ``tau_f`` (which
    goes as ``rate**(-2/3)``) comes out ~1e19 too long and ``L/tau_f``
    collapses to ~0 -- measured ``1e-18`` on the production LWFA record, whose
    dead fraction is the same 65%, instead of the ``O(10)``-``O(100)`` its
    length actually supports.  That is how ``RecordLengthWarning`` came to
    fire on records that are long enough.

    Samples slower than ``1e-3`` of the record's peak speed are therefore
    dropped, together with any interval touching one of them, so that the
    rest-to-moving transient does not count either -- the rate must come from
    intervals whose *both* ends move.  The threshold is relative, so the
    diagnostic stays scale invariant; on a record with no dead segment every
    sample is active and the result is bit-for-bit unchanged.
    """
    time = np.asarray(time, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if time.shape[0] < 3:
        return 0.0
    speed = np.linalg.norm(beta, axis=1)
    active = speed > 1e-3 * float(speed.max())
    moving = active[:-1] & active[1:]
    if not np.any(moving):
        return 0.0
    dv = np.linalg.norm(np.diff(beta, axis=0), axis=1) / np.diff(time)
    return float(np.median(dv[moving]))


def velocity_swing(beta):
    """Largest angle (radians) between ``beta(t)`` and ``beta[0]``.

    The spectrum is angle-integrated over a cone about a *fixed* axis (the
    initial velocity by default), but each part of the record radiates about
    the *instantaneous* velocity.  Emitted directions therefore reach out to
    ``velocity_swing + theta_c``, so a record that turns by more than
    ``theta_max`` is truncated no matter how fine the cone is.  A closed orbit
    swings by ``2 pi`` -- which is why every closed-orbit validation passes
    ``theta_max = pi`` explicitly.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(beta, axis=1)
    if norms[0] <= 0.0:
        return 0.0
    cosang = (beta @ beta[0]) / (norms * norms[0])
    return float(np.arccos(np.clip(np.min(cosang), -1.0, 1.0)))


def emission_half_angle(omega, epsilon, epsilon_prime, curvature_rate):
    """Stationary-phase emission half-angle ``theta_c`` per ``omega`` (radians).

    ``theta_c = (4 eps' Omega_eff / (omega eps))**(1/3)`` -- equivalently
    ``(4/m)**(1/3)`` with ``m = omega eps / (eps' Omega_eff)`` the
    recoil-shifted harmonic index.  It is the angle at which the cubic
    curvature term of the accumulated phase, ``Omega_eff**2 tau**3/6``,
    overtakes the angular term ``(1/gamma**2 + theta**2)``, i.e. where emission
    from one formation length starts to dephase.

    Measured on a ``gamma = 10``, ``chi = 0.5`` circle at *integer*
    recoil-shifted harmonics (BK kernel, converged 128-node rule), the
    half-energy emission angle is ``0.35 theta_c`` (``theta_50 gamma m**(1/3)
    = 4.6 ... 6.2`` over ``delta = 0.02 ... 0.7``).  ``theta_c`` therefore
    *narrows* towards high ``omega`` -- ``4.6/gamma`` at ``delta = 0.02`` down
    to ``0.95/gamma`` at ``delta = 0.7``; the soft-photon cone is the wide one.
    (The roadmap's P-4 originally claimed the opposite direction.)  Returns
    zeros where ``Omega_eff = 0``: a straight record radiates nothing.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if curvature_rate <= 0.0:
        return np.zeros(omega.shape)
    return (
        4.0 * np.asarray(epsilon_prime, dtype=np.float64) * curvature_rate
        / (omega * epsilon)
    ) ** (1.0 / 3.0)


def default_theta_max(time, beta, omega, epsilon, epsilon_prime=None,
                      safety=3.0, floor=None):
    """Adaptive cone half-angle covering the whole emission band.

    ``min(pi, velocity_swing + safety * max_omega theta_c(omega))``, floored at
    ``floor`` so the default is never tighter than the historical fixed
    ``5/gamma``.  The maximum over the ``omega`` grid is taken because one
    direction grid is shared by every frequency (the numba batch path sums all
    ``(omega, n)`` pairs in a single pass), so the widest cone in the grid has
    to cover the rest.

    Both terms are needed: ``theta_c`` grows towards low ``omega``, while the
    velocity swing grows with the record length -- and at low ``omega`` the
    record must be at least ``~1.5`` formation times long (see
    :func:`record_adequacy`), long enough on a curved orbit to swing far from
    the initial direction.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    eps_p = _final_energy(omega, epsilon, epsilon_prime)
    theta_c = emission_half_angle(omega, epsilon, eps_p,
                                  _curvature_rate(time, beta))
    value = velocity_swing(beta) + safety * float(np.max(theta_c))
    if floor is not None:
        value = max(value, float(floor))
    return min(np.pi, value)


def angular_edge_fraction(d2E, dom, n_phi, offset=0):
    """Fraction of the angular integral carried by the outermost ``theta`` node.

    Free diagnostic (no recompute).  :func:`cone_directions` stores each panel
    with ``theta`` **descending** (the Gauss--Legendre nodes come out ascending
    in ``cos(theta)``), so the node nearest ``theta_max`` is the *first*
    ``n_phi`` entries of the **last** panel -- not the last entries.  ``offset``
    is the index of that first entry: ``0`` for a single panel,
    ``n_inner * n_phi`` for a two-panel grid.  Build it from :func:`cone_bands`::

        bands = cone_bands(theta_max, n_theta, split, n_inner)
        offset = bands[0][1] * n_phi if len(bands) == 2 else 0

    (An earlier version read ``d2E[:, -n_phi:]``, i.e. the node nearest the
    *axis*, which reported a small value exactly when the cone was truncating.)

    A large value means the spectrum is fed by emission at the edge of the cone
    -- i.e. the cone is close to truncating and ``theta_max`` should be raised.
    """
    d2E = np.atleast_2d(np.asarray(d2E, dtype=np.float64))
    dom = np.asarray(dom, dtype=np.float64)
    band = slice(int(offset), int(offset) + int(n_phi))
    edge = np.sum(d2E[:, band] * dom[band], axis=1)
    total = d2E @ dom
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0.0, edge / total, 0.0)


#: Angular-convergence guard threshold: a change of the angle-integrated
#: ``dE/domega`` above this fraction of its peak, on grid refinement, is a
#: warning.  Peak-referenced rather than pointwise -- see
#: :class:`AngularConvergenceWarning`.
ANGULAR_RTOL = 0.02


def sampling_margin(time, beta, dirs, omega, factor):
    """Per-``omega`` trapezoid-aliasing margin ``dt * f * max|1 - n.v| / pi``.

    The double-time phase ``Phi = f [(t2-t1) - n.(r2-r1)]`` with
    ``f = omega eps/eps'`` oscillates at up to ``f * max|1 - n.v|``; the
    trapezoid sum resolves it only if that rate times the step is below the
    Nyquist limit ``pi`` (two samples per oscillation).  **A margin ``>= 1``
    means the double sum is aliased**: the returned value is then a sampling
    artifact, arbitrarily large and of either sign, not a spectrum.

    On a circle of period ``T`` the margin is ``4m/Nt`` with ``m`` the
    recoil-shifted harmonic index, i.e. it requires ``Nt > 4m`` -- twice the
    ``Nt > 2m`` quoted in :func:`validation.check_quantum_synchrotron`, which
    permits only one sample per oscillation and is measurably too lax (at
    ``m = 1990``, ``Nt = 2m`` still errs by 116%, ``Nt = 4m`` by 0.07%).

    Parameters
    ----------
    time : (Nt,) array; the largest step of the (possibly non-uniform) grid is used.
    beta : (Nt, 3) velocities.
    dirs : (Ndir, 3) unit vectors.
    omega : (N_omega,) array.
    factor : (N_omega,) phase factor ``f`` (``omega eps/eps'`` or ``omega``).

    Returns
    -------
    (N_omega,) array of margins; values ``>= 1`` are aliased.
    """
    time = np.asarray(time, dtype=np.float64)
    dt = float(np.max(np.diff(time))) if time.shape[0] > 1 else 0.0
    shape = _max_phase_rate(beta, dirs)
    return dt * np.asarray(factor, dtype=np.float64) * shape / np.pi


def formation_time(time, beta, omega, epsilon, epsilon_prime=None):
    """Curvature-limited formation (coherence) time ``tau_f`` per ``omega``.

    ``tau_f = (6 eps' / (omega Omega_eff^2))**(1/3)`` -- the time for the
    curvature term of the accumulated phase, ``(omega/2eps') Omega_eff^2
    tau^3/3``, to reach unity (``Omega_eff`` = median ``|dv/dt|`` over the
    moving samples, equal to ``chi/gamma^2`` on a circle).  Returns ``inf`` for
    a straight record or for
    ``omega <= 0`` (where ``tau_f -> inf``), cases in which the diagnostic does
    not apply -- :func:`record_adequacy` maps ``inf`` to "not applicable" rather
    than to "maximally inadequate".

    This is the coherence scale of the double-time integral: pairs with
    ``|t2 - t1| >> tau_f`` oscillate and cancel, so the spectrum is carried by
    the band ``|t2 - t1| <~ tau_f``.  It grows towards low ``omega``
    (``~ omega^{-1/3}``), which is why the soft-photon region is the first to
    suffer when the record is short.
    """
    time = np.asarray(time, dtype=np.float64)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    eps_p = epsilon - omega if epsilon_prime is None else np.asarray(epsilon_prime, dtype=np.float64)
    rate = _curvature_rate(time, beta)
    if rate <= 0.0:
        return np.full(omega.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_f = (6.0 * eps_p / (omega * rate ** 2)) ** (1.0 / 3.0)
    return np.where(omega > 0.0, tau_f, np.inf)


def record_adequacy(time, beta, omega, epsilon, epsilon_prime=None):
    """Record length in formation times, ``L / tau_f(omega)``, per ``omega``.

    The double integral is bulk-dominated only for ``L >> tau_f``; at
    ``L <~ tau_f`` the hard record endpoints dominate and the spectrum is
    under-reported -- it can even go negative.  Measured on arcs of a
    ``gamma = 10``, ``chi = 0.5`` circle at the first harmonic, against
    ``L * dP/domega``:

    ==========  =========  =======
    ``L/tau_f``  deficit     note
    ==========  =========  =======
    1.60         0%         closed orbit (periodic -- exact; the guard's
                            ``1.5`` threshold deliberately leaves it alone)
    1.44        11%
    1.28        28%
    1.12        52%
    0.96        79%
    0.80        collapse   sign flips ($-0.03$)
    0.64        collapse   ($-0.19$)
    ==========  =========  =======

    So treat ``L/tau_f <~ 3`` as "quoted spectrum is a lower bound" and
    ``<~ 1.5`` as "not trustworthy".  :func:`formation_time` documents
    ``tau_f``; a straight record returns ``inf`` (the diagnostic does not
    apply -- there the kernel's vacuum subtraction removes the straight-line
    part and no truncation deficit arises).
    """
    time = np.asarray(time, dtype=np.float64)
    span = float(time[-1] - time[0]) if time.shape[0] > 1 else 0.0
    tau_f = np.atleast_1d(formation_time(time, beta, omega, epsilon, epsilon_prime))
    with np.errstate(divide="ignore", invalid="ignore"):
        # a straight record has tau_f = inf -> the check does not apply (inf,
        # not 0: 0 would read as "maximally inadequate")
        return np.where(np.isinf(tau_f), np.inf, span / tau_f)


def _run_adequacy_checks(time, beta, dirs, omega, factor, epsilon, epsilon_prime,
                         phase, energy, checks):
    """Emit :class:`SamplingWarning` / :class:`RecordLengthWarning` as requested.

    ``checks`` is ``"warn"`` (default), ``"raise"`` or ``"ignore"``.  The two
    checks are independent: aliasing is a property of the time grid, the
    record-length deficit of the span versus the formation time.
    """
    if checks == "ignore":
        return
    if checks not in ("warn", "raise"):
        raise ValueError("checks must be 'warn', 'raise' or 'ignore'")

    def emit(message, category):
        if checks == "raise":
            raise category(message)
        warnings.warn(message, category)

    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    margin = sampling_margin(time, beta, dirs, omega, factor)
    if np.any(margin >= 1.0):
        dt = float(np.max(np.diff(np.asarray(time, dtype=np.float64))))
        k = int(np.argmax(margin))
        emit(
            "double-time phase is under-resolved (trapezoid aliasing): "
            f"{int(np.sum(margin >= 1.0))}/{omega.size} omega values have "
            f"dt*max|dPhi/dt|/pi >= 1 (max margin {margin.max():.3g} at "
            f"omega = {omega[k]:.6g}, dt = {dt:.6g}). Those values are sampling "
            "artifacts -- arbitrarily large and of either sign -- not a spectrum. "
            f"Refine the time grid by ~{np.ceil(2.0 * margin.max()):.0f}x "
            "(Nyquist needs two samples per phase oscillation) or use smaller omega.",
            SamplingWarning,
        )

    if energy is not None:
        eps_ref = float(np.mean(np.asarray(energy, dtype=np.float64)))
        eps_prime_ref = (eps_ref - omega) if phase == "recoil" else eps_ref
    else:
        eps_ref = float(epsilon)
        eps_prime_ref = (eps_ref - omega) if epsilon_prime is None else epsilon_prime
    adequacy = record_adequacy(time, beta, omega, eps_ref, eps_prime_ref)
    if np.any(adequacy < 1.5):
        k = int(np.argmin(adequacy))
        emit(
            "record is short compared with the formation time: "
            f"L/tau_f < 1.5 for {int(np.sum(adequacy < 1.5))}/{omega.size} omega "
            f"values (min L/tau_f = {adequacy.min():.3g} at omega = {omega[k]:.6g}). "
            "The double integral is then dominated by the hard record endpoints "
            "and under-reports the spectrum (it can go negative). Record longer, "
            "or evaluate at larger omega.",
            RecordLengthWarning,
        )


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

    ``checks`` selects the adequacy guards: ``"warn"`` (default) emits
    :class:`SamplingWarning` when the time grid under-resolves the phase
    (trapezoid aliasing) and :class:`RecordLengthWarning` when the record is
    short compared with the formation time; ``"raise"`` turns either into an
    exception and ``"ignore"`` disables both.
    """

    def __init__(self, trajectory: Trajectory, params: Parameters,
                 backend: str = "auto", checks: str = "warn") -> None:
        self.trajectory = trajectory
        self.params = params
        self.backend = _resolve_backend(backend)
        self.checks = checks
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
            energy=self.energy, checks=self.checks,
        )
        return PREFACTOR * omega_grid[:, None] ** 2 * self.params.charge ** 2 * I

    def d2_probability_batch(self, omega_grid, dirs):
        """``d2W/(dw dO)`` on the full ``(N_omega, N_dir)`` grid."""
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
        if not np.all(np.isfinite(omega_grid)) or np.any(omega_grid <= 0.0):
            raise ValueError(
                "omega_grid must be finite and strictly positive: dW/domega = "
                "dE/domega / omega is undefined at omega = 0 (dE/domega carries "
                "omega^2, so the quotient is 0/0 there)"
            )
        return self.d2_energy_batch(omega_grid, dirs) / omega_grid[:, None]

    # -- single (omega, n) -------------------------------------------------
    def d2_energy(self, omega, n):
        return float(self.d2_energy_batch([_as_scalar_omega(omega)], [n])[0, 0])

    def d2_probability(self, omega, n):
        omega = _as_scalar_omega(omega)
        if omega <= 0.0:
            raise ValueError(
                "omega must be strictly positive: dW/domega = dE/domega / omega "
                f"is undefined at omega = 0 (got {omega!r}); use d2_energy for "
                "omega = 0"
            )
        return self.d2_energy(omega, n) / omega

    # -- adequacy diagnostics ----------------------------------------------
    def adequacy(self, omega_grid, dirs):
        """Per-``omega`` ``(sampling_margin, L/tau_f)`` for these directions.

        ``sampling_margin >= 1``: the time grid under-resolves the double-time
        phase and the double sum is a trapezoid-aliasing artifact (arbitrarily
        large, either sign) rather than a spectrum.  ``L/tau_f <~ 1``: the
        record is shorter than the formation time, so the integral is
        endpoint-dominated and under-reports (it can go negative).  See
        :func:`sampling_margin` and :func:`record_adequacy` for the definitions
        and for the measured size of each failure.
        """
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
        dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        phase, eps_p = _recoil_args(self.params)
        if self.energy is not None:
            _, _, f = _local_vertex_arrays(self.params.kernel, omega_grid,
                                           self.energy, phase, self.params.mass)
            # omega * eps/eps': the phase rate, matching the fixed-energy branch
            factor = omega_grid * f.max(axis=1)
            eps_ref = float(np.mean(self.energy))
            eps_prime_ref = (eps_ref - omega_grid) if phase == "recoil" else eps_ref
        else:
            eps_prime_ref = _final_energy(omega_grid, self.params.epsilon, eps_p)
            factor = _phase_factor(phase, omega_grid, self.params.epsilon, eps_prime_ref)
            eps_ref = self.params.epsilon
        return (sampling_margin(self.time, self.beta, dirs, omega_grid, factor),
                record_adequacy(self.time, self.beta, omega_grid, eps_ref, eps_prime_ref))

    # -- spectrum ----------------------------------------------------------
    def _angular_convergence(self, omega_grid, dE, axis, theta_max, n_theta,
                             n_phi, split, n_inner, n_probe=3):
        """Refine the direction grid at a few ``omega`` and compare.

        Returns ``(max_change, probe_omega)``.  The probes are the ``n_probe``
        frequencies carrying the most spectrum (largest ``|dE/domega|``) --
        that is where a quadrature error would corrupt the answer -- and the
        change is measured against the **peak** of the probed spectrum, not
        point by point.  ``n_theta``, ``n_phi`` and the inner-panel count are all
        doubled, so the cost is about ``4 n_probe / N_omega`` of the base
        angular integral.

        Both choices exist for the same reason: a spectral density has nulls.
        Between the harmonics of a closed orbit the value is a cancellation
        residual which is *measured* to swing by more than its own size under
        grid refinement, while the line centres are stable to ``1e-5``.  A
        guard that probed those nulls pointwise would fire on every structured
        spectrum while saying nothing about the lines.  Weighted probes with a
        peak reference instead bound the error on the spectrum as a whole, which
        is what the quadrature actually controls.
        """
        n = omega_grid.size
        if n == 0:
            return 0.0, np.array([])
        order = np.argsort(np.abs(dE))
        probe = np.sort(order[::-1][:max(2, min(n_probe, n))])
        dirs, dom = cone_directions(
            axis, theta_max, 2 * n_theta, 2 * n_phi, split=split,
            n_inner=None if n_inner is None else 2 * n_inner,
        )
        refined = self.d2_energy_batch(omega_grid[probe], dirs) @ dom
        base = dE[probe]
        scale = float(np.max(np.abs(base)))
        if scale <= 0.0:
            return 0.0, omega_grid[probe]
        return float(np.max(np.abs(refined - base)) / scale), omega_grid[probe]

    def compute_spectrum(self, omega_grid, theta_max=None, n_theta=16, n_phi=8,
                         axis=None, split=None, n_inner=None) -> Spectrum:
        """Angle-integrate ``dW/dw`` and ``dE/dw`` over a cone of directions.

        The cone axis defaults to the initial velocity direction (beam axis).

        ``theta_max=None`` (the default) adapts the cone to the record through
        :func:`default_theta_max`, ``velocity_swing + 3 max_omega theta_c``
        capped at ``pi`` and floored at the historical fixed ``5 / gamma``.  The
        fixed cone was wrong at both ends: ``theta_c`` reaches ``4.6 / gamma`` at
        ``delta = 0.02`` (so the soft-photon end, the region this module exists
        to get right, was truncated by ~7%) while at high ``delta`` the cone is
        actually narrower than ``5 / gamma``; and it ignored the velocity swing
        over the record entirely, which for a closed orbit is ``2 pi`` -- for
        which the fixed cone captures about 1%.  An explicit ``theta_max`` is
        honoured (clamped to ``pi``).

        ``split``/``n_inner`` select the two-panel direction rule of
        :func:`cone_directions`.  When ``theta_max`` is auto they default to
        ``min(8/gamma, theta_max/2)`` and ``n_theta``; an explicit ``theta_max``
        keeps the historical single-panel rule unless ``split`` is passed, so
        existing callers are numerically unchanged.

        The angular diagnostics go into ``metadata``
        (``angular_edge_fraction`` -- free; ``angular_convergence`` and
        ``angular_probe_omega`` -- from the refinement check) and the refinement
        check raises :class:`AngularConvergenceWarning` under
        ``checks="warn"`` (the default), with ``"raise"``/``"ignore"`` available.
        """
        omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
        if not np.all(np.isfinite(omega_grid)) or np.any(omega_grid <= 0.0):
            raise ValueError(
                "omega_grid must be finite and strictly positive: dW/domega = "
                "dE/domega / omega is undefined at omega = 0 (dE/domega carries "
                "omega^2, so the quotient is 0/0 there)"
            )
        if axis is None:
            beta0 = self.beta[0]
            if np.linalg.norm(beta0) < 1e-12:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = beta0 / np.linalg.norm(beta0)
        gamma0 = float(self.trajectory.gamma()[0])
        floor = 5.0 / max(gamma0, 1e-6)

        phase, eps_p = _recoil_args(self.params)
        if self.energy is not None:
            eps_ref = float(np.mean(self.energy))
            eps_prime_ref = ((eps_ref - omega_grid) if phase == "recoil"
                             else np.full_like(omega_grid, eps_ref))
        else:
            eps_ref = float(self.params.epsilon)
            eps_prime_ref = _final_energy(omega_grid, eps_ref, eps_p)

        auto = theta_max is None
        swing = velocity_swing(self.beta)
        if auto:
            theta_max = default_theta_max(self.time, self.beta, omega_grid,
                                          eps_ref, eps_prime_ref, safety=3.0,
                                          floor=floor)
        theta_max = float(min(np.pi, max(float(theta_max), 0.0)))

        if split is None and auto:
            split = min(8.0 / max(gamma0, 1e-6), 0.5 * theta_max)
        if split is not None and n_inner is None:
            n_inner = int(n_theta)

        # Effective panels: cone_bands raises on a split that cannot be honoured
        # instead of silently returning a single panel (the metadata used to
        # record the requested split even when it was dropped).
        bands = cone_bands(theta_max, n_theta, split, n_inner)
        two_panel = len(bands) == 2
        edge_offset = bands[0][1] * n_phi if two_panel else 0

        dirs, dom = cone_directions(axis, theta_max, n_theta, n_phi,
                                    split=split, n_inner=n_inner)

        d2E = self.d2_energy_batch(omega_grid, dirs)   # (N_omega, N_dir)
        dE = d2E @ dom
        dW = dE / omega_grid

        sampling, adequacy = self.adequacy(omega_grid, dirs)
        edge = angular_edge_fraction(d2E, dom, n_phi, offset=edge_offset)
        if self.checks == "ignore":
            convergence, probe_omega = float("nan"), np.array([])
        else:
            convergence, probe_omega = self._angular_convergence(
                omega_grid, dE, axis, theta_max, n_theta, n_phi, split, n_inner)
            if convergence > ANGULAR_RTOL:
                message = (
                    "direction grid under-resolves the angular integral: "
                    "doubling n_theta/n_phi moves the angle-integrated "
                    f"spectrum by up to {convergence * 100:.2f}% of its peak "
                    f"(> {ANGULAR_RTOL * 100:.0f}%) at the probed "
                    f"omega = {probe_omega.tolist()}. The quoted spectrum is a "
                    "quadrature artifact of the grid. Raise n_theta/n_phi, or "
                    "pass an explicit theta_max/n_inner matched to the emission "
                    "cone (see integrator.emission_half_angle)."
                    + ("" if swing < 0.25 * theta_max else
                       f" The record turns by {swing:.3f} rad "
                       f"({swing * gamma0:.2f}/gamma) against a cone of "
                       f"{theta_max:.3f} rad, so the emission is NOT confined "
                       "to a cone about axis=beta[0]: pass axis= (a symmetry "
                       "or beam axis) or a converged n_theta over theta_max=pi.")
                )
                if self.checks == "raise":
                    raise AngularConvergenceWarning(message)
                warnings.warn(message, AngularConvergenceWarning)

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
            theta_max_auto=bool(auto),
            # effective grid actually used by cone_directions (bands), not the
            # requested split/n_inner
            theta_split=float(bands[0][0]) if two_panel else None,
            n_inner=int(bands[0][1]) if two_panel else None,
            n_theta=int(n_theta),
            n_phi=int(n_phi),
            backend=self.backend,
            units="natural (c=hbar=1, m_e=1)",
            # adequacy guards: sampling >= 1 -> aliased, record_adequacy <~ 1 ->
            # endpoint-dominated.  See integrator.sampling_margin /
            # record_adequacy for definitions and measured failure sizes.
            sampling_margin=sampling,
            record_adequacy=adequacy,
            # angular guards: edge -> fraction carried by the outermost node
            # (free); convergence -> largest change under grid refinement,
            # measured against the peak over angular_probe_omega (nan when
            # checks="ignore").  The cone is only meaningful while the velocity
            # swing stays well inside theta_max, so both the swing and the
            # emission half-angle go in.
            velocity_swing=float(swing),
            emission_half_angle=emission_half_angle(omega_grid, eps_ref,
                                                    eps_prime_ref,
                                                    _curvature_rate(self.time, self.beta)),
            angular_edge_fraction=edge,
            angular_convergence=convergence,
            angular_probe_omega=probe_omega,
        )
        return Spectrum(omega=omega_grid, dW_domega=dW, dE_domega=dE, metadata=meta)


def compute_spectrum(trajectory: Trajectory, params: Parameters, omega_grid,
                     theta_max=None, n_theta=16, n_phi=8, axis=None,
                     split=None, n_inner=None, backend="auto",
                     checks="warn") -> Spectrum:
    """Convenience function: build an integrator and return the spectrum."""
    return BKIntegrator(trajectory, params, backend=backend, checks=checks).compute_spectrum(
        omega_grid, theta_max=theta_max, n_theta=n_theta, n_phi=n_phi, axis=axis,
        split=split, n_inner=n_inner,
    )
