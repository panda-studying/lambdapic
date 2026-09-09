"""Regression tests for the decoupled Baier--Katkov reference module.

These pin the conventions that the coarse validation script
(``lambdapic.core.qed.baier_katkov.validation``) checks by eye:

* the numba backend reproduces the numpy reference double sum;
* the imaginary part of the full double integral is round-off;
* the absolute normalization (``alpha/(4 pi^2)`` prefactor) reproduces the
  Schwinger synchrotron spectrum at a high harmonic of a closed circle;
* the analytic quantum synchrotron reference is anchored three ways
  (Macdonald == Airy == the project's LCFA table, classical limit, Larmor);
* the quantum regime: the one-turn BK spectrum at the recoil-shifted
  harmonics reproduces the exact constant-field quantum synchrotron spectrum
  (recoil phase, ``(eps^2 + eps'^2)/2eps'^2`` prefactor, ``omega^2/gamma^2``
  term) to ``O(1/gamma^2)`` for both kernel forms;
* ``Parameters`` options are honoured and invalid input is rejected.
"""

import numpy as np
import pytest

from lambdapic.core.qed.baier_katkov import integrator as bki
from lambdapic.core.qed.baier_katkov import kernel as bkk
from lambdapic.core.qed.baier_katkov import reference as bkr
from lambdapic.core.qed.baier_katkov import validation as bkv
from lambdapic.core.qed.baier_katkov.types import Parameters, Trajectory
from lambdapic.core.qed.baier_katkov.units import fine_structure, natural_time_unit

HAS_NUMBA = "numba" in bki.available_backends()


def circle(gamma, rho, n_samples, turns=1.0):
    """Closed uniform circular motion in the x-y plane."""
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    Omega = beta / rho
    t = np.linspace(0.0, 2.0 * np.pi * turns / Omega, n_samples)
    phi = Omega * t
    r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])
    u = gamma * beta * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(t)])
    return Trajectory(time=t, position=r, momentum=u), Omega


def energy_varying_circle(gamma0, n_samples, delta, turns=1.0):
    """Exactly-closed planar loop with ``gamma(t) = gamma0 (1 + delta sin 2 pi t)``.

    ``rho`` is chosen from ``2 pi rho = int |beta| dt`` so the loop closes on
    itself (no finite-record boundary term), while ``gamma`` sweeps over an
    ``O(1)`` range -- the regime where the fixed incident energy of P-2 is
    wrong.  The momentum ``u = gamma beta`` is built on shell, so the local
    energy ``eps = mass * gamma`` is exactly consistent with ``beta``.
    """
    T0 = 1.0
    t = np.linspace(0.0, turns * T0, n_samples)
    gam = gamma0 * (1.0 + delta * np.sin(2.0 * np.pi * t / T0))
    beta_mag = np.sqrt(1.0 - 1.0 / gam ** 2)
    rho = float(np.trapezoid(beta_mag, t) / (2.0 * np.pi * turns))
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (beta_mag[:-1] + beta_mag[1:]) * np.diff(t))]) / rho
    r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])
    u = (gam * beta_mag)[:, None] * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(t)])
    return Trajectory(time=t, position=r, momentum=u, energy=gam), rho


@pytest.fixture(scope="module")
def small_circle():
    traj, Omega = circle(gamma=10.0, rho=200.0, n_samples=200)
    dirs = np.array([[0.0, 0.0, 1.0],
                     [0.0, 0.5, np.sqrt(0.75)],
                     [np.sin(1.2), 0.0, np.cos(1.2)]])
    omegas = np.array([0.5 * Omega, 3.0 * Omega, 2.0, 6.0])
    return traj, Omega, dirs, omegas


@pytest.mark.skipif(not HAS_NUMBA, reason="numba backend not available")
@pytest.mark.parametrize("kernel", ["dot", "trace", "velocity"])
@pytest.mark.parametrize("phase,eps_prime", [("recoil", None), ("classical", 10.0)])
def test_numba_matches_numpy(small_circle, kernel, phase, eps_prime):
    traj, _, dirs, omegas = small_circle
    args = (traj.time, traj.position, traj.beta(), omegas, dirs, 10.0)
    kw = dict(kernel=kernel, phase=phase, epsilon_prime=eps_prime)
    ref = bki.double_time_integral_batch(*args, backend="numpy", **kw)
    fast = bki.double_time_integral_batch(*args, backend="numba", **kw)
    # a closed circle has exact zeros at some (harmonic, direction) pairs, so
    # compare against the scale of the grid rather than point-wise
    assert np.max(np.abs(fast - ref)) < 1e-12 * np.max(np.abs(ref))


def test_full_square_is_real(small_circle):
    traj, _, dirs, omegas = small_circle
    I = bki.double_time_integral(traj.time, traj.position, traj.beta(), omegas[2],
                                 dirs[1], 10.0, backend="numpy")
    assert abs(I.imag) < 1e-10 * abs(I.real)


def test_velocity_kernel_equals_amplitude_squared(small_circle):
    traj, _, dirs, _ = small_circle
    beta = traj.beta()
    omega, n = 1.0, dirs[1]
    amp = bkk.classical_amplitude(traj.time, traj.position, beta, omega, n)
    via_sq = np.sum(np.abs(amp) ** 2)
    via_double = bki.double_time_integral(traj.time, traj.position, beta, omega, n,
                                          epsilon=10.0, kernel="velocity",
                                          phase="classical", epsilon_prime=10.0).real
    assert abs(via_double - via_sq) < 1e-8 * via_sq


def test_absolute_normalization_schwinger():
    """Angle-integrated dE/dw at a high harmonic equals T * dP/dw (Schwinger).

    dP/dw = (sqrt3 / 2 pi) (alpha gamma beta / rho) F(w / w_c) in Gaussian
    natural units (e^2 = alpha); this is the normalization of the 88.5 keV
    per-turn synchrotron loss and of the LCFA tables in core/qed.
    """
    from scipy.integrate import quad
    from scipy.special import kv, roots_legendre

    gamma, rho = 30.0, 1.0e4
    traj, Omega = circle(gamma=gamma, rho=rho, n_samples=2000)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    T = 2.0 * np.pi / Omega
    w_c = 1.5 * gamma ** 3 * Omega
    m = 64
    omega = m * Omega

    x, wx = roots_legendre(32)
    theta = np.arccos(x)
    dirs = np.column_stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    d2E = bki.BKIntegrator(traj, Parameters(epsilon=gamma, recoil="classical")
                           ).d2_energy_batch([omega], dirs)[0]
    code = 2.0 * np.pi * np.sum(wx * d2E)          # azimuthal symmetry

    F = (omega / w_c) * quad(lambda s: kv(5.0 / 3.0, s), omega / w_c, np.inf)[0]
    exact = T * (np.sqrt(3.0) / (2.0 * np.pi)) * fine_structure * gamma * beta / rho * F
    assert abs(code / exact - 1.0) < 0.03


def test_prefactor_constant():
    assert np.isclose(bki.PREFACTOR, fine_structure / (4.0 * np.pi ** 2))


def test_trace_kernel_equals_dot_kernel_on_shell():
    """The two BK kernel forms are the same function for on-shell velocities."""
    rng = np.random.default_rng(3)
    for eps, omega in ((10.0, 3.0), (10.0, 7.0), (50.0, 1.0)):
        beta = np.sqrt(1.0 - 1.0 / eps ** 2)
        u1 = rng.normal(size=(8, 3))
        u2 = rng.normal(size=(8, 3))
        b1 = beta * u1 / np.linalg.norm(u1, axis=1, keepdims=True)
        b2 = beta * u2 / np.linalg.norm(u2, axis=1, keepdims=True)
        n_dot = bkk.dot_kernel(b1, b2, eps, omega)
        n_trace = bkk.trace_kernel(b1, b2, eps, omega)
        assert np.allclose(n_dot, n_trace, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# analytic reference (constant-field quantum synchrotron spectrum)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("chi", [0.05, 0.5, 2.0])
def test_reference_macdonald_airy_and_lcfa_table_agree(chi):
    """K_{1/3},K_{2/3} form == Airy form == the project's LCFA table integrand.

    The LCFA table rate (``optical_depth_tables.gen_photon_prob_rate_for_delta``)
    is per unit *proper* time in SI; ``* natural_time_unit / gamma`` converts
    it to the module's per-lab-time natural-unit rate.
    """
    from lambdapic.core.qed.optical_depth_tables import gen_photon_prob_rate_for_delta

    gamma = 10.0
    table = gen_photon_prob_rate_for_delta(chi)
    deltas = np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    mac = bkr.quantum_synchrotron_rate(deltas, chi, gamma, form="macdonald")
    airy_form = bkr.quantum_synchrotron_rate(deltas, chi, gamma, form="airy")
    lcfa = np.array([table(d) for d in deltas]) * natural_time_unit / gamma
    scale = mac.max()
    assert np.all(np.abs(airy_form - mac) < 1e-5 * scale)
    assert np.all(np.abs(lcfa - mac) < 1e-5 * scale)


def test_reference_classical_limit_and_power():
    """chi -> 0 reproduces the classical (Schwinger) spectrum; powers close."""
    gamma = 10.0
    for chi in (1e-3, 1e-4):
        d = 0.3 * chi                                  # fixed y0 = 0.2
        q = bkr.quantum_synchrotron_rate(d, chi, gamma)
        c = bkr.classical_synchrotron_rate(d, chi, gamma)
        assert abs(q / c - 1.0) < 3.0 * chi
    # dP/domega = delta * rate is Schwinger's (sqrt3/2pi)(alpha gamma beta^2/rho) F(y0)
    rho = 200.0
    chi = bkr.circle_chi(gamma, rho)
    for d in (0.01, 0.1, 0.5):
        y0 = 2.0 * d / (3.0 * chi)
        lhs = d * bkr.classical_synchrotron_rate(d, chi, gamma)
        rhs = (np.sqrt(3.0) / (2.0 * np.pi)) * fine_structure * gamma * (1 - 1 / gamma ** 2) / rho \
            * bkr.synchrotron_F(y0)
        assert abs(lhs / rhs - 1.0) < 1e-8
    # classical power is Larmor, quantum power is Larmor * g(chi) with g < 1
    for chi in (0.1, 1.0):
        larmor = (2.0 / 3.0) * fine_structure * chi ** 2
        assert abs(bkr.synchrotron_power(chi, gamma, quantum=False) / larmor - 1.0) < 1e-6
        g = bkr.synchrotron_power(chi, gamma, quantum=True) / larmor
        assert 0.0 < g < 1.0
        assert abs(g / bkr.erber_g(chi) - 1.0) < 0.03


def test_direction_grid_covers_sphere():
    dirs, w = bkr.orbit_direction_grid(10.0)
    assert np.isclose(w.sum(), 4.0 * np.pi)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


# --------------------------------------------------------------------------
# quantum synchrotron cross-check (validation V8)
# --------------------------------------------------------------------------
def _assert_quantum_agreement(res, tol):
    for kernel, ratio in res["ratio"].items():
        assert np.all(np.abs(ratio - 1.0) < tol), (kernel, ratio)


def test_quantum_synchrotron_quick():
    """BK at recoil-shifted harmonics == exact quantum spectrum (chi=0.5, gamma=10).

    Two hard photons (delta = 0.3, 0.5) where recoil suppresses the classical
    spectrum by 30-55%; both kernel forms must agree with the exact result to
    1% (the residual is the O(1/gamma^2) error of the reference).
    """
    res = bkv.check_quantum_synchrotron(gamma=10.0, chi=0.5, deltas=(0.3, 0.5),
                                        kernels=("dot", "trace"))
    assert np.all(res["quantum_over_classical"] < 0.75)   # a genuinely quantum regime
    _assert_quantum_agreement(res, tol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize("chi,deltas,tol", [
    (0.5, (0.05, 0.1, 0.2, 0.3, 0.5), 1e-2),
    (2.0, (0.1, 0.2, 0.3, 0.5, 0.7), 2e-2),
])
def test_quantum_synchrotron_spectrum(chi, deltas, tol):
    """Full V8 sweep: both kernels, delta from the soft tail to delta = 0.7."""
    res = bkv.check_quantum_synchrotron(gamma=10.0, chi=chi, deltas=deltas,
                                        kernels=("dot", "trace"))
    _assert_quantum_agreement(res, tol=tol)


def test_bare_harmonic_evaluation_is_wrong():
    """Documenting the trap: BK evaluated at the bare m*Omega is off by O(m omega/eps)."""
    res = bkv.check_quantum_synchrotron(gamma=10.0, chi=0.5, deltas=(0.1, 0.2),
                                        kernels=("dot",), include_bare=True)
    assert np.all(np.abs(res["ratio"]["dot"] - 1.0) < 1e-2)
    assert np.any(np.abs(res["bare_ratio"] - 1.0) > 0.2)


def test_recoil_option_is_honoured(small_circle):
    traj, Omega, dirs, _ = small_circle
    omega, n = 0.5 * Omega, dirs[0]
    classical = bki.BKIntegrator(traj, Parameters(epsilon=10.0, recoil="classical"))
    bk = bki.BKIntegrator(traj, Parameters(epsilon=10.0))
    explicit = bki.d2_energy(traj.time, traj.position, traj.beta(), omega, n,
                             epsilon=10.0, phase="classical", epsilon_prime=10.0)
    assert np.isclose(classical.d2_energy(omega, n), explicit, rtol=1e-12)
    assert not np.isclose(classical.d2_energy(omega, n), bk.d2_energy(omega, n), rtol=1e-6)


def test_spectrum_consistency(small_circle):
    traj, Omega, _, _ = small_circle
    grid = np.array([m * Omega for m in range(1, 5)])
    spec = bki.compute_spectrum(traj, Parameters(epsilon=10.0), grid, theta_max=np.pi,
                                n_theta=8, n_phi=1, axis=[0.0, 0.0, 1.0])
    assert np.allclose(spec.dE_domega, spec.omega * spec.dW_domega)
    assert np.all(spec.dW_domega > 0.0)          # closed orbit, no boundary terms
    assert spec.metadata["recoil"] == "baier_katkov"


def test_invalid_input_rejected(small_circle):
    traj, _, dirs, _ = small_circle
    args = (traj.time, traj.position, traj.beta())
    with pytest.raises(ValueError):
        bki.double_time_integral(*args, 1.0, dirs[0], 10.0, kernel="foo")
    with pytest.raises(ValueError):
        bki.double_time_integral(*args, 1.0, dirs[0], 10.0, phase="foo")
    with pytest.raises(ValueError):
        bki.d2_energy(*args, 10.0, dirs[0], epsilon=10.0)      # omega == epsilon
    with pytest.raises(ValueError):
        bki.d2_energy(*args, 15.0, dirs[0], epsilon=10.0)      # omega > epsilon
    with pytest.raises(ValueError):
        Trajectory(time=np.array([0.0]), position=np.zeros((1, 3)))
    with pytest.raises(ValueError):
        Parameters(epsilon=10.0, recoil="foo")
    with pytest.raises(NotImplementedError):
        Parameters(epsilon=10.0, spin_averaged=False)
    with pytest.raises(ValueError):
        bki.double_time_integral(*args, 1.0, dirs[0], 10.0, backend="foo")


# --------------------------------------------------------------------------
# P-2: local-energy (sec. 4.2) generalization -- eps(t1), eps(t2)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def varying_circle():
    """Exactly-closed, on-shell trajectory with gamma(t) sweeping 7 -> 13."""
    traj, rho = energy_varying_circle(gamma0=10.0, n_samples=256, delta=0.3)
    beta0 = np.sqrt(1.0 - 1.0 / 10.0 ** 2)
    Omega = beta0 / rho
    dirs = np.array([[0.0, 0.0, 1.0],
                     [0.0, 0.5, np.sqrt(0.75)],
                     [np.sin(1.2), 0.0, np.cos(1.2)]])
    omegas = np.array([0.5 * Omega, 3.0 * Omega, 2.0, 6.0])
    omegas = omegas[omegas < 0.9 * traj.energy.min()]
    return traj, Omega, dirs, omegas


@pytest.mark.parametrize("kernel", ["dot", "trace"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_local_energy_degenerates_to_fixed(small_circle, kernel, backend):
    """A constant energy history reproduces the fixed-epsilon path exactly."""
    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("numba backend not available")
    traj, _, dirs, omegas = small_circle
    beta = traj.beta()
    fixed = bki.double_time_integral_batch(
        traj.time, traj.position, beta, omegas, dirs, 10.0,
        kernel=kernel, backend=backend,
    )
    local = bki.double_time_integral_batch(
        traj.time, traj.position, beta, omegas, dirs, 10.0,
        kernel=kernel, backend=backend, energy=np.full(traj.n_samples, 10.0),
    )
    assert np.max(np.abs(local - fixed)) < 1e-10 * np.max(np.abs(fixed))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_local_energy_kernels_coincide_on_shell(varying_circle, backend):
    """With per-vertex recoil eps'_i = eps_i - omega the on-shell identity
    survives vertex by vertex, so the local dot and trace kernels coincide."""
    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("numba backend not available")
    traj, _, dirs, omegas = varying_circle
    beta = traj.beta()
    dot = bki.double_time_integral_batch(
        traj.time, traj.position, beta, omegas, dirs, 10.0,
        kernel="dot", backend=backend, energy=traj.energy,
    )
    trace = bki.double_time_integral_batch(
        traj.time, traj.position, beta, omegas, dirs, 10.0,
        kernel="trace", backend=backend, energy=traj.energy,
    )
    assert np.max(np.abs(trace - dot)) < 1e-10 * np.max(np.abs(dot))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba backend not available")
@pytest.mark.parametrize("kernel", ["dot", "trace"])
def test_local_energy_matches_numpy(varying_circle, kernel):
    traj, _, dirs, omegas = varying_circle
    beta = traj.beta()
    args = (traj.time, traj.position, beta, omegas, dirs, 10.0)
    ref = bki.double_time_integral_batch(*args, kernel=kernel, backend="numpy",
                                         energy=traj.energy)
    fast = bki.double_time_integral_batch(*args, kernel=kernel, backend="numba",
                                          energy=traj.energy)
    assert np.max(np.abs(fast - ref)) < 1e-11 * np.max(np.abs(ref))


def test_local_energy_is_hermitian(varying_circle):
    """Full-square imaginary part is round-off: Phi antisymmetric, N symmetric."""
    traj, _, dirs, omegas = varying_circle
    I = bki.double_time_integral(traj.time, traj.position, traj.beta(), omegas[1],
                                 dirs[1], 10.0, kernel="trace", backend="numpy",
                                 energy=traj.energy)
    assert abs(I.imag) < 1e-10 * abs(I.real)


def test_local_energy_spectrum_positive(varying_circle):
    """dW/dw stays positive at boundary-free harmonics of a closed varying-energy orbit."""
    traj, Omega, _, _ = varying_circle
    grid = np.array([bkr.recoil_shifted_harmonic(m, Omega, float(traj.energy.min()))
                     for m in range(1, 9)])
    grid = grid[grid < 0.99 * traj.energy.min()]
    spec = bki.compute_spectrum(traj, Parameters(epsilon=float(traj.energy[0])),
                                grid, theta_max=np.pi, n_theta=16, n_phi=8)
    assert np.all(spec.dW_domega > 0.0)
    assert spec.total_probability() > 0.0
    assert spec.metadata["local_energy"] is True


def test_local_energy_classical_is_energy_independent(varying_circle):
    """Classical mode (eps'_i = eps_i, f = 1) is blind to the energy history:
    the local trace kernel is exactly b1.b2 - 1 and the recoil factor 1, so a
    varying and a constant energy history give identical results."""
    traj, _, dirs, omegas = varying_circle
    beta = traj.beta()
    n = dirs[1]
    for omega in omegas:
        varying = bki.double_time_integral(traj.time, traj.position, beta, omega, n,
                                           10.0, kernel="trace", phase="classical",
                                           backend="numpy", energy=traj.energy).real
        const = bki.double_time_integral(traj.time, traj.position, beta, omega, n,
                                         10.0, kernel="trace", phase="classical",
                                         backend="numpy",
                                         energy=np.full(traj.n_samples, 10.0)).real
        assert np.isclose(varying, const, rtol=1e-12, atol=0.0)


def test_local_energy_validation(varying_circle):
    traj, _, dirs, _ = varying_circle
    args = (traj.time, traj.position, traj.beta())
    good = traj.energy
    with pytest.raises(ValueError):
        # wrong length
        bki.double_time_integral_batch(traj.time, traj.position, traj.beta(),
                                       [1.0], dirs, 10.0, energy=good[:-1])
    with pytest.raises(ValueError):
        # non-positive energy
        bad = good.copy(); bad[3] = -1.0
        bki.double_time_integral_batch(traj.time, traj.position, traj.beta(),
                                       [1.0], dirs, 10.0, energy=bad)
    with pytest.raises(ValueError):
        # omega above a local energy (eps'_i = eps_i - omega <= 0)
        bki.double_time_integral_batch(traj.time, traj.position, traj.beta(),
                                       [traj.energy.min() + 1.0], dirs, 10.0,
                                       energy=good)
    with pytest.raises(ValueError):
        # epsilon_prime is ambiguous in local mode
        bki.double_time_integral(*args, 1.0, dirs[0], 10.0, energy=good,
                                 epsilon_prime=9.0)
    with pytest.raises(ValueError):
        # velocity kernel has no energy dependence
        bki.double_time_integral(*args, 1.0, dirs[0], 10.0, kernel="velocity",
                                 phase="classical", energy=good)
    with pytest.raises(ValueError):
        Trajectory(time=traj.time, position=traj.position, momentum=traj.momentum,
                   energy=np.full((traj.n_samples, 1), 10.0))
