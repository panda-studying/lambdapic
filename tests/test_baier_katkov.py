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

import warnings

import numpy as np
import pytest

from lambdapic.core.qed.baier_katkov import integrator as bki
from lambdapic.core.qed.baier_katkov import kernel as bkk
from lambdapic.core.qed.baier_katkov import reference as bkr
from lambdapic.core.qed.baier_katkov import validation as bkv
from lambdapic.core.qed.baier_katkov import coherence as bkc
from lambdapic.core.qed.baier_katkov.types import Parameters, Trajectory
from lambdapic.core.qed.baier_katkov.units import fine_structure, natural_time_unit

# Most tests here are *structural* checks that deliberately use non-production
# regimes: coarse grids for the round-off / numba-vs-numpy / |A|^2-identity
# checks, and short records for the local-energy degeneracy, hermiticity and
# positivity checks.  They trip the integrator's adequacy guards, whose job is
# to protect production spectra -- so they are silenced here.  The guards
# themselves are pinned by test_adequacy_* / test_sampling_margin_* /
# test_record_adequacy_* below, and physical adequacy by validation V1--V8.
pytestmark = pytest.mark.filterwarnings(
    "ignore::lambdapic.core.qed.baier_katkov.integrator.SamplingWarning",
    "ignore::lambdapic.core.qed.baier_katkov.integrator.RecordLengthWarning",
    "ignore::lambdapic.core.qed.baier_katkov.integrator.AngularConvergenceWarning",
)

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


# --------------------------------------------------------------------------
# adequacy guards (P-3): trapezoid aliasing and record-length truncation
#
# The two guards protect against the two ways the double-time sum silently
# returns non-physics.  Both are pinned here against the *measured* failure
# boundaries, so the thresholds cannot drift away from the data.
# --------------------------------------------------------------------------
def _recorded_warnings(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return list(rec)


def test_sampling_margin_tracks_the_aliasing_boundary():
    """The Nyquist margin reproduces where the measured spectrum breaks down.

    Closed circle at ``delta = 0.5`` (``gamma = 10``, ``chi = 0.5``): the
    recoil-shifted harmonic is ``m ~ 1990``, so ``Nt`` must exceed ``4m``.
    Measured ``dE/domega`` / exact = 2126 (Nt=512), -82 (Nt=2000), 2.16
    (Nt=3980) and 1.0007 (Nt=7960).  The margin must split them at exactly the
    same place: ``>= 1`` for the first three, ``< 1`` for the last.  (``Nt =
    2m`` -- the laxer criterion quoted in validation.V8 -- is *inside* the
    broken region, which is why the guard is a Nyquist one.)
    """
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(round(gamma / Omega))                      # delta = 0.5
    omega = float(bkr.recoil_shifted_harmonic([m], Omega, gamma)[0])
    dirs, _ = bkr.orbit_direction_grid(gamma, n_inner=24, n_outer=8)

    for n_samples, aliased in ((512, True), (2000, True), (3980, True),
                               (7960, False), (15920, False)):
        traj, _ = circle(gamma, rho, n_samples)
        bk = bki.BKIntegrator(traj, Parameters(epsilon=gamma), checks="ignore")
        margin = float(bk.adequacy([omega], dirs)[0][0])
        assert (margin >= 1.0) is aliased, (n_samples, margin)


def test_record_adequacy_tracks_the_open_arc_deficit():
    """``L/tau_f`` separates the exact closed orbit from collapsing arcs.

    Measured at the first harmonic of the same circle, ratio to ``L dP/domega``:
    1.00 (closed), 0.89 (0.9 turns), 0.72 (0.8), 0.48 (0.7), 0.21 (0.6),
    -0.03 (0.5), -0.19 (0.4) -- degradation below ``L/tau_f ~ 1.3`` and collapse
    (sign flip) below ``~ 1``, while the closed orbit is exact at 1.61.  The
    1.5 threshold must therefore flag the arcs and leave the closed orbit alone.
    """
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    omega = float(bkr.recoil_shifted_harmonic([1], Omega, gamma)[0])
    n_hat = [[1.0, 0.0, 0.0]]

    def adequacy(turns, n_samples):
        traj, _ = circle(gamma, rho, n_samples, turns=turns)
        bk = bki.BKIntegrator(traj, Parameters(epsilon=gamma), checks="ignore")
        return float(bk.adequacy([omega], n_hat)[1][0])

    closed = adequacy(1.0, 1200)
    half = adequacy(0.5, 600)
    quarter = adequacy(0.25, 300)
    assert closed > 1.5            # closed orbit: exact, must not be flagged
    assert half < 1.5              # collapses (measured -0.03)
    assert quarter < 1.5           # collapses (measured -0.19)
    assert quarter < half < closed  # monotone in record length


def test_adequacy_guards_fire_and_are_controllable():
    """Both guards warn by default, can be raised, and can be silenced."""
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(round(gamma / Omega))
    omega = float(bkr.recoil_shifted_harmonic([m], Omega, gamma)[0])
    dirs, _ = bkr.orbit_direction_grid(gamma, n_inner=24, n_outer=8)
    traj, _ = circle(gamma, rho, 512)                 # heavily aliased

    def run(checks):
        return bki.BKIntegrator(traj, Parameters(epsilon=gamma),
                                checks=checks).d2_energy_batch([omega], dirs)

    cats = {type(w.message).__name__ for w in _recorded_warnings(lambda: run("warn"))}
    assert "SamplingWarning" in cats

    with pytest.raises(bki.SamplingWarning):
        run("raise")

    assert _recorded_warnings(lambda: run("ignore")) == []

    with pytest.raises(ValueError):
        run("nonsense")


def test_record_adequacy_is_not_applicable_to_a_straight_record():
    """A straight record has ``tau_f = inf`` -> adequacy ``inf``, never flagged.

    There the kernel's vacuum subtraction removes the straight-line part, so no
    truncation deficit arises and the check must not fire (returning 0 instead
    of inf would read as "maximally inadequate").
    """
    n_samples = 256
    t = np.linspace(0.0, 100.0, n_samples)
    v = 0.9
    traj = Trajectory(
        time=t,
        position=np.column_stack([v * t, np.zeros_like(t), np.zeros_like(t)]),
        momentum=np.tile([10.0 * v, 0.0, 0.0], (n_samples, 1)),
    )
    bk = bki.BKIntegrator(traj, Parameters(epsilon=10.0), checks="ignore")
    adequacy = bk.adequacy([1.0], [[1.0, 0.0, 0.0]])[1]
    assert np.isinf(adequacy[0])
    assert _recorded_warnings(
        lambda: bki.BKIntegrator(traj, Parameters(epsilon=10.0)).d2_energy_batch(
            [1.0], [[1.0, 0.0, 0.0]])) == []


def test_spectrum_metadata_carries_adequacy_diagnostics():
    """``compute_spectrum`` reports the per-omega margins alongside the spectrum."""
    gamma = 10.0
    traj, _ = circle(gamma, 200.0, 512)
    grid = np.array([bkr.recoil_shifted_harmonic([1], np.sqrt(1 - 1 / gamma ** 2) / 200.0,
                                                 gamma)[0]])
    spec = bki.compute_spectrum(traj, Parameters(epsilon=gamma), grid,
                                theta_max=np.pi, n_theta=16, n_phi=1,
                                axis=np.array([0.0, 0.0, 1.0]))
    assert spec.metadata["sampling_margin"].shape == grid.shape
    assert spec.metadata["record_adequacy"].shape == grid.shape
    assert np.all(spec.metadata["sampling_margin"] >= 0.0)


# --------------------------------------------------------------------------
# angular adequacy (P-4): adaptive cone, two-panel grid, convergence guard
#
# The historical fixed cone (``theta_max = 5/gamma`` about ``beta[0]``)
# silently truncated the angular integral for two independent reasons: the
# emission half-angle ``theta_c = (4/m)**(1/3)`` *widens* towards low omega
# (4.6/gamma at delta = 0.02, down to 0.95/gamma at delta = 0.7 -- the opposite
# of what roadmap P-4 originally claimed), and it ignored the velocity swing
# over the record, which is ``2 pi`` for a closed orbit where the fixed cone
# captures about 1%.  The cone is now adapted to the record, the ``1/gamma``
# core is split off as its own Gauss--Legendre panel, and a third guard reports
# when the direction grid still under-resolves the result.
# --------------------------------------------------------------------------
def test_cone_directions_two_panel_conserves_weight():
    """Two panels partition the cone exactly; directions stay unit vectors."""
    axis = [0.0, 0.0, 1.0]
    for theta_max, split, n_inner in ((np.pi, 0.8, 12), (0.5, 0.25, 8)):
        d1, w1 = bki.cone_directions(axis, theta_max, 16, 8)
        d2, w2 = bki.cone_directions(axis, theta_max, 16, 8,
                                     split=split, n_inner=n_inner)
        solid = 2.0 * np.pi * (1.0 - np.cos(theta_max))
        assert w1.sum() == pytest.approx(solid)
        assert w2.sum() == pytest.approx(solid)
        assert w2.shape == ((n_inner + 16) * 8,)
        assert np.allclose(np.linalg.norm(d2, axis=1), 1.0)


def test_emission_half_angle_is_widest_at_low_omega():
    """``theta_c**3 = 4 eps' Omega_eff / (omega eps)`` narrows towards high omega.

    At the integer recoil-shifted harmonics of the ``gamma = 10``, ``chi = 0.5``
    circle, ``theta_c`` falls from ``4.6/gamma`` at ``delta = 0.02`` to
    ``0.95/gamma`` at ``delta = 0.7``.  The measured half-energy emission angle
    (BK kernel, converged 128-node rule) is ``0.35 theta_c``, i.e.
    ``theta_50 gamma m**(1/3) = 4.6 ... 6.2`` over the same range -- so the
    soft-photon cone is the wide one, and the old fixed cone cut it off.
    """
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=512)
    rate = bki._curvature_rate(traj.time, traj.beta())
    # finite differences on a 512-point turn: O(dt**2) below the exact beta*Omega
    assert rate == pytest.approx(beta * Omega, rel=1e-5)

    omega = bkr.recoil_shifted_harmonic(
        bkv.harmonics_for_deltas([0.02, 0.1, 0.3, 0.7], Omega, gamma), Omega, gamma)
    eps_p = gamma - omega
    theta_c = bki.emission_half_angle(omega, gamma, eps_p, rate)

    assert np.all(np.diff(theta_c) < 0.0)                      # narrows with omega
    assert theta_c[0] * gamma == pytest.approx(4.6, abs=0.15)
    assert theta_c[-1] * gamma == pytest.approx(0.95, abs=0.05)
    # theta_c**3 == 4 / m with m = omega eps / (eps' Omega) the harmonic index
    m_eff = omega * gamma / (eps_p * Omega)
    assert np.allclose(theta_c ** 3, 4.0 * beta / m_eff)


def test_velocity_swing_separates_a_closed_orbit_from_a_straight_record():
    """A full turn reaches the antipode (``pi``); a straight record never turns."""
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    n = 2000
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=n)
    # the antipode is only reached to within one sample spacing of phi
    assert bki.velocity_swing(traj.beta()) == pytest.approx(np.pi, abs=2 * np.pi / n)

    n = 64
    t = np.linspace(0.0, 20.0, n)
    v = 0.9
    line = Trajectory(
        time=t,
        position=np.column_stack([v * t, np.zeros_like(t), np.zeros_like(t)]),
        momentum=np.tile([10.0 * v, 0.0, 0.0], (n, 1)),
    )
    assert bki.velocity_swing(line.beta()) == 0.0


def test_default_theta_max_covers_the_emission_band():
    """Swing + 3 theta_c, capped at ``pi`` and floored at the old ``5/gamma``."""
    gamma, chi = 10.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=512)
    omega = bkr.recoil_shifted_harmonic(
        bkv.harmonics_for_deltas([0.5], Omega, gamma), Omega, gamma)

    # a closed orbit turns by pi -> the cone has to open all the way
    assert bki.default_theta_max(traj.time, traj.beta(), omega, gamma) == np.pi

    # a straight record has no curvature (theta_c = 0) and no swing, so the
    # floor keeps the historical default rather than collapsing to zero
    n = 64
    t = np.linspace(0.0, 20.0, n)
    v = 0.9
    line = Trajectory(
        time=t,
        position=np.column_stack([v * t, np.zeros_like(t), np.zeros_like(t)]),
        momentum=np.tile([10.0 * v, 0.0, 0.0], (n, 1)),
    )
    assert bki.emission_half_angle([1.0], 10.0, 9.0,
                                   bki._curvature_rate(line.time, line.beta()))[0] == 0.0
    assert bki.default_theta_max(line.time, line.beta(), [1.0], 10.0,
                                 floor=0.5) == pytest.approx(0.5)


def test_angular_convergence_guard_tracks_the_measured_grid_boundary():
    """The guard fires exactly where the angular integral is measurably wrong.

    ``gamma = 5``, ``chi = 0.5`` circle at the integer recoil-shifted harmonic
    of ``delta = 0.3`` (``m = 105``, ``Nt = 8m = 840``), ``axis = z``,
    ``theta_max = pi``.  Measured against the exact constant-field spectrum:
    ``n_theta = 12`` -> ``dE/domega`` / exact = 1.046 and the guard fires
    (conv = 0.040); ``n_theta = 16`` -> 1.00000 and it is silent
    (conv = 0.004).  The threshold must split them the same way.
    """
    gamma, chi = 5.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(bkv.harmonics_for_deltas([0.3], Omega, gamma)[0])
    omega = np.array([bkr.recoil_shifted_harmonic([m], Omega, gamma)[0]])
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=8 * m)
    exact = ((2.0 * np.pi / Omega) * (omega[0] / gamma)
             * bkr.quantum_synchrotron_rate(omega[0] / gamma,
                                            bkr.circle_chi(gamma, rho), gamma))

    def spectrum(n_theta):
        return bki.BKIntegrator(traj, Parameters(epsilon=gamma)).compute_spectrum(
            omega, theta_max=np.pi, n_theta=n_theta, n_phi=1, axis=[0.0, 0.0, 1.0])

    coarse = spectrum(12)
    assert coarse.dE_domega[0] / exact == pytest.approx(1.046, abs=0.01)
    assert coarse.metadata["angular_convergence"] > bki.ANGULAR_RTOL

    fine = spectrum(16)
    assert fine.dE_domega[0] / exact == pytest.approx(1.0, abs=0.005)
    assert fine.metadata["angular_convergence"] < bki.ANGULAR_RTOL


def test_angular_convergence_guard_fires_on_a_swinging_record():
    """A closed orbit with the default axis is flagged, not silently returned.

    The emission of a full turn is a ring at 90 degrees from ``beta[0]``, so no
    cone about the initial velocity can hold it: the adaptive default opens to
    ``pi`` but a 16-node grid is still not converged, and the guard must say so
    (measured ``dE/domega`` / exact = 2.39 at ``delta = 0.3``, conv = 0.45).
    """
    gamma, chi = 5.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(bkv.harmonics_for_deltas([0.3], Omega, gamma)[0])
    omega = np.array([bkr.recoil_shifted_harmonic([m], Omega, gamma)[0]])
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=8 * m)
    bk = bki.BKIntegrator(traj, Parameters(epsilon=gamma))

    cats = {type(w.message).__name__
            for w in _recorded_warnings(lambda: bk.compute_spectrum(omega))}
    assert "AngularConvergenceWarning" in cats


def test_angular_convergence_guard_is_controllable():
    """``checks="raise"`` turns the guard into an exception; ``ignore`` silences."""
    gamma, chi = 5.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(bkv.harmonics_for_deltas([0.3], Omega, gamma)[0])
    omega = np.array([bkr.recoil_shifted_harmonic([m], Omega, gamma)[0]])
    traj, _, _ = bkv.circle_trajectory(gamma=gamma, rho=rho, n_samples=8 * m)

    def run(checks):
        return bki.BKIntegrator(traj, Parameters(epsilon=gamma),
                                checks=checks).compute_spectrum(
            omega, theta_max=np.pi, n_theta=12, n_phi=1, axis=[0.0, 0.0, 1.0])

    assert _recorded_warnings(lambda: run("ignore")) == []
    assert np.isnan(run("ignore").metadata["angular_convergence"])

    with pytest.raises(bki.AngularConvergenceWarning):
        run("raise")

    cats = {type(w.message).__name__ for w in _recorded_warnings(lambda: run("warn"))}
    assert "AngularConvergenceWarning" in cats


def test_spectrum_metadata_carries_angular_diagnostics():
    """The cone, its panel split and the angular margins go into ``metadata``."""
    gamma = 10.0
    traj, _, Omega = bkv.circle_trajectory(
        gamma=gamma, rho=gamma ** 2 * (1 - 1 / gamma ** 2) / 0.5, n_samples=512)
    grid = np.array([bkr.recoil_shifted_harmonic([1], Omega, gamma)[0]])
    spec = bki.compute_spectrum(traj, Parameters(epsilon=gamma), grid,
                                theta_max=np.pi, n_theta=16, n_phi=1,
                                axis=np.array([0.0, 0.0, 1.0]))
    meta = spec.metadata
    assert meta["theta_max_auto"] is False
    assert meta["theta_max"] == pytest.approx(np.pi)
    assert meta["theta_split"] is None               # explicit theta_max: old grid
    assert meta["angular_edge_fraction"].shape == grid.shape
    assert meta["emission_half_angle"].shape == grid.shape
    assert meta["velocity_swing"] >= 0.0

    auto = bki.compute_spectrum(traj, Parameters(epsilon=gamma), grid)
    assert auto.metadata["theta_max_auto"] is True
    assert auto.metadata["theta_split"] == pytest.approx(min(0.8, 0.5 * auto.metadata["theta_max"]))
    assert auto.metadata["n_inner"] == 16



# --------------------------------------------------------------------------
# Test B: inter-turn coherence (coherence.py) -- deliberately NOT part of V1-V8
#
# LCFA is a locally constant field approximation: an incoherent integral of a
# local rate, with no phase.  BK keeps the phase, so on a closed orbit the turns
# add coherently and the line-centre density grows as n**2 while the LCFA
# continuum grows only as n.  These two tests pin the cheapest end of that law
# (full scan and figure: `python -m lambdapic.core.qed.baier_katkov.coherence`).
# --------------------------------------------------------------------------
def test_coherence_window_sits_on_the_recoil_shifted_harmonic():
    """The omega window is centred on ``omega_m`` and spaced ``Omega (eps'/eps)**2``."""
    gamma, chi = 5.0, 0.5
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(bkv.harmonics_for_deltas([0.5], Omega, gamma)[0])
    centre, spacing, grid = bkc.line_centre_grid(m, Omega, gamma, 11, 2.5)

    assert centre == pytest.approx(
        float(bkr.recoil_shifted_harmonic([m], Omega, gamma)[0]))
    assert spacing == pytest.approx(Omega * ((gamma - centre) / gamma) ** 2)
    assert grid.size == 11
    assert np.min(np.abs(grid - centre)) < 1e-15   # odd count -> centre on the grid


def test_coherence_line_centre_density_scales_as_n_squared():
    """Two cheap points pin the ``n**2`` law (n = 1 is V8's own statement).

    gamma = 5, chi = 0.5, delta = 0.5.  Measured line-centre density divided by
    the one-turn ``T dP/domega``: 1.0030 at n = 1 and 4.0120 at n = 2, i.e. the
    LCFA continuum -- which grows only linearly with the observation time --
    under-reports the coherent line density by a factor n.  The full scan gives
    1.0030, 4.0120, 16.0480, 64.1920, 256.7679 for n = 1, 2, 4, 8, 16, a
    constant +0.30% (the angular quadrature, common to all n) on n**2.
    """
    ratio = {}
    for n in (1, 2):
        row = bkc.turn_spectrum(5.0, 0.5, 0.5, n, n_omega=11)
        ratio[n] = row["dE_domega"][row["i_centre"]] / row["lcfa_one_turn"]
    assert ratio[1] == pytest.approx(1.003, abs=0.01)
    assert ratio[2] == pytest.approx(4.012, abs=0.04)
    assert ratio[2] / ratio[1] == pytest.approx(4.0, rel=0.01)


def test_coherence_line_has_no_width_at_one_turn():
    """A one-turn record has no line structure at all.

    The Dirichlet kernel of a single period is identically 1, so one turn is
    wider than the line spacing and there is no width to measure -- which is
    precisely why V8 (one turn, evaluated at the harmonic) cannot see this
    effect.
    """
    row = bkc.turn_spectrum(5.0, 0.5, 0.5, 1, n_omega=11)
    width = bkc.measure_line_width(row["omega"], row["dE_domega"], row["i_centre"])
    assert np.isnan(width)
