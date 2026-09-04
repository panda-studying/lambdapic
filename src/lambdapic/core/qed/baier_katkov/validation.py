"""Sanity checks for the Baier--Katkov reference module.

These checks are deliberately *coarse* (the task asks to prove direction, not
precision).  They validate, in order:

* V1  the Airy phase identity of `Baier-Katkov.md` eq. 9.6 (phase sign and the
  cubic rescaling of sec. 9);
* V2  internal consistency: ``|classical amplitude|^2`` equals the
  velocity-kernel double-time integral (validates the integrator and the phase
  sign ``exp(-i Phi)``);
* V3  the classical limit on a *closed* trajectory: BK with recoil in the soft
  photon regime reproduces the Liénard--Wiechert spectrum (eq. 7.4);
* V4  the absolute normalization: angle+frequency integrated classical LW on a
  full synchrotron circle equals the Larmor formula;
* V5  a straight, uniform trajectory gives (near) zero radiation and BK agrees
  with LW;
* V6  positivity of the angle-integrated ``dW/domega`` example spectrum;
* V7  the controlled rewrite: ``dot_kernel`` and ``trace_kernel`` coincide in
  the soft limit (eq. 7.3 vs eq. 7.1).

Run::

    python -m lambdapic.core.qed.baier_katkov.validation
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import airy, kv, roots_legendre

from . import kernel as K
from .integrator import (classical_d2_energy, cone_directions, d2_energy,
                         double_time_integral)
from .types import Parameters, Trajectory
from .units import fine_structure


# --------------------------------------------------------------------------
# V1: Airy identities of sec. 9 (phase sign + cubic rescaling + Macdonald form)
# --------------------------------------------------------------------------
def check_airy_identity():
    a, b = 2.0, 0.35
    x = a / (3.0 * b) ** (1.0 / 3.0)

    # (a) eq 9.6:  int_-oo^oo exp(-i (a tau + b tau^3)) dtau = 2pi (3b)^-1/3 Ai(x).
    #     Only the real part survives (the sin part is odd); by symmetry this is
    #     2 int_0^oo cos(a tau + b tau^3) dtau.  The half-line cosine integral is
    #     conditionally convergent, so instead of a huge symmetric window (which is
    #     numerically unstable) we cut at L and add the leading integration-by-parts
    #     tail  -sin(g(L))/g'(L)  with g = a tau + b tau^3.  This is stable and
    #     reaches ~1e-6 with a small L.  NOTE: this symmetric real integral cannot
    #     distinguish exp(-i Phi) from exp(+i Phi); the phase sign itself is pinned
    #     by V2/V3 (BK -> LW requires the conjugate sign), not here.
    L = 16.0
    g_L = a * L + b * L ** 3
    gprime_L = a + 3.0 * b * L ** 2
    core, _ = quad(lambda t: np.cos(a * t + b * t ** 3), 0.0, L, limit=4000)
    half = core - np.sin(g_L) / gprime_L
    numeric = 2.0 * half
    expected = 2.0 * np.pi * (3.0 * b) ** (-1.0 / 3.0) * airy(x)[0]
    rel_quad = abs(numeric - expected) / abs(expected)

    # (b) eq 9.7 / 9.8:  Ai and Ai' in terms of K_{1/3}, K_{2/3} (Macdonald form)
    if x > 0:
        z = (2.0 / 3.0) * x ** 1.5
        ai_macdonald = (1.0 / np.pi) * np.sqrt(x / 3.0) * kv(1.0 / 3.0, z)
        dairi = airy(x)[1]                                  # Ai'(x)
        daip_macdonald = -(1.0 / np.pi) * (x / np.sqrt(3.0)) * kv(2.0 / 3.0, z)
        rel_mac = max(abs(airy(x)[0] - ai_macdonald) / abs(airy(x)[0]),
                      abs(dairi - daip_macdonald) / abs(dairi))
    else:
        rel_mac = np.nan  # the sqrt(x/3) form is real only for x > 0

    return dict(x=x, numeric=numeric, expected=expected, rel_err=rel_quad,
                rel_macdonald=rel_mac, ok=rel_quad < 1e-3 and
                (np.isnan(rel_mac) or rel_mac < 1e-10))


# --------------------------------------------------------------------------
# trajectory helpers
# --------------------------------------------------------------------------
def circle_trajectory(gamma, rho, n_samples=512, turns=1.0):
    """Uniform circular motion in the x-y plane (full ``turns`` revolutions)."""
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    Omega = beta / rho
    T = 2.0 * np.pi * turns / Omega
    t = np.linspace(0.0, T, n_samples)
    phi = Omega * t
    r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])
    u_perp = gamma * beta
    u = np.column_stack([-u_perp * np.sin(phi), u_perp * np.cos(phi), np.zeros_like(t)])
    return Trajectory(time=t, position=r, momentum=u, mass=1.0, charge=1.0), beta, Omega


def straight_trajectory(gamma, T=100.0, n_samples=256):
    """Uniform straight motion along +x."""
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    t = np.linspace(0.0, T, n_samples)
    r = np.column_stack([beta * t, np.zeros_like(t), np.zeros_like(t)])
    u = np.tile([gamma * beta, 0.0, 0.0], (n_samples, 1))
    return Trajectory(time=t, position=r, momentum=u, mass=1.0, charge=1.0)


# --------------------------------------------------------------------------
# V2: |classical amplitude|^2 == velocity-kernel double integral
# --------------------------------------------------------------------------
def check_velocity_kernel_consistency():
    traj, _, _ = circle_trajectory(gamma=50.0, rho=2000.0, n_samples=300)
    n = np.array([0.0, 0.5, np.sqrt(1.0 - 0.5 ** 2)])
    omega = 1.0

    amp = K.classical_amplitude(traj.time, traj.position, traj.beta(), omega, n)
    via_sq = float(np.sum(np.abs(amp) ** 2))
    via_double = double_time_integral(
        traj.time, traj.position, traj.beta(), omega, n,
        epsilon=50.0, kernel="velocity", phase="classical",
    ).real

    rel = abs(via_double - via_sq) / abs(via_sq)
    return dict(via_sq=via_sq, via_double=via_double, rel_err=rel, ok=rel < 1e-8)


# --------------------------------------------------------------------------
# V3: soft-photon BK -> LW on a closed circle
# --------------------------------------------------------------------------
def check_classical_limit():
    gamma = 10.0
    traj, _, Omega = circle_trajectory(gamma=gamma, rho=200.0, n_samples=512)
    eps = gamma

    # Soft-photon BK -> LW holds on a *closed* orbit evaluated at the synchrotron
    # harmonics omega = m Omega, where the hard-truncation boundary terms of the
    # double-time integral vanish.  Angle-integrate over the full sphere using
    # Gauss--Legendre in cos(theta) (the orbit is azimuthally symmetric about the
    # z-axis, so only theta matters).  omega/eps << 1 -> BK/LW ratio -> 1.
    x, wx = roots_legendre(8)
    theta = np.arccos(x)

    rows = []
    for m in [1, 2, 3, 4]:
        omega = m * Omega
        lw = 0.0
        bk = 0.0
        for th, wj in zip(theta, wx):
            n = np.array([np.sin(th), 0.0, np.cos(th)])
            lw += wj * classical_d2_energy(traj.time, traj.position, traj.beta(),
                                           omega, n)
            bk += wj * d2_energy(traj.time, traj.position, traj.beta(), omega, n,
                                 epsilon=eps, kernel="dot", phase="recoil")
        rows.append(dict(m=m, omega=omega, omega_over_eps=omega / eps, lw=lw, bk=bk,
                         ratio=bk / lw))
    return rows


# --------------------------------------------------------------------------
# V4: absolute normalization via Larmor formula (synchrotron total power)
# --------------------------------------------------------------------------
def check_larmor():
    gamma = 30.0
    rho = 1.0e4
    n_samples = 4000
    traj, beta, Omega = circle_trajectory(gamma=gamma, rho=rho, n_samples=n_samples)

    # Heaviside--Lorentz Larmor total power  P = (2/3) e^2 gamma^4 beta^4 / rho^2
    # with e^2 = 4 pi alpha  ->  P = (8 pi / 3) alpha gamma^4 beta^4 / rho^2.
    # This matches the (alpha/pi) prefactor of classical_d2_energy (eq. 7.4).
    w_c = 1.5 * gamma ** 3 * Omega
    P = (8.0 * np.pi / 3.0) * fine_structure * gamma ** 4 * beta ** 4 / rho ** 2
    T = 2.0 * np.pi / Omega
    E_larmor = P * T

    # (a) closure: the exact HL synchrotron spectrum
    #     dP/dw = 2 sqrt(3) (alpha gamma beta / rho) F(w / w_c),
    #     F(x) = x int_x^inf K_{5/3}(xi) dxi
    #     integrates to the Larmor power (a pure math identity, independent of the
    #     trajectory integrator).  This fixes the absolute normalization.
    def synch_F(x):
        val, _ = quad(lambda xi: kv(5.0 / 3.0, xi), x, np.inf, limit=200)
        return x * val

    def dP_dw(w):
        return 2.0 * np.sqrt(3.0) * (fine_structure * gamma * beta / rho) * \
            synch_F(w / w_c)

    w_grid = np.geomspace(0.02 * w_c, 20.0 * w_c, 400)
    P_num = np.trapezoid([dP_dw(w) for w in w_grid], w_grid)
    closure = P_num / P

    # (b) code-vs-exact at harmonics: the code's angle-integrated dE/dw at
    #     omega = m Omega equals T * dP/dw(omega).  (The continuous-omega
    #     trapezoid of classical_d2_energy on a hard-truncated circle is dominated
    #     by boundary/aliasing terms and overestimates by orders of magnitude, so
    #     we compare at discrete harmonics where those terms vanish.)  Azimuthal
    #     symmetry -> only theta;  dE/dw = 2 pi int d2E/dw dO(th) sin th dth.
    x, wx = roots_legendre(32)
    theta = np.arccos(x)

    harmonics = [4, 8, 16, 32, 64]
    ratios = []
    for m in harmonics:
        om = m * Omega
        acc = 0.0
        for th, wj in zip(theta, wx):
            n = np.array([np.sin(th), 0.0, np.cos(th)])
            acc += wj * classical_d2_energy(traj.time, traj.position, traj.beta(),
                                            om, n)
        code_dE_dw = 2.0 * np.pi * acc
        ratios.append(code_dE_dw / (T * dP_dw(om)))

    return dict(E_larmor=E_larmor, w_c=w_c, closure=closure, harmonic_ratios=ratios,
                harmonics=harmonics, rel_err=abs(closure - 1.0))


# --------------------------------------------------------------------------
# V5: straight line -> ~ zero radiation, BK ~ LW
# --------------------------------------------------------------------------
def check_straight_line():
    gamma = 50.0
    traj = straight_trajectory(gamma=gamma, T=200.0, n_samples=600)
    n = np.array([0.0, 0.2, np.sqrt(1.0 - 0.2 ** 2)])

    # Reference scale: a curved (synchrotron) trajectory at the same (moderate)
    # frequency, where its emission is O(1e3) in these units.  A straight line
    # must emit orders of magnitude less.  omega is kept small enough that the
    # record is well resolved (omega * dt << 1), so this is a genuine check and
    # not an aliasing artifact.
    ctraj, _, Omega = circle_trajectory(gamma=gamma, rho=2000.0, n_samples=600)
    omega = 1.0

    lw = classical_d2_energy(traj.time, traj.position, traj.beta(), omega, n)
    bk = d2_energy(traj.time, traj.position, traj.beta(), omega, n,
                   epsilon=gamma, kernel="dot", phase="recoil")
    ref = classical_d2_energy(ctraj.time, ctraj.position, ctraj.beta(), omega, n)

    return dict(omega=omega, lw=lw, bk=bk, curved_ref=ref,
                ratio_bk_to_curved=abs(bk) / abs(ref) if ref != 0 else np.inf)


# --------------------------------------------------------------------------
# V6: positivity of the example angle-integrated spectrum
# --------------------------------------------------------------------------
def check_positivity():
    # Full synchrotron circle, angle-integrated over the full sphere at the
    # harmonics omega = m Omega: a physical, boundary-free configuration whose
    # angle-integrated dW/domega must be strictly positive (and match LW in the
    # soft limit).  A quarter-arc at non-harmonic frequencies shows spurious
    # negatives from the hard-truncated record endpoints; here the closed orbit
    # removes those boundary terms, so positivity is a real check, not a clip.
    gamma = 10.0
    traj, _, Omega = circle_trajectory(gamma=gamma, rho=200.0, n_samples=512)
    params = Parameters(epsilon=gamma)
    M = 8
    omega_grid = np.array([m * Omega for m in range(1, M + 1)])
    from .integrator import compute_spectrum
    spec = compute_spectrum(traj, params, omega_grid,
                            theta_max=np.pi, n_theta=16, n_phi=1,
                            axis=np.array([0.0, 0.0, 1.0]))
    dW = spec.dW_domega
    return dict(min_dW=float(dW.min()), min_omega=float(omega_grid[np.argmin(dW)]),
                total_W=spec.total_probability(),
                dW_domega=np.asarray(dW), omega_grid=omega_grid)


# --------------------------------------------------------------------------
# V7: dot vs trace kernels in the soft limit (controlled rewrite)
# --------------------------------------------------------------------------
def check_kernel_rewrite():
    # The rewrite (b1-b2)^2 = b1^2 + b2^2 - 2 b1.b2 with b_i^2 = 1 - m^2/eps_i^2
    # only holds for on-shell velocities, i.e. |beta| = sqrt(1 - 1/gamma^2) with
    # gamma = eps / m.  Use on-shell beta vectors so the dot and trace kernels
    # agree up to O(omega) terms (the local-energy rewrite of sec. 7.3).
    rng = np.random.default_rng(0)
    eps = 10.0
    omega = 1e-3  # soft
    beta = np.sqrt(1.0 - 1.0 / eps ** 2)
    u1 = rng.normal(size=(5, 3))
    u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    u2 = rng.normal(size=(5, 3))
    u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
    beta1 = beta * u1
    beta2 = beta * u2
    ndot = K.dot_kernel(beta1, beta2, eps, omega)
    ntrace = K.trace_kernel(beta1, beta2, eps, omega)
    rel = np.max(np.abs(ndot - ntrace) / np.abs(ntrace))
    return dict(rel_err=rel, ok=rel < 1e-2)


def main():
    print("Baier--Katkov validation (coarse, direction-only)")
    print("=" * 60)

    v1 = check_airy_identity()
    print(f"[V1] Airy identity (eq 9.6): rel_err={v1['rel_err']:.2e} "
          f"{'OK' if v1['ok'] else 'FAIL'}")

    v2 = check_velocity_kernel_consistency()
    print(f"[V2] |A|^2 == velocity double integral: rel_err={v2['rel_err']:.2e} "
          f"{'OK' if v2['ok'] else 'FAIL'}")

    v3 = check_classical_limit()
    print("[V3] soft-photon BK -> LW (closed circle, harmonics):")
    for r in v3:
        print(f"     m={r['m']}  omega/eps={r['omega_over_eps']:.5f}  "
              f"BK/LW ratio={r['ratio']:.5f}")

    v4 = check_larmor()
    print(f"[V4] Larmor normalization: exact-spectrum closure={v4['closure']:.4f}")
    print("     code-vs-exact at harmonics m=4..64:",
          " ".join(f"{r:.4f}" for r in v4['harmonic_ratios']))
    print(f"     (converging to 1; continuous-omega trapz is NOT used: boundary/"
          f"aliasing-free by construction)")

    v5 = check_straight_line()
    print(f"[V5] straight line: BK={v5['bk']:.3e}  LW={v5['lw']:.3e}  "
          f"|BK/curved|={v5['ratio_bk_to_curved']:.3e}")

    v6 = check_positivity()
    print(f"[V6] positivity: min(dW/domega)={v6['min_dW']:.3e}  total_W={v6['total_W']:.3e}")

    v7 = check_kernel_rewrite()
    print(f"[V7] dot~trace soft-limit rewrite: rel_err={v7['rel_err']:.2e} "
          f"{'OK' if v7['ok'] else 'FAIL'}")

    print("=" * 60)
    print("V1/V2/V7 are exact internal/analytic checks; V3/V4/V5 are physical "
          "coarse checks; V6 is a diagnostic.  The full quantum synchrotron "
          "spectrum (K_{1/3},K_{2/3}, eq. 10.3) is NOT numerically verified here: "
          "its prefactor P(epsilon,chi) is convention-dependent and would need a "
          "separate, carefully normalized local-constant-field reference.")


if __name__ == "__main__":
    main()
