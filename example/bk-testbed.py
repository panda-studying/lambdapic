"""Testbed: record a *real* PIC trajectory and validate Baier--Katkov on it.

Two jobs, in the order they matter:

1. **End-to-end validation against a known answer** (2026-09-16).  Every
   ingredient of the PIC -> spectrum chain is validated on its own -- V1--V8 on
   analytic trajectories, the recorder on synthetic records -- but the *seam*
   (Yee fields -> Boris push -> recorder -> SI adapter -> double-time integral)
   had never been compared with an analytic spectrum.  A uniform field makes it
   possible: the field along the orbit is exactly constant, so
   ``reference.quantum_synchrotron_rate`` **is** the analytic evaluation of the
   integral the module computes (the same argument as V8, which does it on an
   analytic circle rather than on a PIC record).  ``--compare`` turns it on.
2. **An adequate record for the truncation question**.  The banded-truncation
   verdict is decisive only on a closed orbit; the aperiodic case is what a PIC
   actually produces.  This orbit is periodic, but it is a *real* PIC record:
   Boris pusher, field interpolation, SI units, the recorder, the adapter.

Why a magnetic field and not a laser: the module's sampling criterion is
``dt * (omega * eps/eps') * max_n|1 - n.v| < pi``, and the ``max`` is taken over
directions -- it is set by the *backward* direction (about 2), not by the
forward ``1/gamma`` lobe.  A laser-driven record evaluated near its spectral
peak needs 10^2-10^3 steps per laser period, while a CFL-limited ``dx =
lambda/32`` grid delivers only 48.  The gyro-orbit is under-resolved by a much
smaller factor -- but note it is *not* resolved at ``omega_c``: with the
defaults the margin is 3.17 there, so only ``omega <~ 0.6 omega_c`` is usable
(the driver prints the margin; trust it, not this paragraph).  The comparison
therefore uses ``delta <= 0.3``.

**There is no field-gradient option, deliberately.**  A static, spatially
varying ``B`` is not a source-free Maxwell solution: ``B = B_z(y) z-hat`` is
divergence-free but ``(curl B)_x = dB_z/dy != 0`` with ``j = 0``, so the Yee
update pumps ``E_x`` by ``c^2 dt dB_z/dy`` every step.  Measured with such a
gradient: ``E_x`` reaches 1e-3 E_cr, the electron's ``|u|`` ripples by 6.5%
(exactly linear in the gradient) and the apparent guiding-centre motion is
E x B drift, not grad-B drift -- i.e. the record is not the recoilless
background the Baier-Katkov module requires.  Any future decoherence knob has
to be a genuine Maxwell solution (a laser pulse, or a field with its
supporting current).

Usage::

    python example/bk-testbed.py --periods 10 --out /tmp/bk_g0.npz
    python example/bk-testbed.py --periods 4 --compare          # + spectrum check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# The simulation checks PyPI for a newer release at the start of every run().
# This machine has no direct internet, so that call blocks for ~9 s and then
# times out -- it dominates the whole runtime (measured: 9.1 s of a 9.6 s run).
os.environ.setdefault("LAMBDAPIC_CHECK_UPDATE", "0")

from scipy.constants import c, e, hbar, m_e                     # noqa: E402

from lambdapic import Simulation                                 # noqa: E402
from lambdapic.callback.trajectory import TrajectoryRecorder     # noqa: E402
from lambdapic.core.species import Electron                      # noqa: E402

# natural (Compton) unit scales, matching baier_katkov.units
LAMBDA_C = hbar / (m_e * c)          # length unit  [m]
TIME_C = hbar / (m_e * c ** 2)       # time unit    [s]
# Schwinger *magnetic* field, B_cr = m^2 c^2 / (e hbar) = 4.41e9 T.  Note the
# critical *electric* field is m^2 c^3 / (e hbar) = 1.32e18 V/m, larger by c --
# core/qed/inline.py's factor e*hbar/(m^2 c^3) is 1/E_cr, and there the B field
# enters multiplied by c, so the same factor converts B/B_cr correctly.
B_CRIT = m_e ** 2 * c ** 2 / (e * hbar)

# physics of the testbed (natural units unless stated)
GAMMA0 = 5.0
CHI = 0.5
DX_NAT = 0.5                 # cell size, natural units
NX = NY = 256               # box; the gyro-orbit diameter is 2*rho ~ 99 natural
NPATCH_X = NPATCH_Y = 4


def larmor_radius(gamma: float, chi: float) -> float:
    """``rho = gamma^2 beta^2 / chi`` (natural units), as in reference.circle_chi."""
    beta2 = 1.0 - 1.0 / gamma ** 2
    return gamma ** 2 * beta2 / chi


def build_simulation(periods: float):
    """One electron in a uniform static ``B_z`` (a source-free Maxwell solution)."""
    gamma = GAMMA0
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = larmor_radius(gamma, CHI)
    dx = DX_NAT * LAMBDA_C
    dt = 0.95 * dx / (np.sqrt(2.0) * c)          # 2D CFL, as Simulation._validate
    dt_nat = dt / TIME_C
    # chi = gamma*beta*B/B_cr, so B = chi/(gamma*beta) -- using chi/gamma would
    # realize chi = 0.4899 and rho = 48.99 instead of the stated 0.5 and 48.
    B = CHI / (gamma * beta) * B_CRIT
    omega_c_nat = CHI / (gamma ** 2 * beta)      # gyro-frequency B/γ, natural units
    omega_crit = 1.5 * gamma ** 3 * omega_c_nat  # reference.circle_omega_c
    # margin = dt * (omega * eps/eps') * max_n|1 - n.v| / pi, taken from the
    # module itself.  The omega factor is essential and is easy to drop when the
    # formula is re-derived by hand -- which happened here (0.85 printed instead
    # of the true 2.93 at omega_c), so call the module rather than inline it.
    from lambdapic.core.qed.baier_katkov.integrator import (cone_directions,
                                                            sampling_margin)
    _dirs, _ = cone_directions(np.array([0.0, 0.0, 1.0]), np.pi, 32, 1)
    margin = float(sampling_margin(
        np.arange(3) * dt_nat, np.tile([0.0, beta, 0.0], (3, 1)), _dirs,
        np.array([omega_crit]),
        np.array([omega_crit * gamma / (gamma - omega_crit)]))[0])

    nsteps = int(round(periods * 2.0 * np.pi / omega_c_nat / dt_nat))
    ny = NY

    print(f"gamma={gamma}  chi={CHI}  rho={rho:.2f} natural  "
          f"(box {NX * DX_NAT:.0f} x {ny * DX_NAT:.0f} natural)")
    print(f"dx={DX_NAT} natural  dt={dt_nat:.4f} natural  "
          f"steps/period={2 * np.pi / omega_c_nat / dt_nat:.0f}")
    print(f"B={B:.4g} T  periods={periods}  nsteps={nsteps}")
    print(f"sampling margin at omega_c = {margin:.3f} "
          f"({'OK' if margin < 1 else 'ALIASED -- only omega <~ 0.6 omega_c is usable'})")

    sim = Simulation(nx=NX, ny=ny, dx=dx, dy=dx,
                     npatch_x=NPATCH_X, npatch_y=NPATCH_Y, dt_cfl=0.95)

    # Exactly one electron with weight 1: a hard-edged disc covering a single
    # cell centre.  A Gaussian would fill the whole box (the density is only a
    # threshold test and a Gaussian is positive everywhere), and the earlier
    # smoke test duly produced 6032 particles spread to the walls.  w =
    # density*dx*dy/ppc, so density = ppc/(dx*dy) gives w = 1 -- a single
    # electron, not a macroparticle standing for many.
    #
    # The disc is centred on the *start* of the gyro-orbit, not on the box
    # centre: the particle is created inside whichever patch owns that cell, and
    # setting its position afterwards to a cell in a different patch makes the
    # guard exchange treat it as out of bounds and kill it (measured: x -> NaN
    # and is_dead on the first step).
    x_start = 0.5 * NX * dx
    y_start = 0.5 * ny * dx - rho * LAMBDA_C
    cell_area = dx * dx

    def disc(x, y):
        return (1.0 / cell_area
                if (x - x_start) ** 2 + (y - y_start) ** 2 <= (0.5 * dx) ** 2 else 0.0)

    ele = Electron(density=disc, ppc=1)          # radiation=None -> recoilless
    sim.add_species([ele])
    sim.initialize()

    _set_magnetic_field(sim, B)
    # Start at the bottom of the gyro-orbit so the guiding centre sits at the
    # box centre: a particle at position r with velocity +x about B = +z circles
    # a centre at r + rho y-hat, so starting at the box centre would push half
    # the orbit (2*rho ~ 98 natural here) outside the box and into the PML.
    _place_electron(sim, ele, x_start, y_start, gamma, beta)

    return sim, ele, nsteps, dict(gamma=gamma, beta=beta, rho=rho, dt=dt,
                                  dt_nat=dt_nat, omega_c=omega_c_nat,
                                  B=B, margin=margin, nsteps=nsteps)


def _set_magnetic_field(sim, B: float) -> None:
    """Write a **uniform** ``B_z = B`` (tesla) into every patch.

    Uniform and static is the whole point: a spatially varying static ``B`` is
    not source-free (``curl B != 0`` with no current), so the solver would pump
    an E field and the record would stop being a recoilless background.  See the
    module docstring.

    The array layout is ``[interior, upper guard, lower guard]``, *not* centred
    on the interior -- ``p.fields.yaxis[0, :]`` carries the true coordinate of
    each row and is the only safe way to index it (a hand-built ``(j - ng)*dy``
    profile is shifted by ``n_guard`` cells).
    """
    for p in sim.patches:
        p.fields.bz[:, :] = B


def _place_electron(sim, ele, x0: float, y0: float, gamma: float, beta: float) -> None:
    """Put the single electron on a gyro-orbit in the x-y plane."""
    u = gamma * beta
    for p in sim.patches:
        part = p.particles[ele.ispec]
        idx = np.nonzero(part.is_alive)[0]
        if not idx.size:
            continue
        part.x[idx[0]] = x0
        part.y[idx[0]] = y0
        part.z[idx[0]] = 0.0
        part.ux[idx[0]] = u          # perpendicular to B (= z)
        part.uy[idx[0]] = 0.0
        part.uz[idx[0]] = 0.0
        part.inv_gamma[idx[0]] = 1.0 / gamma


def _orbit_omega(traj):
    """Measured rotation rate of the recorded orbit, and the fit residual.

    The gyro-phase ``atan2(beta_y, beta_x)`` is linear in time for uniform
    circular motion.  The rate is **measured** rather than taken from
    ``chi/(gamma^2 beta)`` because the Boris rotation advances by
    ``2 atan(Omega dt/2)`` per step: the discrete orbit has a slightly longer
    period and a slightly larger radius than the continuous one, and the
    reference spectrum must be evaluated for the orbit the particle actually
    has.  Reporting the difference separately keeps the pusher's discretization
    error apart from the module's.

    The residual is a one-number noise diagnostic: it is the angular jitter the
    record carries on top of uniform rotation (field interpolation, round-off,
    the half-step staggering of the pusher).
    """
    beta = traj.beta()
    theta = np.unwrap(np.arctan2(beta[:, 1], beta[:, 0]))
    A = np.column_stack([traj.time, np.ones_like(traj.time)])
    fit = np.linalg.lstsq(A, theta, rcond=None)[0]
    return float(fit[0]), float(np.max(np.abs(theta - A @ fit)))


def compare_spectrum(path, meta, turns=(1, 2)) -> None:
    """BK on the recorded PIC orbit vs the exact synchrotron spectrum.

    The end-to-end check: the same comparison V8 makes on an *analytic* circle,
    but here the trajectory comes out of the real pipeline (Yee fields, Boris
    pusher, field interpolation, SI units, recorder, SI adapter).  For uniform
    circular motion the field along the trajectory is exactly constant, so
    ``reference.quantum_synchrotron_rate`` is the analytic evaluation of the
    integral the module computes, and one turn's angle-integrated
    ``dE/domega`` at a recoil-shifted harmonic equals ``T dP/domega``.

    Three things are reported: the ratio to the exact spectrum (the end-to-end
    number), the measured-vs-continuous orbit (the pusher's error), and the
    ratio between one and two turns at the same harmonics (the inter-turn
    coherence law of section 3.4, now on a PIC record: it must be 4).
    """
    from lambdapic.core.qed.baier_katkov import reference as R
    from lambdapic.core.qed.baier_katkov.integrator import (BKIntegrator,
                                                            sampling_margin)
    from lambdapic.core.qed.baier_katkov.trajectory import as_trajectory_si
    from lambdapic.core.qed.baier_katkov.types import Parameters, Trajectory
    from lambdapic.core.qed.baier_katkov.validation import harmonics_for_deltas

    d = np.load(path)
    n = int(np.atleast_1d(d["n_samples"])[0])
    t_si, x_si, u = d["t"][0][:n], d["x"][0][:n], d["u"][0][:n]

    # The PIC stores u, and the adapter derives eps(t) = sqrt(1 + |u|^2) from
    # it when energy is not given -- exactly the on-shell energy the kernel
    # assumes.  Nothing has to be passed, and nothing can be silently wrong.
    traj = as_trajectory_si(t_si, x_si, u=u)
    assert traj.energy is not None          # derived, not the fixed-eps fallback

    Omega, resid = _orbit_omega(traj)
    gamma = float(traj.energy.mean())
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    chi = gamma ** 2 * beta * Omega                  # = gamma*beta*B/B_cr
    rho = beta / Omega
    T = 2.0 * np.pi / Omega
    print(f"\nmeasured orbit: Omega={Omega:.6f} "
          f"(continuous {meta['omega_c']:.6f}, ratio {Omega / meta['omega_c']:.6f})  "
          f"rho={rho:.3f} (continuous {meta['rho']:.3f})  chi={chi:.4f}")
    print(f"phase-fit residual: {resid:.2e} rad over "
          f"{traj.time[-1] - traj.time[0]:.1f} natural units")

    per_turn = int(round(T / float(np.median(np.diff(traj.time)))))
    deltas = np.array([0.05, 0.1, 0.2, 0.3])
    m = harmonics_for_deltas(deltas, Omega, gamma)
    omega = R.recoil_shifted_harmonic(m, Omega, gamma)
    delta = omega / gamma
    exact = T * delta * R.quantum_synchrotron_rate(delta, chi, gamma)

    def one_turn(k_turns):
        k = int(round(k_turns * per_turn))
        if k + 1 > traj.n_samples:
            raise SystemExit(f"record holds {traj.n_samples} samples, "
                             f"need {k + 1} for {k_turns} turns")
        sl = slice(0, k + 1)
        return Trajectory(time=traj.time[sl], position=traj.position[sl],
                          momentum=traj.momentum[sl], energy=traj.energy[sl])

    dirs, wts = R.orbit_direction_grid(gamma)
    margin = np.array([float(sampling_margin(
        one_turn(1).time, one_turn(1).beta(), dirs, np.array([w]),
        np.array([w * gamma / (gamma - w)]))[0]) for w in omega])

    print(f"\n{NX}x{NY} cells, {per_turn} steps/turn, comparing at the "
          f"recoil-shifted harmonics")
    print(f"{'kernel':>6} {'delta':>6} {'m':>5} {'margin':>7} {'BK dE/dw':>12} "
          f"{'exact':>12} {'ratio':>8}")
    spectra = {}
    for kernel in ("dot", "trace"):
        spec = one_turn(1)
        bk = BKIntegrator(spec, Parameters(epsilon=gamma, kernel=kernel),
                          checks="ignore")
        dE = bk.d2_energy_batch(omega, dirs) @ wts
        spectra[kernel] = dE
        for i in range(delta.size):
            print(f"{kernel:>6} {delta[i]:6.3f} {m[i]:5d} {margin[i]:7.3f} "
                  f"{dE[i]:12.5e} {exact[i]:12.5e} {dE[i] / exact[i]:8.4f}")

    # fixed-epsilon cross-check: |u| is constant here, so it must agree
    fixed = Trajectory(time=one_turn(1).time, position=one_turn(1).position,
                       momentum=one_turn(1).momentum)
    dE_fixed = BKIntegrator(fixed, Parameters(epsilon=gamma, kernel="dot"),
                            checks="ignore").d2_energy_batch(omega, dirs) @ wts
    print(f"local vs fixed-epsilon (same record): "
          f"{np.array2string(dE_fixed / spectra['dot'], precision=6)}")

    # inter-turn coherence on a PIC record: line-centre density ~ n^2
    if len(turns) > 1:
        n_turns = int(max(turns))
        wide = one_turn(n_turns)
        dE_n = BKIntegrator(wide, Parameters(epsilon=gamma, kernel="dot"),
                            checks="ignore").d2_energy_batch(omega, dirs) @ wts
        print(f"{n_turns} turns / 1 turn at the same harmonics (expect "
              f"{n_turns ** 2}): {np.array2string(dE_n / spectra['dot'], precision=3)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--periods", type=float, default=10.0,
                    help="gyro-periods to record (10 gives L/tau_f ~ 125)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/bk_testbed.npz"))
    ap.add_argument("--compare", action="store_true",
                    help="also run the end-to-end spectrum comparison")
    args = ap.parse_args()

    sim, ele, nsteps, meta = build_simulation(args.periods)

    rec = TrajectoryRecorder(ele, args.out, interval=1, max_particles=1)
    sim.run(nsteps=nsteps, callbacks=[rec])
    print(f"recorded -> {rec.write()}")

    # load what was actually written: np.savez would have appended ".npz" to a
    # suffix-less --out, so args.out is not necessarily the file
    d = np.load(rec.path)
    n = int(np.atleast_1d(d["n_samples"])[0])
    u = d["u"][0][:n]
    p = np.sqrt((u ** 2).sum(axis=1))
    print(f"Nt={n}  w={d['w'][0]:.6g}  "
          f"t_end-t_start={(d['t'][0][n - 1] - d['t'][0][0]) / TIME_C:.1f} natural")
    # a uniform static B does no work, so |u| must be constant to round-off;
    # this is the check that the record really is a recoilless background
    print(f"|u|: {p.min():.6f} .. {p.max():.6f}  "
          f"(fluctuation {(p.max() - p.min()) / p.mean():.2e}, expect round-off level)")
    print(f"t strictly increasing: {bool(np.all(np.diff(d['t'][0][:n]) > 0))}")

    if args.compare:
        compare_spectrum(rec.path, meta)


if __name__ == "__main__":
    main()
