"""Testbed: record a *real* PIC trajectory for the Baier--Katkov module.

The banded-truncation verdict in ``REVIEW_AND_ROADMAP.md`` §8.2 is decisive only
on a closed orbit; the aperiodic case -- which is what a PIC actually produces --
could not be settled because the synthetic record used there was barely four
turns long.  This script builds a cheap but adequate testbed: **one electron
gyrating in a static magnetic field**, recorded through the real pipeline
(Boris pusher, field interpolation, SI units) over many gyro-periods.

Why a magnetic field and not a laser: the module's sampling criterion is
``dt * (omega * eps/eps') * max_n|1 - n.v| < pi``, and the ``max`` is taken over
directions -- it is set by the *backward* direction (about 2), not by the
forward ``1/gamma`` lobe.  A laser-driven record evaluated near its spectral
peak needs 10^2-10^3 steps per laser period, while a CFL-limited ``dx =
lambda/32`` grid delivers only 48.  The gyro-orbit is under-resolved by a much
smaller factor -- but note it is *not* resolved at ``omega_c``: with the
defaults the margin is 3.17 there, so only ``omega <~ 0.6 omega_c`` is usable
(the driver prints the margin; trust it, not this paragraph).

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--periods", type=float, default=10.0,
                    help="gyro-periods to record (10 gives L/tau_f ~ 125)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/bk_testbed.npz"))
    args = ap.parse_args()

    sim, ele, nsteps, meta = build_simulation(args.periods)

    rec = TrajectoryRecorder(ele, args.out, interval=1, max_particles=1)
    sim.run(nsteps=nsteps, callbacks=[rec])
    path = rec.write()
    print(f"recorded -> {path}")

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


if __name__ == "__main__":
    main()
