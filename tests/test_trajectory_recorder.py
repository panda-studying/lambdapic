"""Tests for the PIC trajectory recorder and the SI unit adapter.

The recorder feeds the Baier--Katkov module, whose input must be a
non-radiating classical background sampled at self-consistent times.  The
properties pinned here are the ones that fail *silently* if they regress:
the stage the callback registers at (a pusher stage would disable the unified
pusher), the fact that the tracked set is chosen globally, the NaN-padded
fixed-shape file layout, and the recoil guard.
"""

import numpy as np
import pytest

from lambdapic import Simulation
from lambdapic.callback.trajectory import TrajectoryRecorder
from lambdapic.core.species import Electron
from lambdapic.core.qed.baier_katkov.trajectory import as_trajectory_si
from lambdapic.core.qed.baier_katkov.types import Parameters, Trajectory

pytestmark = pytest.mark.filterwarnings("ignore")


def _sim(nx=64, ny=64, seed=42):
    """A small 2D box with a handful of weight-1 electrons in a disc."""
    dx = 0.8e-6 / 20
    sim = Simulation(nx=nx, ny=ny, dx=dx, dy=dx, npatch_x=2, npatch_y=2,
                     dt_cfl=0.95, random_seed=seed)
    x0, y0 = 0.35 * nx * dx, 0.5 * ny * dx

    def disc(x, y):
        # w = density*dx*dy/ppc, so density = 1/(dx*dy) gives w = 1, i.e. real
        # single electrons rather than macroparticles.
        return (1.0 / (dx * dx)
                if (x - x0) ** 2 + (y - y0) ** 2 <= (2.5 * dx) ** 2 else 0.0)

    ele = Electron(density=disc, ppc=1)
    sim.add_species([ele])
    sim.initialize()
    return sim, ele


class _DummyPusherStage:
    """A no-op callback registered at a pusher stage (disables the fast path)."""

    stage = "_push_position_2"
    interval = 1

    def __call__(self, sim):
        return None


def test_recorder_registers_at_a_non_pusher_stage():
    """`stage` must be set on the *instance*.

    ``Callback`` only annotates ``stage``; without the instance attribute the
    registration silently falls back to ``"end"``, where ``sim.time`` is a full
    step behind the particle state.  And the stage must not be one of the five
    pusher stages, which switch the unified pusher off for every species.
    """
    rec = TrajectoryRecorder(_sim()[1], "/tmp/unused.npz")
    assert rec.stage == "start"
    assert rec.stage not in ("_push_position_1", "_interpolator", "_qed",
                             "_push_momentum", "_push_position_2")
    assert rec.DEFAULT_STAGE == "start"


def test_recorder_rejects_a_float_interval():
    """A float interval is a floating-point time comparison across ranks."""
    with pytest.raises(TypeError, match="int"):
        TrajectoryRecorder(_sim()[1], "/tmp/unused.npz", interval=0.5)


def test_recorder_refuses_a_radiating_record(tmp_path):
    """Feeding a recoil-laden record to the BK module would double count.

    Building a genuinely radiating species needs a paired ``Photon`` species
    (``radiation.py`` asserts on it at initialisation), so the flag is set
    directly instead -- the guard reads exactly that flag.
    """
    sim, ele = _sim()
    rec = TrajectoryRecorder(ele, tmp_path / "x.npz", interval=1)
    ele.radiation = "photons"
    with pytest.raises(ValueError, match="radiation"):
        sim.run(nsteps=2, callbacks=[rec])


def test_recorder_writes_a_usable_trajectory(tmp_path):
    """End-to-end: record, round-trip through the file, adapt to natural units."""
    sim, ele = _sim()
    out = tmp_path / "traj.npz"
    rec = TrajectoryRecorder(ele, out, interval=2, max_particles=3)
    sim.run(nsteps=40, callbacks=[rec])
    assert rec.write() == str(out)

    with np.load(out) as d:
        assert set(d.files) >= {"id", "n_samples", "t", "x", "u", "w", "samples"}
        k = d["id"].size
        assert k == 3 and d["x"].shape == (k, 20, 3) and d["u"].shape == (k, 20, 3)
        assert np.allclose(d["w"], 1.0, rtol=1e-12)   # w = dens*dx*dy/ppc, dens = 1/dx²
        assert np.all(d["n_samples"] == 20)          # interval=2 over 40 steps
        assert np.all(np.diff(d["samples"]) == 2)
        t, x, u, ns = d["t"], d["x"], d["u"], d["n_samples"]

    # every particle must be finite over its whole valid prefix
    for i in range(k):
        n = int(ns[i])
        assert np.all(np.isfinite(t[i][:n])) and np.all(np.isfinite(x[i][:n]))
        assert np.all(np.diff(t[i][:n]) > 0)

    traj = as_trajectory_si(t[0], x[0], u=u[0], n_samples=int(ns[0]))
    assert isinstance(traj, Trajectory)
    assert traj.n_samples == 20
    # the adapter must agree with the module's own on-shell energy fallback
    assert np.allclose(traj.local_energy(), traj.gamma())
    assert np.all(traj.gamma() >= 1.0)


def test_recorder_tracks_the_same_ids_as_the_run_proceeds(tmp_path):
    """Ids are stable and the record has no gaps, even across patch migration."""
    sim, ele = _sim()
    out = tmp_path / "traj.npz"
    rec = TrajectoryRecorder(ele, out, interval=1, max_particles=2)
    sim.run(nsteps=120, callbacks=[rec])
    rec.write()
    with np.load(out) as d:
        # a gap would show as a NaN *inside* the valid prefix, which
        # _valid_prefix() raises on; reaching here means there was none.
        # nsteps=120 fires 'start' at itime 0..119, so 120 samples, not 121.
        assert np.all(d["n_samples"] == 120)
        assert len(set(d["id"].tolist())) == 2


def test_pusher_stage_callback_changes_only_the_speed(tmp_path):
    """Registering at a pusher stage must not change the trajectory.

    The recorder's whole reason for using ``"start"`` is that a pusher-stage
    callback switches the unified pusher off; this pins that the two paths
    agree, rather than assuming it.
    """
    fast = tmp_path / "fast.npz"
    slow = tmp_path / "slow.npz"

    sim_a, ele_a = _sim()
    rec_a = TrajectoryRecorder(ele_a, fast, interval=1, max_particles=2)
    sim_a.run(nsteps=60, callbacks=[rec_a])
    rec_a.write()

    sim_b, ele_b = _sim()
    rec_b = TrajectoryRecorder(ele_b, slow, interval=1, max_particles=2)
    sim_b.run(nsteps=60, callbacks=[rec_b, _DummyPusherStage()])
    rec_b.write()

    with np.load(fast) as a, np.load(slow) as b:
        assert np.array_equal(a["id"], b["id"])
        assert np.allclose(a["t"], b["t"], rtol=0, atol=0)
        assert np.allclose(a["x"], b["x"], rtol=1e-12, atol=1e-18)
        assert np.allclose(a["u"], b["u"], rtol=1e-12, atol=1e-15)


def test_adapter_energy_argument_switches_to_local_energy_mode():
    """The integrator switches on ``Trajectory.energy``, not on ``local_energy()``.

    ``Trajectory.local_energy()`` is never consulted by the integrator, so an
    accelerating record passed without ``energy`` is silently evaluated at a
    fixed incident energy.  Pin that passing it actually changes the result.
    """
    from lambdapic.core.qed.baier_katkov import integrator as bki

    gamma = 10.0
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho, turns, n = 200.0, 1.0, 600
    Omega = beta / rho
    t = np.linspace(0.0, turns * 2.0 * np.pi / Omega, n)
    phi = Omega * t
    # a 5 -> 15 ramp along the record, built through the SI adapter
    gam = np.linspace(5.0, 15.0, n)
    r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])
    u = (gam * beta)[:, None] * np.column_stack(
        [-np.sin(phi), np.cos(phi), np.zeros_like(t)])

    from lambdapic.core.qed.baier_katkov import units as U
    t_si = U.natural_time_to_si(t)
    x_si = U.natural_length_to_si(r)

    fixed = as_trajectory_si(t_si, x_si, u=u)
    local = as_trajectory_si(t_si, x_si, u=u, energy=gam)
    assert fixed.energy is None and local.energy is not None

    pars = Parameters(epsilon=10.0)
    a = bki.compute_spectrum(fixed, pars, np.array([0.5]), theta_max=np.pi,
                             n_theta=8, n_phi=1, axis=[0, 0, 1], checks="ignore")
    b = bki.compute_spectrum(local, pars, np.array([0.5]), theta_max=np.pi,
                             n_theta=8, n_phi=1, axis=[0, 0, 1], checks="ignore")
    assert b.metadata["local_energy"] is True
    assert a.metadata["local_energy"] is False
    assert float(a.dE_domega[0]) != pytest.approx(float(b.dE_domega[0]), rel=1e-3)


def test_recorder_returns_the_path_it_actually_wrote(tmp_path):
    """``np.savez`` appends ``.npz``; ``write()`` must not report the other path."""
    sim, ele = _sim()
    rec = TrajectoryRecorder(ele, tmp_path / "no_suffix", interval=1)
    sim.run(nsteps=4, callbacks=[rec])
    written = rec.write()
    assert written == str(rec.path)
    assert rec.path.exists() and rec.path.suffix == ".npz"
    np.load(rec.path)          # must be loadable at the returned path
