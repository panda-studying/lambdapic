"""Record selected particles' trajectories for offline radiation analysis.

The Baier--Katkov double-time integral needs a **classical, non-radiating
background** trajectory sampled on a time grid.  This callback produces exactly
that: it follows a fixed set of particles and writes ``(t, x, u)`` histories to
an ``.npz`` file that
:func:`lambdapic.core.qed.baier_katkov.trajectory.as_trajectory_si` consumes
directly.

Four things here are deliberate and load-bearing:

**Stage ``"start"``.**  Registering a callback at any of the pusher stages
(``_push_position_1``, ``_interpolator``, ``_qed``, ``_push_momentum``,
``_push_position_2``) disables the unified pusher fast path for every species,
and that decision is taken before the loop from the registration table -- so it
costs even when the interval means the callback never fires.  ``"start"`` is not
a pusher stage, and at that point ``t``, ``x`` and ``u`` are mutually consistent
(``sim.time`` is only incremented at the end of the step; see
``Simulation.run``).  ``Callback`` only *annotates* ``stage``, so the instance
attribute must also be set in ``__init__`` or the registration silently falls
back to ``"end"``, where the time label is a full step behind the particle
state.

**A globally identical particle set.**  The tracked ids are chosen once by
gathering every rank's alive ids and taking the globally smallest ``max_particles``
of them.  Selecting locally would break as soon as a tracked particle migrates:
the old rank stops recording it and the new one never started, leaving a *hole in
the middle* of the history.  Times would still be strictly increasing, the
``Trajectory`` validation would still pass, and the Baier--Katkov integrator
would linearly interpolate a straight segment that never happened.  Note the
guarantee this buys is that *every rank agrees on the set*; it is **not**
rank-count independence -- ids pack the creating rank in their high bits, so
which physical particle ends up selected depends on the rank-to-patch
assignment (and on load balancing).  Two runs with different rank counts can
therefore pick different particles.

**Fixed-shape, NaN-padded output.**  A gap shows up as a NaN rather than being
hidden by ragged concatenation, and ``n_samples`` distinguishes "died at sample
500" from "still alive at the end".  ``as_trajectory_si`` slices on
``n_samples`` and raises on a NaN inside the valid prefix.

**Non-radiating background.**  A record that already contains LCFA recoil must
not be fed to the Baier--Katkov module -- the module applies the quantum recoil
itself, so it would be double counted.  The guard below checks the species'
``radiation`` flag and the collision operator, and can only be bypassed
explicitly.

Known limitation: :class:`~lambdapic.callback.utils.MovingWindow` shifts patches
periodically and accumulates the displacement in ``total_shift``.  Pass the
window callback as ``window=`` and that displacement is added back to the
recorded ``x``.  **Whether this is actually needed is unverified**:
``MovingWindow`` does read and write particle ``x`` (``callback/utils.py``,
the ``x_list`` blocks), so a coordinate shift is plausible, but it was not
established by a test -- and if the particle coordinates are already in the lab
frame, the ``window=`` compensation would inject an offset that grows with the
window displacement.  **Check against a moving-window run before trusting it.**
A static-box run does not need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .callback import Callback
from ..core.species import Species

__all__ = ["TrajectoryRecorder"]


class TrajectoryRecorder(Callback):
    """Follow selected particles and write their ``(t, x, u)`` history to ``.npz``.

    Parameters
    ----------
    species:
        The species to record.  Must be a non-radiating one (see the module
        docstring) unless ``allow_recoil`` is set.
    path:
        Output file.  Written by rank 0.
    interval:
        Sampling interval in **steps** (integer only).  A float interval is
        rejected: it is evaluated as ``(sim.time % interval) < sim.dt``, a
        floating-point comparison that need not agree across ranks, and the
        base class takes a ``Barrier`` inside the triggered branch -- a
        disagreement would hang the run.
    max_particles:
        How many particles to follow.  The set is the globally smallest ids, so
        it does not depend on the rank count.
    ids:
        Explicit ``uint64`` ids to follow instead of the automatic selection.
    select_after:
        Wait this many steps before choosing the tracked set -- useful when the
        interesting particles only appear once the drive has arrived.
    window:
        An optional ``MovingWindow`` instance whose ``total_shift`` is added
        back to the recorded ``x`` (see the module docstring).
    allow_recoil:
        Bypass the non-radiating guard.  The resulting record is *not* valid
        input for the Baier--Katkov module; only for diagnostics.
    """

    DEFAULT_STAGE: str = "start"

    def __init__(
        self,
        species: Species,
        path: Union[str, Path],
        interval: int = 1,
        max_particles: int = 1,
        ids: Optional[Sequence[int]] = None,
        select_after: int = 0,
        window: Optional[Callback] = None,
        allow_recoil: bool = False,
    ) -> None:
        if not isinstance(interval, int) or isinstance(interval, bool):
            raise TypeError(
                "interval must be an int (steps); float intervals use a "
                "floating-point time comparison that can desynchronise ranks"
            )
        if interval < 1:
            raise ValueError("interval must be >= 1")
        if max_particles < 1:
            raise ValueError("max_particles must be >= 1")

        # Callback annotates `stage` but does not set it -- without this line the
        # registration silently defaults to "end".
        self.stage = self.DEFAULT_STAGE
        self.interval = interval

        self.species = species
        self.path = Path(path)
        if self.path.suffix != ".npz":
            # np.savez appends ".npz" to a suffix-less path, so write() would
            # otherwise return a path that does not exist
            self.path = self.path.with_suffix(".npz")
        self.max_particles = int(max_particles)
        self._explicit_ids = None if ids is None else np.asarray(ids, dtype=np.uint64)
        self.select_after = int(select_after)
        self.window = window
        self.allow_recoil = allow_recoil

        self._tracked: Optional[np.ndarray] = None      # uint64, identical on all ranks
        self._t: list[np.ndarray] = []                  # (K,) SI seconds per sample
        self._x: list[np.ndarray] = []                  # (K, 3) SI metres
        self._u: list[np.ndarray] = []                  # (K, 3) dimensionless
        self._itimes: list[int] = []                    # sim.itime of each sample
        self._w: Optional[np.ndarray] = None            # (K,)
        self._sim = None                               # last Simulation seen

    # -- sampling ----------------------------------------------------------
    def __call__(self, sim) -> None:
        """Sample every ``interval`` steps.

        Deliberately overrides ``Callback.__call__``: the base class logs and
        wraps every trigger in a spinner/timer, which at ``interval=1`` over a
        ten-thousand-step record is tens of thousands of log lines.  The intent
        is the same as :class:`~lambdapic.callback.utils.MovingWindow`, which
        overrides ``__call__`` for the same reason.
        """
        if sim.itime % self.interval != 0:
            return
        self._call(sim)

    def _call(self, sim) -> None:
        self._sim = sim
        self._check_background(sim)

        if self._tracked is None:
            if sim.itime < self.select_after:
                return
            self._select(sim)

        shift = 0.0 if self.window is None else float(self.window.total_shift or 0.0)
        x = np.full((self._tracked.size, 3), np.nan)
        u = np.full((self._tracked.size, 3), np.nan)
        self._collect(sim, x, u, shift)

        self._t.append(np.full(self._tracked.size, sim.time))
        self._x.append(x)
        self._u.append(u)
        self._itimes.append(int(sim.itime))

    # -- helpers -----------------------------------------------------------
    def _check_background(self, sim) -> None:
        """Refuse to record a record that already contains recoil."""
        if self.allow_recoil:
            return
        radiation = getattr(self.species, "radiation", None)
        if radiation is not None:
            raise ValueError(
                f"species '{self.species.name}' has radiation={radiation!r}: the "
                "record would already contain quantum recoil and the Baier--Katkov "
                "module would count it twice. Record a non-radiating background "
                "(radiation=None), or pass allow_recoil=True for diagnostics."
            )
        if getattr(sim, "collision", None) is not None:
            raise ValueError(
                "the collision operator is enabled, so the recorded momenta are "
                "not the classical background motion. Disable collisions, or pass "
                "allow_recoil=True for diagnostics."
            )

    def _local_ids(self, sim) -> np.ndarray:
        """Alive ids on this rank (uint64), checked for duplicates."""
        ids = [
            p.particles[self.species.ispec].id[p.particles[self.species.ispec].is_alive]
            for p in sim.patches
        ]
        local = np.concatenate(ids) if ids else np.empty(0, dtype=np.uint64)
        if np.unique(local).size != local.size:
            raise RuntimeError(
                "duplicate alive particle ids on one rank -- that is a bug in the "
                "particle exchange, not in this recorder"
            )
        return local

    def _select(self, sim) -> None:
        """Choose the tracked set globally, so it is identical on every rank."""
        if self._explicit_ids is not None:
            self._tracked = self._explicit_ids
        else:
            comm = sim.mpi.comm
            local = self._local_ids(sim)
            gathered = comm.allgather(local)
            everything = (np.concatenate(gathered) if gathered
                          else np.empty(0, dtype=np.uint64))
            if everything.size < self.max_particles:
                raise ValueError(
                    f"only {everything.size} alive particles, cannot track "
                    f"{self.max_particles}"
                )
            self._tracked = np.sort(everything)[: self.max_particles]
        self._w = self._weights(sim)

    def _weights(self, sim) -> np.ndarray:
        """Macro-particle weights of the tracked ids (NaN if not local here)."""
        w = np.full(self._tracked.size, np.nan)
        for p in sim.patches:
            part = p.particles[self.species.ispec]
            idx = np.nonzero(part.is_alive)[0]
            if not idx.size:
                continue
            for k, want in enumerate(self._tracked):
                hit = np.nonzero(part.id[idx] == want)[0]
                if hit.size:
                    w[k] = float(part.w[idx[hit[0]]])
        if sim.mpi.comm.Get_size() > 1:
            gathered = sim.mpi.comm.allgather(w)
            w = np.nanmax(np.vstack(gathered), axis=0)
        return w

    def _collect(self, sim, x: np.ndarray, u: np.ndarray, shift: float) -> None:
        """Fill ``x``/``u`` rows for the tracked ids found on this rank."""
        for p in sim.patches:
            # re-fetch every time: extend()/prune() reallocate these arrays
            part = p.particles[self.species.ispec]
            idx = np.nonzero(part.is_alive)[0]
            if not idx.size:
                continue
            local_id = part.id[idx]
            for k, want in enumerate(self._tracked):
                hit = np.nonzero(local_id == want)[0]
                if not hit.size:
                    continue
                j = idx[hit[0]]
                x[k] = (part.x[j] + shift, part.y[j], part.z[j])
                u[k] = (part.ux[j], part.uy[j], part.uz[j])

    # -- output ------------------------------------------------------------
    def write(self, sim=None) -> Optional[str]:
        """Merge ranks and write the ``.npz``; safe to call more than once.

        Called by the driver after ``Simulation.run`` returns, because
        ``"final"`` is *not* guaranteed to fire: a restart dump and a
        ``stop_callback`` both return from inside the step loop, skipping it.
        ``sim`` defaults to the last simulation seen by :meth:`_call`.
        """
        sim = self._sim if sim is None else sim
        if self._tracked is None or not self._t:
            return None

        t = np.stack(self._t).T                                   # (K, Nt)
        x = np.transpose(np.stack(self._x), (1, 0, 2))            # (K, Nt, 3)
        u = np.transpose(np.stack(self._u), (1, 0, 2))

        if sim is not None and sim.mpi.comm.Get_size() > 1:
            t, x, u = self._merge(sim, t, x, u)
        if sim is not None and sim.mpi.rank != 0:
            return None

        n_samples = self._valid_prefix(t, x)
        w = np.full(self._tracked.size, np.nan) if self._w is None else self._w
        np.savez(
            self.path,
            id=self._tracked,
            n_samples=n_samples,
            t=t,
            x=x,
            u=u,
            w=w,
            samples=np.asarray(self._itimes, dtype=np.int64),
        )
        return str(self.path)

    def _merge(self, sim, t, x, u):
        """Combine per-rank histories: exactly one rank holds each sample."""
        comm = sim.mpi.comm
        gathered = comm.gather((t, x, u), root=0)
        if sim.mpi.rank != 0:
            return t, x, u
        out_t, out_x, out_u = gathered[0]
        for rt, rx, ru in gathered[1:]:
            fill = ~np.isfinite(out_t) & np.isfinite(rt)
            out_t = np.where(fill, rt, out_t)
            fill3 = fill[:, :, None]
            out_x = np.where(fill3, rx, out_x)
            out_u = np.where(fill3, ru, out_u)
        return out_t, out_x, out_u

    @staticmethod
    def _valid_prefix(t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Per-particle number of usable leading samples, rejecting interior gaps."""
        finite = np.isfinite(t) & np.isfinite(x).all(axis=2)     # (K, Nt)
        n_samples = np.zeros(finite.shape[0], dtype=np.int64)
        for k in range(finite.shape[0]):
            good = np.nonzero(finite[k])[0]
            if good.size == 0:
                continue
            last = int(good[-1])
            if good.size != last + 1:
                raise RuntimeError(
                    f"particle {k} has a gap: {good.size} samples are present but "
                    f"they run up to index {last}. Either it was not findable for "
                    "part of the run (a moving window deletes particles it "
                    "sweeps past, and a particle selected from the *end* of a run "
                    "may not have existed early on), or it migrated to a rank "
                    "-- measured on a real LWFA run with MovingWindow: the "
                    "hottest particles at the end were absent for the first 857 "
                    "of 1070 steps."
                )
            n_samples[k] = last + 1
        return n_samples
