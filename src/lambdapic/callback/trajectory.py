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

**The fields at the particle (``fields=True``).**  The Baier--Katkov spectrum
itself needs only ``(t, x, u)``, but the *adiabatic* (LCFA) reference it is
meant to be compared with needs ``chi(t)``, and chi needs the fields:
``chi = gamma|E_perp + v x B| / E_cr``.  With ``fields=True`` the recorder also
writes the ``E`` and ``B`` the particle saw, in **SI** (V/m, T), so the
comparison can be built on the same trajectory.  Two things to know:

* These are the co-located values the pusher already interpolated into
  ``ex_part`` etc. (the same arrays the PIC's own LCFA pipeline reads), not a
  fresh interpolation at the recorded position.  They sit at the half-step
  position ``x - u*dt/2``, which is where a Yee leapfrog defines the fields --
  a half-step offset, not an error, but it is one half-step.
* A sample taken at ``itime == 0`` carries **NaN**: the pusher has not
  interpolated yet, so there is nothing to report.  ``t``/``x``/``u`` of that
  sample are fine, and ``n_samples`` is unaffected (the gap check looks at
  ``t`` and ``x``); a consumer of the fields drops sample 0.
* Reading them costs nothing per step; *storing* them costs twice the ``x``
  payload (three components each for ``E`` and ``B``).  Off by default, because
  a record without them is still valid input for everything except the LCFA
  comparison.

Known limitation: :class:`~lambdapic.callback.utils.MovingWindow` deletes the
particles it sweeps past, so a moving-window run has no full-length history for
a particle it has overtaken -- and a particle identified as interesting at the
*end* of a run may not have existed early on (measured on a production LWFA run:
the hottest particles at the end were unfindable for the first 857 of 1070
steps, which the gap check below reports).  Record with the window off, or pick
particles that exist throughout.

There is deliberately **no** ``window=`` compensation option: ``MovingWindow``
only relabels patches (``p.x0``, ``p.xaxis``, ``p.fields.x0/xaxis``) and never
touches particle coordinates, which are already in the laboratory frame.  Adding
``total_shift`` back would inject a spurious displacement that grows in steps --
equivalent to superimposing a fake velocity on the record and destroying the
phase ``n.(r2 - r1)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .callback import Callback
from ..core.species import Species

__all__ = ["TrajectoryRecorder"]


class TrajectoryRecorder(Callback):
    """Follow selected particles and write their history to ``.npz``.

    Writes ``t``, ``x``, ``u`` always, and ``e``/``b`` (the fields at the
    particle) when ``fields=True``.

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
    allow_recoil:
        Bypass the non-radiating guard.  The resulting record is *not* valid
        input for the Baier--Katkov module; only for diagnostics.
    fields:
        Also record the ``E`` and ``B`` fields at the particle (SI: V/m, T), as
        the ``e`` and ``b`` arrays of the output.  Needed to build the
        adiabatic (LCFA) reference on the same trajectory -- see the module
        docstring for the half-step caveat and the storage cost.  Off by
        default: the Baier--Katkov integral itself needs only ``(t, x, u)``.
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
        allow_recoil: bool = False,
        fields: bool = False,
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
        self.allow_recoil = allow_recoil
        self.fields = bool(fields)

        self._tracked: Optional[np.ndarray] = None      # uint64, identical on all ranks
        self._t: list[np.ndarray] = []                  # (K,) SI seconds per sample
        self._x: list[np.ndarray] = []                  # (K, 3) SI metres
        self._u: list[np.ndarray] = []                  # (K, 3) dimensionless
        self._e: list[np.ndarray] = []                  # (K, 3) V/m, fields=True only
        self._b: list[np.ndarray] = []                  # (K, 3) tesla
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

        x = np.full((self._tracked.size, 3), np.nan)
        u = np.full((self._tracked.size, 3), np.nan)
        e = np.full((self._tracked.size, 3), np.nan) if self.fields else None
        b = np.full((self._tracked.size, 3), np.nan) if self.fields else None
        self._collect(sim, x, u, e, b)

        # ``t`` is written only where this rank actually holds the particle.  It
        # used to be filled with ``sim.time`` unconditionally, which made the
        # merge below a no-op: it looks for ranks whose ``t`` is NaN to fill in
        # from, and a finite ``t`` everywhere meant every rank looked like an
        # owner, so the rank that really had the sample was discarded (measured
        # under mpirun -n 2 tracking one particle owned by each rank:
        # n_samples = [6, 0], the second particle's x all NaN).
        found = np.isfinite(x).all(axis=1)
        self._t.append(np.where(found, sim.time, np.nan))
        self._x.append(x)
        self._u.append(u)
        if self.fields:
            if int(sim.itime) == 0:
                # at itime = 0 the pusher has not interpolated the fields yet,
                # so ``ex_part`` still holds its initialisation.  Report NaN --
                # "not available" -- rather than a zero field that was never
                # measured.  From the second recorded sample on, the arrays hold
                # the previous step's interpolation (one half-step behind the
                # recorded position; see the module docstring).
                e[:] = np.nan
                b[:] = np.nan
            self._e.append(e)
            self._b.append(b)
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

    def _collect(self, sim, x: np.ndarray, u: np.ndarray, e=None, b=None) -> None:
        """Fill ``x``/``u`` (and optionally ``e``/``b``) for the tracked ids.

        Coordinates are taken as they are: the PIC stores them in the laboratory
        frame, and no field configuration shifts them (see the module docstring
        on ``MovingWindow``).
        """
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
                x[k] = (part.x[j], part.y[j], part.z[j])
                u[k] = (part.ux[j], part.uy[j], part.uz[j])
                if e is not None:
                    # the co-located fields the pusher interpolated this step
                    # (the same arrays the PIC's LCFA pipeline reads)
                    e[k] = (part.ex_part[j], part.ey_part[j], part.ez_part[j])
                    b[k] = (part.bx_part[j], part.by_part[j], part.bz_part[j])

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
        names = ["x", "u"] + (["e", "b"] if self.fields else [])
        sources = [self._x, self._u] + ([self._e, self._b] if self.fields else [])
        vectors = [np.transpose(np.stack(v), (1, 0, 2)) for v in sources]  # (K, Nt, 3)

        if sim is not None and sim.mpi.comm.Get_size() > 1:
            t, vectors = self._merge(sim, t, vectors)
        if sim is not None and sim.mpi.rank != 0:
            return None

        n_samples = self._valid_prefix(t, vectors[0])
        w = np.full(self._tracked.size, np.nan) if self._w is None else self._w
        arrays = dict(id=self._tracked, n_samples=n_samples, t=t, w=w,
                      samples=np.asarray(self._itimes, dtype=np.int64))
        arrays.update(zip(names, vectors))
        np.savez(self.path, **arrays)
        return str(self.path)

    def _merge(self, sim, t, vectors):
        """Combine per-rank histories: exactly one rank holds each sample.

        ``t`` is NaN where this rank does not hold the particle (see
        :meth:`_call`), so it is both the ownership marker and the mask applied
        to every per-vector array; the vectors are ``(K, Nt, 3)`` and share it.
        """
        comm = sim.mpi.comm
        gathered = comm.gather((t, *vectors), root=0)
        if sim.mpi.rank != 0:
            return t, vectors
        out = list(gathered[0])
        for row in gathered[1:]:
            fill = ~np.isfinite(out[0]) & np.isfinite(row[0])
            out[0] = np.where(fill, row[0], out[0])
            fill3 = fill[:, :, None]
            for j in range(1, len(out)):
                out[j] = np.where(fill3, row[j], out[j])
        return out[0], out[1:]

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
