"""Cross-rank merge of the trajectory recorder (run under ``mpirun -n 2``).

The tracked particle set is chosen globally, so every rank records *all* tracked
ids but only has data for the ones it owns; only rank 0 writes the file.  The
merge must therefore combine ranks, and that path is only exercised when the
tracked particles are owned by *different* ranks -- automatic selection takes
the globally smallest ids, and ids pack the creating rank in their high bits, so
it always picks rank 0 and never touches the merge.
"""
import os
import warnings

os.environ.setdefault("LAMBDAPIC_CHECK_UPDATE", "0")
warnings.simplefilter("ignore")

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as comm

from lambdapic import Simulation
from lambdapic.callback.trajectory import TrajectoryRecorder
from lambdapic.core.species import Electron

OUT = "/tmp/bk_recorder_mpi_test.npz"


@pytest.mark.mpi
@pytest.mark.parametrize("fields", [False, True])
def test_recorder_merges_histories_across_ranks(fields):
    """A particle owned by a non-root rank must survive the merge.

    Before the fix the merge keyed on ``t`` being NaN, but ``t`` was written
    unconditionally on every rank, so the ranks that did *not* own a particle
    looked like owners of a NaN ``x`` and the real data was discarded.  Measured
    with one tracked particle per rank: ``n_samples = [6, 0]``, the second
    particle's ``x`` entirely NaN, and no error -- ``_valid_prefix`` returns 0
    for an all-NaN particle without complaining.
    """
    size = comm.Get_size()
    if size < 2:
        pytest.skip("needs at least 2 ranks")

    dx = 0.8e-6 / 20
    sim = Simulation(nx=64, ny=64, dx=dx, dy=dx, npatch_x=2, npatch_y=2,
                     dt_cfl=0.95, random_seed=42)

    def disc(x, y):
        return (1.0 / (dx * dx)
                if (x - 0.5 * 64 * dx) ** 2 + (y - 0.5 * 64 * dx) ** 2
                <= (6 * dx) ** 2 else 0.0)

    ele = Electron(density=disc, ppc=1)
    sim.add_species([ele])
    sim.initialize()
    B0 = 10.0                                    # uniform, so the record is checkable
    for p in sim.patches:
        p.fields.bz[:, :] = B0

    ispec = ele.ispec
    local = np.concatenate([p.particles[ispec].id[p.particles[ispec].is_alive]
                            for p in sim.patches])
    allids = np.concatenate(comm.allgather(local))
    owner = np.concatenate(comm.allgather(np.full(local.size, comm.Get_rank(),
                                                  dtype=np.int64)))
    # one particle per rank, chosen identically on every rank
    want = np.array([allids[owner == r][0] for r in range(size)], dtype=np.uint64)

    rec = TrajectoryRecorder(ele, OUT, interval=1, ids=want, fields=fields)
    sim.run(nsteps=6, callbacks=[rec])
    rec.write()

    if comm.Get_rank() == 0:
        with np.load(rec.path) as d:
            assert d["n_samples"].tolist() == [6] * size, d["n_samples"]
            assert np.all(np.isfinite(d["x"])), "merged history has NaN"
            assert np.all(np.diff(d["t"], axis=1)[:, :5] > 0.0)
            if fields:
                # the field channel merges with the same ownership mask as x
                assert {"e", "b"} <= set(d.files)
                assert np.all(d["n_samples"] == 6)
                assert np.all(np.isfinite(d["b"][:, 1:, :]))
                assert np.allclose(d["b"][:, 1:, 2], B0, rtol=1e-12)
                assert np.allclose(d["e"][:, 1:, :], 0.0, atol=1e-9)
            else:
                assert not ({"e", "b"} & set(d.files))
