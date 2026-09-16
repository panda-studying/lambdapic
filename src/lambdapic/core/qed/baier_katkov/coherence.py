"""Test B: inter-turn coherence -- the LCFA-missing signal on a closed orbit.

This is **not** part of the V1-V8 validation suite.  V1-V8 live in
:mod:`.validation` and are pinned by ``tests/test_baier_katkov.py``; this module
is a physics *measurement* whose job is to expose a signal that LCFA cannot
represent, and it is deliberately kept separate so that neither the suite nor
its regression tests move when the measurement is tuned.

Physics
-------
LCFA is a *locally constant field* approximation: its rate is a function of the
instantaneous ``chi`` and the emitted spectrum is the incoherent integral
``int dt dW_LCFA/dt domega (chi(t))``.  What it throws away is the **phase**.
BK keeps the full double-time phase ``iint dt1 dt2 N exp(-i Phi)``, so on a
closed orbit the emission from different turns adds *coherently*.

For a strictly periodic orbit at the recoil-shifted harmonic
``omega_m = m Omega / (1 + m Omega/eps)`` the phase difference between turns is
an exact multiple of ``2 pi``, so the ``n``-turn double integral factorises and

    dE/domega (n turns, omega_m) = n**2 * dE/domega (1 turn, omega_m)
                                = n**2 * T * dP/domega,

the second equality being the one-turn/continuum correspondence of
:mod:`.reference` (V8's own statement).  The LCFA continuum for the same
*observation time* scales only linearly, ``n * T * dP/domega``.  Hence

    Gamma(n) == line-centre density / (n * T * dP/domega) == n.

That factor ``n`` is the whole signal: **the line-centre spectral density of a
coherently radiating electron exceeds the LCFA continuum by the number of turns
it has radiated for.**  Equivalently the line *energy* (density x width) scales
as ``n`` -- which is why ``L * dP/domega`` is the wrong target for a multi-turn
record (roadmap sections 3.4 and 4.1).

Why V8 cannot see this
----------------------
V8 uses **one turn** and evaluates *at* the harmonic.  At ``delta = 0.5`` one
turn already has line width ``0.886 Omega`` against a line spacing of
``0.25 Omega``: the lines overlap fourfold and the spectrum is smooth, so there
is no structure to see.  Lines separate only for ``n >> (eps/eps')**2``.  V8 is
the ``n = 1`` endpoint of this measurement.

Reading the result
------------------
Three independent scalings must all come out:

* line-centre density   / (T dP/domega)      ~ n**2   (panel 2)
* line full width       / line spacing       ~ 1/n    (panel 3)
* line energy / (T dP/domega * spacing)      ~ n      (panel 4)
``Gamma(n) = n`` (panel 5) is the LCFA discriminator: it is the factor by which
the LCFA continuum under-reports the coherent line density.

Cost and caveats
----------------
Per frequency the double sum is ``O(N_t**2 * N_dir)`` and ``N_t`` must grow
linearly with the turn count (the Nyquist bound ``N_t > 4 m`` applies to the
*record*, and the record is ``n`` turns long), so a naive scan costs
``sum_n n**2``.  The defaults below are sized to run in single-digit minutes;
``line_center_only=True`` trims the scan to the line centres when only
``Gamma(n)`` is wanted.  Banded truncation (roadmap section 3.5: decided against)
would turn the ``n**2`` into ``n``.

Two honest limitations, both checked by the measurement rather than assumed:

1. ``n`` coherent turns require that **no photon is emitted in between** -- once
   one is, ``eps`` changes and the phase relation for later turns breaks.  The
   ``n**2`` law is therefore an idealised upper bound; its own breakdown is what
   panel 2's straightness (or lack of it) reveals.
2. The angular quadrature error is *common to all* ``n`` (same grid, same
   ``omega_m``), so it cancels in every ratio above.  Absolute values carry it.

Run with::

    python -m lambdapic.core.qed.baier_katkov.coherence            # table
    python -m lambdapic.core.qed.baier_katkov.coherence_plot       # figure

(The figure is written to ``coherence_summary.png`` and never touches
``validation_summary.png``.)
"""

from __future__ import annotations

import warnings

import numpy as np

from . import reference as R
from . import validation as V
from .integrator import (AngularConvergenceWarning, BKIntegrator, cone_directions,
                         compute_spectrum)
from .types import Parameters

__all__ = [
    "DEFAULTS",
    "line_spacing",
    "line_centre_grid",
    "predicted_fwhm_spacings",
    "lcfa_reference",
    "turn_spectrum",
    "verify_angular_convergence",
    "measure_line_width",
    "measure_line_energy",
    "scan_turns",
    "format_table",
]

#: Defaults for the whole measurement.  ``gamma = 5`` rather than 10 keeps the
#: recoil-shifted harmonic index ``m`` down by a factor ``gamma**3`` (245 vs
#: 1990 at ``delta = 0.5``), which is what makes the ``n**2`` cost affordable;
#: the line width in units of the line spacing is ``0.886 (eps/eps')**2 / n``,
#: independent of ``gamma``, so nothing about the signal is lost.
DEFAULTS = dict(
    gamma=5.0,
    chi=0.5,
    deltas=(0.5,),
    turns=(1, 2, 4, 8, 16),
    #: samples per turn, as a multiple of the recoil-shifted harmonic ``m``.
    #: The Nyquist bound is ``N_t > 4 m`` *per record*, i.e. ``4 m n`` here;
    #: 4.4 leaves the sampling margin at ~0.91 instead of sitting on 1.0.
    sampling_factor=4.4,
    #: omega grid: ``+- n_omega_span`` line spacings, ``n_omega`` points.
    #: An odd count puts the exact line centre on the grid.
    n_omega=61,
    n_omega_span=2.5,
    n_theta=32,
    n_phi=1,
    checks="warn",
    backend="auto",
)


# --------------------------------------------------------------------------
# orbit kinematics
# --------------------------------------------------------------------------
def line_spacing(Omega, epsilon, omega_centre):
    """Spacing of the recoil-shifted harmonics, ``Omega (eps'/eps)**2``."""
    return Omega * ((epsilon - omega_centre) / epsilon) ** 2


def predicted_fwhm_spacings(turns, epsilon, epsilon_prime):
    """FWHM of an ``n``-period line, in units of the line spacing.

    **Measured**, not derived: ``FWHM * n / (eps/eps')`` comes out 0.963, 0.910,
    0.893, 0.888 for ``n = 2, 4, 8, 16`` on the defaults, i.e. a constant
    ``0.444 (eps/eps') / n``.  A naive Dirichlet estimate -- the intensity
    ``|sin(n y)/y|**2`` with ``y = pi u (eps'/eps)``, half maximum at
    ``n y = 1.3915`` -- gives ``0.886 (eps/eps') / n``, exactly twice this.  The
    factor of two is not resolved here, so the measured constant is used and no
    theoretical value is claimed.  Either way the *scaling* ``1/n`` is what the
    measurement tests, and that is clean.

    Used only to size the omega window, so the line is carried by a comparable
    number of grid points whatever ``n`` is.
    """
    return 0.444 * (epsilon / epsilon_prime) / turns


def line_centre_grid(m, Omega, epsilon, n_omega=None, span=None,
                     fwhm_spacings=None):
    """Line centre ``omega_m``, spacing, and an odd ``omega`` grid around it.

    ``span`` is the half width of the window in line spacings, capped at
    ``DEFAULTS["n_omega_span"]``.  Passing ``fwhm_spacings`` shrinks the window
    to about four line widths, which keeps roughly eight grid points across the
    line as it narrows with ``n`` -- without it the width measurement saturates
    at the grid step (measured: at ``n = 16`` the FWHM is one grid step wide and
    comes out 39% too large).
    """
    n_omega = DEFAULTS["n_omega"] if n_omega is None else n_omega
    span_max = DEFAULTS["n_omega_span"] if span is None else span
    if n_omega % 2 == 0:
        raise ValueError("n_omega must be odd so the line centre lands on the grid")
    if fwhm_spacings is None:
        span = span_max
    else:
        span = float(np.clip(3.75 * fwhm_spacings, 0.25, span_max))
    centre = float(R.recoil_shifted_harmonic([m], Omega, epsilon)[0])
    spacing = float(line_spacing(Omega, epsilon, centre))
    grid = np.linspace(centre - span * spacing, centre + span * spacing, n_omega)
    return centre, spacing, grid


def lcfa_reference(delta, chi, epsilon, T):
    """One-turn LCFA continuum density ``T dP/domega`` at ``delta``.

    ``dP/domega = delta * dW/(dt ddelta)``; the rate is the *quantum*
    constant-field synchrotron one, which is also what the project LCFA tables
    use (see :mod:`.reference`).
    """
    rate = R.quantum_synchrotron_rate(delta, chi, epsilon)
    return T * delta * rate


# --------------------------------------------------------------------------
# one n-turn spectrum
# --------------------------------------------------------------------------
def turn_spectrum(gamma, chi, delta, turns, **kw):
    """Spectrum of a ``turns``-turn closed circle around its line centre.

    Returns a dict with the ``omega`` grid, ``dE_domega``, the line centre and
    spacing, the one-turn LCFA continuum density, and the integrator metadata
    (so the adequacy guards travel with the numbers).
    """
    opts = dict(DEFAULTS)
    opts.update(kw)

    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    T = 2.0 * np.pi / Omega

    m = int(V.harmonics_for_deltas([delta], Omega, gamma)[0])
    fwhm = predicted_fwhm_spacings(turns, gamma,
                                   gamma - R.recoil_shifted_harmonic([m], Omega, gamma)[0])
    centre, spacing, grid = line_centre_grid(
        m, Omega, gamma, opts["n_omega"], opts["n_omega_span"], fwhm)
    n_samples = int(np.ceil(opts["sampling_factor"] * m * turns))

    traj, _, _ = V.circle_trajectory(gamma=gamma, rho=rho, n_samples=n_samples,
                                     turns=float(turns))
    # The generic angular guard probes by |dE/domega| over the whole window and
    # is therefore dominated by the inter-harmonic residuals.  On a one-turn
    # record those are not small: the Dirichlet kernel of a single period is
    # identically 1, so the line comb leaves no imprint and the window is filled
    # by the within-turn interference pattern, which is a cancellation residual
    # and is *measured* to swing by 140% of the peak under angular refinement
    # while the line centres move by 1e-5.  Silencing that one warning here is
    # legitimate only because :func:`verify_angular_convergence` checks the
    # quantity this module actually reads -- the line-centre density -- and
    # ``main`` prints it.  The sampling and record-length guards still fire.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AngularConvergenceWarning)
        spec = compute_spectrum(
            traj, Parameters(epsilon=gamma, kernel="dot"), grid,
            theta_max=np.pi, n_theta=opts["n_theta"], n_phi=opts["n_phi"],
            axis=np.array([0.0, 0.0, 1.0]), backend=opts["backend"],
            checks=opts["checks"],
        )

    delta_actual = centre / gamma
    ref1 = lcfa_reference(delta_actual, R.circle_chi(gamma, rho), gamma, T)
    i_c = int(np.argmin(np.abs(grid - centre)))
    return dict(
        turns=int(turns), gamma=gamma, chi=chi, rho=rho, Omega=Omega, T=T,
        m=m, delta=delta_actual, omega=grid, dE_domega=spec.dE_domega,
        centre=centre, spacing=spacing, i_centre=i_c,
        lcfa_one_turn=ref1, n_samples=n_samples,
        metadata=spec.metadata,
    )


# --------------------------------------------------------------------------
# line-shape measurements
# --------------------------------------------------------------------------
def verify_angular_convergence(gamma, chi, delta, turns, **kw):
    """Line-centre density at ``n_theta`` and ``2 n_theta``, and their change.

    The check the measurement actually needs, in place of the generic
    :class:`AngularConvergenceWarning` (see the note in
    :func:`turn_spectrum`): the observable here is the spectral density *at*
    ``omega_m``, so that is what must be shown to be grid-converged.  Measured
    on the defaults: ``n_theta = 32`` sits within 3e-5 of ``n_theta = 64``,
    while ``n_theta = 16`` is already 4% low.
    """
    opts = dict(DEFAULTS)
    opts.update(kw)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    rho = gamma ** 2 * beta ** 2 / chi
    Omega = beta / rho
    m = int(V.harmonics_for_deltas([delta], Omega, gamma)[0])
    centre = float(R.recoil_shifted_harmonic([m], Omega, gamma)[0])
    n_samples = int(np.ceil(opts["sampling_factor"] * m * turns))
    traj, _, _ = V.circle_trajectory(gamma=gamma, rho=rho, n_samples=n_samples,
                                     turns=float(turns))
    bk = BKIntegrator(traj, Parameters(epsilon=gamma, kernel="dot"),
                      backend=opts["backend"], checks="ignore")

    def value(n_theta, n_phi):
        dirs, dom = cone_directions([0.0, 0.0, 1.0], np.pi, n_theta, n_phi)
        return float((bk.d2_energy_batch([centre], dirs) @ dom)[0])

    coarse = value(opts["n_theta"], opts["n_phi"])
    fine = value(2 * opts["n_theta"], 2 * opts["n_phi"])
    return coarse, fine, abs(fine - coarse) / abs(coarse)


def measure_line_width(omega, dE, i_centre, min_steps=4.0):
    """Full width at half maximum around ``i_centre`` (``nan`` if unresolved).

    Walks out from the peak on both sides, interpolating linearly between grid
    points; returns ``nan`` when the profile has not fallen to half maximum
    inside the window -- which is what happens at ``n = 1``, where the line is
    wider than the line spacing and there is no line to measure -- or when the
    width is below ``min_steps`` grid steps, where the answer is set by the
    sampling rather than by the physics.
    """
    omega = np.asarray(omega, dtype=np.float64)
    dE = np.asarray(dE, dtype=np.float64)
    n = dE.size
    step = float(np.min(np.diff(omega))) if n > 1 else 0.0
    i0 = int(np.argmax(dE[max(0, i_centre - 2):min(n, i_centre + 3)])) + max(0, i_centre - 2)
    half = 0.5 * dE[i0]

    def cross(step_dir):
        i = i0
        while 0 <= i + step_dir < n:
            j = i + step_dir
            if dE[j] <= half:
                # linear interpolation between dE[i] and dE[j]
                t = (dE[i] - half) / (dE[i] - dE[j]) if dE[i] != dE[j] else 0.0
                return omega[i] + t * (omega[j] - omega[i])
            i = j
        return None

    lo, hi = cross(-1), cross(+1)
    if lo is None or hi is None:
        return float("nan")
    width = float(hi - lo)
    if step > 0.0 and width < min_steps * step:
        return float("nan")
    return width


def measure_line_energy(omega, dE, centre, spacing):
    """Energy in the window ``+- spacing/2`` around the line centre.

    For ``n = 1`` the neighbouring lines overlap into this window, so the value
    is an upper bound there; from ``n >= 4`` the line is resolved and it is the
    line energy proper.
    """
    sel = np.abs(omega - centre) <= 0.5 * spacing
    return float(np.trapezoid(np.asarray(dE)[sel], np.asarray(omega)[sel]))


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------
def scan_turns(gamma=None, chi=None, deltas=None, turns=None, **kw):
    """Run the measurement over the turn list; returns a list of row dicts.

    Each row carries the raw spectrum plus the three derived scalings:
    ``ratio_n2`` (line-centre density / one-turn LCFA density, expect
    ``n**2``), ``gamma_lcfa`` (the same divided by ``n``, expect ``n`` -- the
    LCFA discriminator), ``fwhm_over_spacing`` (expect ``1/n``) and
    ``line_energy_ratio`` (expect ``n``).
    """
    opts = dict(DEFAULTS)
    opts.update(kw)
    gamma = opts["gamma"] if gamma is None else gamma
    chi = opts["chi"] if chi is None else chi
    deltas = opts["deltas"] if deltas is None else deltas
    turns = opts["turns"] if turns is None else turns

    rows = []
    for delta in deltas:
        for n in turns:
            spec = turn_spectrum(gamma, chi, delta, n, **kw)
            dE = spec["dE_domega"]
            peak = float(dE[spec["i_centre"]])
            density_ratio = peak / spec["lcfa_one_turn"]
            fwhm = measure_line_width(spec["omega"], dE, spec["i_centre"])
            row = dict(spec)
            row.update(
                peak_density=peak,
                ratio_n2=density_ratio,
                gamma_lcfa=density_ratio / n,
                fwhm=fwhm,
                fwhm_over_spacing=fwhm / spec["spacing"],
                line_energy=measure_line_energy(
                    spec["omega"], dE, spec["centre"], spec["spacing"]),
                line_energy_ratio=(
                    measure_line_energy(spec["omega"], dE, spec["centre"],
                                        spec["spacing"])
                    / (spec["lcfa_one_turn"] * spec["spacing"])),
            )
            rows.append(row)
            print(f"  n={n:3d}  N_t={spec['n_samples']:7d}  m={spec['m']:5d}  "
                  f"line-centre/LCFA={density_ratio:10.4f}  (n^2={n * n:6d})  "
                  f"Gamma={row['gamma_lcfa']:8.4f}  "
                  f"FWHM/D={row['fwhm_over_spacing']:.4f}")
    return rows


def format_table(rows):
    """Plain-text table of the scalings, with the expected laws as columns."""
    lines = [
        "Test B: inter-turn coherence on a closed orbit",
        "  prediction: line-centre density = n^2 * (T dP/domega); Gamma = n;"
        " FWHM ~ 0.444 (eps/eps') / n (measured constant)",
        "",
        f"  {'n':>4} {'N_t':>8} {'peak/LCFA':>12} {'n^2':>9} {'Gamma':>9} "
        f"{'n':>5} {'FWHM/D':>9} {'1/n':>7} {'E_line/LCFA':>12} {'n':>5}",
        "  " + "-" * 92,
    ]
    for r in rows:
        lines.append(
            f"  {r['turns']:>4} {r['n_samples']:>8} {r['ratio_n2']:>12.4f} "
            f"{r['turns'] ** 2:>9} {r['gamma_lcfa']:>9.4f} {r['turns']:>5} "
            f"{r['fwhm_over_spacing']:>9.4f} {1.0 / r['turns']:>7.3f} "
            f"{r['line_energy_ratio']:>12.4f} {r['turns']:>5}")
    return "\n".join(lines)


def _probe_options(kw):
    """``kw`` minus the keys ``verify_angular_convergence`` takes positionally."""
    return {k: v for k, v in kw.items()
            if k not in ("gamma", "chi", "deltas", "turns")}


def main(**kw):
    """Run the scan and print the table.  Returns the row list."""
    opts = dict(DEFAULTS)
    opts.update(kw)
    print(f"Test B: gamma={opts['gamma']} chi={opts['chi']} "
          f"deltas={opts['deltas']} turns={opts['turns']} "
          f"n_omega={opts['n_omega']} n_theta={opts['n_theta']}")
    coarse, fine, rel = verify_angular_convergence(
        opts["gamma"], opts["chi"], opts["deltas"][0], int(opts["turns"][-1]),
        **_probe_options(kw))
    print(f"  angular convergence at the line centre (n={opts['turns'][-1]}): "
          f"n_theta={opts['n_theta']} -> {coarse:.6e}, "
          f"doubled -> {fine:.6e},  change {rel:.2e}")
    rows = scan_turns(**kw)
    print()
    print(format_table(rows))
    print()
    print("Gamma > 1 means the LCFA continuum under-reports the coherent line")
    print("density; the ratio should track n.  For the figure run")
    print("  python -m lambdapic.core.qed.baier_katkov.coherence_plot")
    return rows


if __name__ == "__main__":
    main()
