"""Figure for Test B (inter-turn coherence), separate from the V1-V8 summary.

Renders ``coherence_summary.png`` next to this script.  This is deliberately a
*separate* figure from ``validation_summary.png``: Test B is a physics
measurement, not a member of the V1-V8 suite, and the two must be free to move
independently (see :mod:`.coherence`).

Panels
------
1. the spectrum around one harmonic for several turn counts, with the LCFA
   continuum level -- the discrete structure emerging as coherence builds up;
2. line-centre density / (one-turn LCFA density) vs ``n``, log-log, against the
   expected ``n**2``;
3. line full width / line spacing vs ``n``, against the expected ``1/n``;
4. the discriminator: ``Gamma(n)`` = line-centre density / (``n`` x one-turn
   LCFA density), against ``Gamma = n``.  ``Gamma`` is by how much the LCFA
   continuum under-reports the coherent line density;
5. a text panel with the parameters, the angular-convergence verification and
   the measured exponents.

Run with::

    python -m lambdapic.core.qed.baier_katkov.coherence_plot
"""

from __future__ import annotations

import numpy as np

from . import coherence as _c

# -- palette, identical to validation_plot.py (plain white journal style) -----
INK = "#111111"
INK2 = "#333333"
MUTED = "#7a7a7a"
BASE = "#c8c8c8"
BLUE = "#1f5fa8"
ORANGE = "#d1632a"
AQUA = "#2e8b6e"
GOOD = "#1a7f37"
BAD = "#b3261e"

#: one colour per turn count, ordered light -> dark so the coherence build-up
#: reads as a progression
TURN_COLOURS = ("#b8cbe4", "#8aaed6", "#5588c4", BLUE, "#123f73")


def set_publication_style() -> None:
    """Use a clean, mainstream publication plotting style.

    ``font.family`` is given as the concrete font *list* rather than the usual
    ``font.family="serif"`` plus ``font.serif=[...]``: the latter does **not**
    chain per glyph on this matplotlib, so a CJK label whose first font
    (Times New Roman / DejaVu Serif) lacks the hanzi is left missing rather than
    falling through to a CJK face (measured: 28 missing glyphs with the
    ``font.serif`` form, 0 with the ``font.family`` list).  The serif CJK faces
    come before the sans fallback so hanzi match the Times-like body text.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # last entry is a symbols fallback: the serif faces and the CJK faces
        # all lack the check marks (U+2713/U+2717) used in the verdict panel
        "font.family": ["Times New Roman", "DejaVu Serif",
                        "AR PL UMing CN", "AR PL SungtiL GB",
                        "Droid Sans Fallback", "DejaVu Sans"],
        "axes.unicode_minus": True,
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _journal(ax):
    ax.set_facecolor("white")


def _strip_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=12)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(False)


def _panel_title(ax, text):
    ax.set_title(text, color=INK, fontsize=12, loc="left", pad=10)


def _stack(*lines) -> str:
    """Join non-empty label lines with a newline.

    A single label *line* must not carry both mathtext and CJK -- see
    :func:`_check_labels` -- so mixed labels are written one line at a time
    here, with the formula (mathtext, raw string) on its own line.
    """
    return "\n".join(line for line in lines if line)


def _check_labels(fig) -> None:
    """Fail if any label line mixes mathtext (``$``) with CJK.

    Matplotlib parses a line containing ``$`` through mathtext, and the non-math
    segments of such a line are drawn with the *math* font, which has no CJK
    coverage -- every hanzi then becomes a dummy box.  Measured on this machine
    (matplotlib 3.10.8, fonts DejaVu Sans + Droid Sans Fallback): 5 missing
    glyphs for ``"$\\Gamma$ = 线中心密度"``, 0 for the same label with the hanzi
    on their own line.  Hence the rule: one line carries either mathtext or CJK,
    never both.  This guard turns a silent tofu box into a loud error.
    """
    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        for line in str(text.get_text()).split("\n"):
            if "$" in line and any(ord(c) > 0x2E7F for c in line):
                raise ValueError(
                    "mathtext and CJK cannot share a line (the math font has no "
                    f"CJK glyphs) -- split the label across lines: {line!r}"
                )


def _guide(ax, x, y, label, colour=MUTED):
    ax.plot(x, y, color=colour, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(-4, 6),
                textcoords="offset points", color=colour, fontsize=12,
                ha="right", va="bottom")


def _fit_exponent(x, y):
    """Least-squares slope of ``log y`` vs ``log x`` (``nan`` if degenerate)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main(save_path=None, **kw) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Formulas are mathtext with the STIX fontset (LaTeX look, no TeX
    # installation on this machine).  The load-bearing constraint is that a label
    # *line* must carry either mathtext or CJK, never both: matplotlib parses a
    # line containing ``$`` through mathtext, whose non-math segments use the
    # math font, and that font has no CJK glyphs -- every hanzi on such a line
    # becomes a dummy box (measured: 5 missing glyphs, 0 once the hanzi move to
    # their own line).  All side-by-side ``$``/hanzi labels below are therefore
    # split with :func:`_stack`, and :func:`_check_labels` enforces the rule on
    # the finished figure.  ``font.family`` must also itself be the *list*:
    # ``"serif"`` plus a ``font.serif`` list does not chain.
    set_publication_style()

    opts = dict(_c.DEFAULTS)
    opts.update(kw)
    gamma, chi = opts["gamma"], opts["chi"]
    delta = opts["deltas"][0]
    turns = opts["turns"]

    print("running Test B ...")
    rows = _c.scan_turns(**kw)
    n_arr = np.array([r["turns"] for r in rows], dtype=float)
    r2 = np.array([r["ratio_n2"] for r in rows])
    width = np.array([r["fwhm_over_spacing"] for r in rows])
    gam = np.array([r["gamma_lcfa"] for r in rows])
    e_ratio = np.array([r["line_energy_ratio"] for r in rows])

    coarse, fine, rel = _c.verify_angular_convergence(
        gamma, chi, delta, int(turns[-1]), **_c._probe_options(kw))

    # taller figure and a lower gridspec top: the 12 pt suptitle + subtitle +
    # the two-line panel titles need more headroom than the 9.5 pt version did
    # (measured overlap at figsize 20x14 / top 0.915)
    fig = plt.figure(figsize=(20.0, 15.0))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28,
                          left=0.055, right=0.985, top=0.885, bottom=0.055)

    # ---- panel 1: the lines emerge ----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _strip_axes(ax)
    ref1 = rows[0]["lcfa_one_turn"]
    for n, colour in zip(turns, TURN_COLOURS[:len(turns)]):
        row = next(r for r in rows if r["turns"] == n)
        u = (row["omega"] - row["centre"]) / row["spacing"]
        ax.plot(u, row["dE_domega"], color=colour, linewidth=1.8,
                marker="o" if n == turns[-1] else "none",
                markersize=3, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3, label=rf"$n = {n}$")
    ax.axhline(ref1, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("LCFA 连续谱（1 圈）", xy=(0.02, ref1), xycoords=("axes fraction", "data"),
                xytext=(4, 5), textcoords="offset points", color=MUTED, fontsize=12)
    ax.set_yscale("log")
    ax.set_xlabel(_stack(r"$(\omega - \omega_m)/\Delta\omega$", "线间距归一"))
    ax.set_ylabel(_stack(r"$dE/d\omega$", "自然单位"))
    ax.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=INK2,
              ncol=2)
    _panel_title(ax, _stack("1 · 谱线随圈数浮现（线中心密度）", r"$\propto n^2$"))

    # ---- panel 2: n**2 law -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _strip_axes(ax)
    ax.plot(n_arr, r2, linestyle="none", marker="o", markersize=7, color=BLUE,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4,
            label=_stack("线中心密度 /", r"$T\,dP/d\omega$"))
    _guide(ax, n_arr, n_arr ** 2, _stack("斜率 2（预期）", r"$n^2$"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_stack("圈数", r"$n$"))
    ax.set_ylabel("线中心密度 / LCFA 单圈密度")
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack(
        f"2 · 相干累积：实测斜率 {_fit_exponent(n_arr, r2):.3f}", r"$n^2$"))

    # ---- panel 3: 1/n width ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    _strip_axes(ax)
    ax.plot(n_arr, width, linestyle="none", marker="o", markersize=7,
            color=ORANGE, markeredgecolor="white", markeredgewidth=1.5,
            zorder=4, label="半高全宽 / 线间距")
    _guide(ax, n_arr,
           0.444 * (gamma / (gamma - rows[0]["centre"])) / n_arr,
           _stack("斜率 −1", r"$= 0.444\,(\varepsilon/\varepsilon')/n$"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_stack("圈数", r"$n$"))
    ax.set_ylabel(_stack(r"$\mathrm{FWHM}/\,\Delta\omega$", "线间距归一"))
    ax.legend(loc="lower left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack(
        f"3 · 谱线随圈数变窄：实测斜率 {_fit_exponent(n_arr, width):.3f}",
        r"$\propto 1/n$"))

    # ---- panel 4: the LCFA discriminator -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _strip_axes(ax)
    ax.plot(n_arr, gam, linestyle="none", marker="o", markersize=7, color=AQUA,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4,
            label=_stack(r"$\Gamma$", "线中心密度 /（n × LCFA 单圈密度）"))
    _guide(ax, n_arr, n_arr, r"$\Gamma = n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_stack("圈数", r"$n$"))
    ax.set_ylabel(r"$\Gamma$")
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack(
        f"4 · LCFA 判别量：实测斜率 {_fit_exponent(n_arr, gam):.3f}", r"$\Gamma$"))

    # ---- panel 5: line energy ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _strip_axes(ax)
    ax.plot(n_arr, e_ratio, linestyle="none", marker="o", markersize=7,
            color="#8a5cd6", markeredgecolor="white", markeredgewidth=1.5,
            zorder=4, label="线能量 /（LCFA 单圈密度 × 间距）")
    _guide(ax, n_arr, n_arr, "斜率 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_stack("圈数", r"$n$"))
    ax.set_ylabel("线能量比")
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack(
        f"5 · 线能量：实测斜率 {_fit_exponent(n_arr, e_ratio):.3f}", r"$\propto n$"))

    # ---- panel 6: verdict ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    _journal(ax)
    ax.axis("off")
    _panel_title(ax, "参数与判据")
    slope2 = _fit_exponent(n_arr, r2)
    slope_w = _fit_exponent(n_arr, width)
    slope_g = _fit_exponent(n_arr, gam)
    lines = [
        rf"$\gamma = {gamma:g}$,  $\chi = {chi:g}$,  $\delta = {delta:g}$",
        rf"$\rho = {rows[0]['rho']:.1f}$,  $m = {rows[0]['m']}$",
        f"ωm = {rows[0]['centre']:.6f},  线间距 = {rows[0]['spacing']:.4e}",
        f"ω 窗口 = ±{opts['n_omega_span']} 间距 × {opts['n_omega']} 点",
        f"Nt = {opts['sampling_factor']} m·n，nθ = {opts['n_theta']}",
        "",
        "角收敛（线中心，n = %d）：" % turns[-1],
        rf"  $n_\theta = {opts['n_theta']}$ → {coarse:.6e}",
        f"  加倍      → {fine:.6e}",
        f"  相对变化   {rel:.2e}",
        "",
        f"实测斜率：  n² 律 {slope2:.3f}（预期 2）",
        f"          线宽      {slope_w:.3f}（预期 −1）",
        f"          Γ      {slope_g:.3f}（预期 1）",
    ]
    for i, text in enumerate(lines):
        ax.text(0.0, 0.99 - 0.061 * i, text, transform=ax.transAxes,
                color=INK2 if text else "white", fontsize=12.8, va="top")
    verdict = (abs(slope2 - 2) < 0.15 and abs(slope_w + 1) < 0.15
               and abs(slope_g - 1) < 0.15)
    ax.text(0.0, 0.02, "✓ 三个标度同时成立" if verdict else "✗ 标度不符，先解释再谈 LCFA",
            transform=ax.transAxes, color=GOOD if verdict else BAD,
            fontsize=12, va="bottom")

    # One-line suptitle, parameters in Unicode: a two-line suptitle (or a second
    # subtitle line) collides with the panel titles -- the top margin does not
    # have room for it.  The LaTeX rendering of these parameters is in the
    # "参数与判据" panel.
    fig.suptitle(
        "Test B · 圈间相干：LCFA 给不出的离散谐波结构"
        f"（匀速圆周，γ = {gamma:g}, χ = {chi:g}, δ = {delta:g}）",
        color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.945,
             "BK 双时间积分保留相位，n 圈相干叠加给出线中心密度 ∝ n²；"
             "LCFA 是局域非相干的速率积分，只随观测时间线性增长。",
             color=MUTED, fontsize=12, ha="left")

    if save_path is None:
        import pathlib
        save_path = pathlib.Path(__file__).with_name("coherence_summary.png")
    _check_labels(fig)
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {save_path}")
    return str(save_path)


if __name__ == "__main__":
    main()
