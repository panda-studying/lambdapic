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

# -- palette, identical to validation_plot.py (light mode reference palette) --
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GOOD = "#0ca30c"
BAD = "#d03b3b"

#: one colour per turn count, ordered light -> dark so the coherence build-up
#: reads as a progression
TURN_COLOURS = ("#c9d8ee", "#9dbde0", "#5f95cf", BLUE, "#174e8f")


def _strip_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _panel_title(ax, text):
    ax.set_title(text, color=INK, fontsize=11.5, loc="left", pad=10)


def _guide(ax, x, y, label, colour=MUTED):
    ax.plot(x, y, color=colour, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(-4, 6),
                textcoords="offset points", color=colour, fontsize=9,
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

    # Labels use plain Unicode (ω, Γ, ε, ×, ∝, ²), never mathtext ``$...$``, and
    # that is load-bearing, not style: matplotlib resolves the glyphs of any
    # string containing ``$`` through mathtext, whose 'default' font is the
    # single font picked by ``findfont`` -- the per-glyph fallback chain is only
    # consulted on the non-math path.  On a machine without YaHei/SimHei every
    # hanzi next to a ``$`` would come out as a tofu box (measured: 48 missing
    # glyphs in a five-label sample, 0 with the same labels in Unicode).
    # ``font.family`` must also itself be the *list*: ``"sans-serif"`` plus a
    # ``font.sans-serif`` list does not chain (measured: 24 vs 0).
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback",
                                   "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

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

    fig = plt.figure(figsize=(15.0, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.97, top=0.90, bottom=0.07)

    # ---- panel 1: the lines emerge ----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _strip_axes(ax)
    ref1 = rows[0]["lcfa_one_turn"]
    for n, colour in zip(turns, TURN_COLOURS[:len(turns)]):
        row = next(r for r in rows if r["turns"] == n)
        u = (row["omega"] - row["centre"]) / row["spacing"]
        ax.plot(u, row["dE_domega"], color=colour, linewidth=1.8,
                marker="o" if n == turns[-1] else "none",
                markersize=3, markeredgecolor=SURFACE, markeredgewidth=0.8,
                zorder=3, label=f"{n} 圈")
    ax.axhline(ref1, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("LCFA 连续谱（1 圈）", xy=(0.02, ref1), xycoords=("axes fraction", "data"),
                xytext=(4, 5), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("(ω − ωm) / 线间距")
    ax.set_ylabel("dE/dω（自然单位）")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2,
              ncol=2)
    _panel_title(ax, "1 · 谱线随圈数浮现（线中心密度 ∝ n²）")

    # ---- panel 2: n**2 law -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _strip_axes(ax)
    ax.plot(n_arr, r2, linestyle="none", marker="o", markersize=7, color=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
            label="线中心密度 / (T dP/dω)")
    _guide(ax, n_arr, n_arr ** 2, "斜率 2（预期 n²）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("圈数 n")
    ax.set_ylabel("线中心密度 / LCFA 单圈密度")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _panel_title(ax, f"2 · 相干累积：n² 律（实测斜率 {_fit_exponent(n_arr, r2):.3f}）")

    # ---- panel 3: 1/n width ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    _strip_axes(ax)
    ax.plot(n_arr, width, linestyle="none", marker="o", markersize=7,
            color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=4, label="半高全宽 / 线间距")
    _guide(ax, n_arr,
           0.444 * (gamma / (gamma - rows[0]["centre"])) / n_arr,
           "斜率 −1（=0.444(ε/ε')/n）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("圈数 n")
    ax.set_ylabel("FWHM / 线间距")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _panel_title(ax, f"3 · 谱线随圈数变窄（实测斜率 {_fit_exponent(n_arr, width):.3f}）")

    # ---- panel 4: the LCFA discriminator -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _strip_axes(ax)
    ax.plot(n_arr, gam, linestyle="none", marker="o", markersize=7, color=AQUA,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
            label="Γ = 线中心密度 /（n × LCFA 单圈密度）")
    _guide(ax, n_arr, n_arr, "Γ = n")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("圈数 n")
    ax.set_ylabel("Γ")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _panel_title(ax, f"4 · LCFA 判别量 Γ（实测斜率 {_fit_exponent(n_arr, gam):.3f}）")

    # ---- panel 5: line energy ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _strip_axes(ax)
    ax.plot(n_arr, e_ratio, linestyle="none", marker="o", markersize=7,
            color="#8a5cd6", markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=4, label="线能量 /（LCFA 单圈密度 × 间距）")
    _guide(ax, n_arr, n_arr, "斜率 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("圈数 n")
    ax.set_ylabel("线能量比")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _panel_title(ax, f"5 · 线能量 ∝ n（实测斜率 {_fit_exponent(n_arr, e_ratio):.3f}）")

    # ---- panel 6: verdict ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    _panel_title(ax, "参数与判据")
    slope2 = _fit_exponent(n_arr, r2)
    slope_w = _fit_exponent(n_arr, width)
    slope_g = _fit_exponent(n_arr, gam)
    lines = [
        f"γ = {gamma:g},  χ = {chi:g},  δ = {delta:g}",
        f"ρ = {rows[0]['rho']:.1f},  m = {rows[0]['m']}",
        f"ωm = {rows[0]['centre']:.6f},  线间距 = {rows[0]['spacing']:.4e}",
        f"ω 窗口 = ±{opts['n_omega_span']} 间距 × {opts['n_omega']} 点",
        f"Nt = {opts['sampling_factor']} m·n，nθ = {opts['n_theta']}",
        "",
        "角收敛（线中心，n = %d）：" % turns[-1],
        f"  nθ = {opts['n_theta']} → {coarse:.6e}",
        f"  加倍      → {fine:.6e}",
        f"  相对变化   {rel:.2e}",
        "",
        f"实测斜率：  n² 律 {slope2:.3f}（预期 2）",
        f"          线宽      {slope_w:.3f}（预期 −1）",
        f"          Γ      {slope_g:.3f}（预期 1）",
    ]
    for i, text in enumerate(lines):
        ax.text(0.0, 0.99 - 0.061 * i, text, transform=ax.transAxes,
                color=INK2 if text else SURFACE, fontsize=8.8, va="top")
    verdict = (abs(slope2 - 2) < 0.15 and abs(slope_w + 1) < 0.15
               and abs(slope_g - 1) < 0.15)
    ax.text(0.0, 0.02, "✓ 三个标度同时成立" if verdict else "✗ 标度不符，先解释再谈 LCFA",
            transform=ax.transAxes, color=GOOD if verdict else BAD,
            fontsize=10.5, va="bottom")

    fig.suptitle(
        "Test B · 圈间相干：LCFA 给不出的离散谐波结构"
        f"（匀速圆周，γ = {gamma:g}, χ = {chi:g}, δ = {delta:g}）",
        color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.945,
             "BK 双时间积分保留相位，n 圈相干叠加给出线中心密度 ∝ n²；"
             "LCFA 是局域非相干的速率积分，只随观测时间线性增长。",
             color=MUTED, fontsize=9.5, ha="left")

    if save_path is None:
        import pathlib
        save_path = pathlib.Path(__file__).with_name("coherence_summary.png")
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {save_path}")
    return str(save_path)


if __name__ == "__main__":
    main()
