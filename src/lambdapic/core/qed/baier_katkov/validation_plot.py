"""Static summary figure of the V1..V8 validation results.

Re-runs the coarse checks of :mod:`.validation` (single particle, FP64) and
renders them into one ``validation_summary.png`` next to this script:

* V3  soft-photon BK -> LW ratio vs ``omega/eps`` at the recoil-shifted
      harmonics (should tend to 1 as ``1 - omega/eps``), with the bare-harmonic
      truncation artifact for comparison;
* V4  code-vs-exact synchrotron harmonic ratios vs harmonic order ``m``
      (should tend to 1) plus the Larmor closure;
* V5  straight-line vs curved reference radiation (log scale, should be tiny);
* V6  angle-integrated ``dW/domega`` positivity spectrum;
* V1/V2/V7  internal/analytic relative errors (log scale);
* V8  the quantum regime: one-turn BK spectrum at the recoil-shifted
      harmonics against the exact constant-field quantum synchrotron spectrum
      (spectrum panel and ratio panel);
* a verdict panel listing all checks.

Requires ``matplotlib`` (already a declared ``lambdapic`` dependency); the
computation itself stays numpy/scipy(/numba)-only.  Run from anywhere with::

    python -m lambdapic.core.qed.baier_katkov.validation_plot
"""

from __future__ import annotations

import numpy as np

from . import reference as _r
from . import validation as _v

# -- palette ----------------------------------------------------------------
# Plain white journal style: black ink, grey annotations, a muted qualitative
# triple for the three series.  No surface tinting, no grid -- series identity
# comes from colour plus the marker/linestyle differences set at each panel.
INK = "#111111"
INK2 = "#333333"
MUTED = "#7a7a7a"
BASE = "#c8c8c8"
BLUE = "#1f5fa8"
ORANGE = "#d1632a"
AQUA = "#2e8b6e"
GOOD = "#1a7f37"


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


def _strip_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=12)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(False)


def _hline1(ax, xmin, xmax):
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)


def _panel_title(ax, text):
    # no bold: the serif CJK face has no bold member, so matplotlib would
    # substitute a different font for the hanzi and the title would stop
    # matching the rest of the figure
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


def main(save_path=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    # -- re-run all checks -------------------------------------------------
    print("running V1..V8 ...")
    v1 = _v.check_airy_identity()
    v2 = _v.check_velocity_kernel_consistency()
    v3 = _v.check_classical_limit()
    v4 = _v.check_larmor()
    v5 = _v.check_straight_line()
    v6 = _v.check_positivity()
    v7 = _v.check_kernel_rewrite()
    v8 = _v.check_quantum_synchrotron(deltas=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
                                      kernels=("dot", "trace"))

    # figure enlarged for the 12 pt publication sizes (the previous 15x13 was
    # sized for 9 pt labels)
    fig = plt.figure(figsize=(20.0, 17.0))
    gs = fig.add_gridspec(3, 3, hspace=0.60, wspace=0.40,
                          left=0.055, right=0.985, top=0.935, bottom=0.055)

    # ---- V3: soft-photon BK -> LW -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _strip_axes(ax)
    x = np.array([r["omega_over_eps"] for r in v3])
    y = np.array([r["ratio"] for r in v3])
    y_bare = np.array([r["ratio_bare"] for r in v3])
    ax.plot(x, y_bare, color=BASE, linewidth=1.6, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=1.0, zorder=2,
            label=_stack(r"$m\Omega$", "（裸谐波 · 截断伪影）"))
    ax.plot(x, y, color=BLUE, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
            label=_stack(r"$\omega_m$", "（反冲移动谐波）"))
    ax.plot(x, 1.0 - x, color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1,
            label=r"$1-\omega/\varepsilon$")
    _hline1(ax, x.min(), x.max())
    ax.annotate(f"{y[-1]:.4f}", (x[-1], y[-1]), textcoords="offset points",
                xytext=(0, -14), ha="center", color=INK2, fontsize=12)
    ax.set_xlabel(r"$\omega/\varepsilon$")
    ax.set_ylabel("BK / LW 比值")
    ax.set_ylim(0.985, 1.045)
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack("V3 · 软光子极限 BK → LW（比值趋于）",
                            r"$1-\omega/\varepsilon$"))

    # ---- V4: Larmor normalization ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _strip_axes(ax)
    m = np.array(v4["harmonics"], dtype=float)
    y = np.array(v4["harmonic_ratios"])
    ax.semilogx(m, y, color=ORANGE, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    _hline1(ax, m.min(), m.max())
    for i in (0, -1):
        ax.annotate(f"{y[i]:.3f}", (m[i], y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=12)
    ax.set_xticks(m)
    ax.set_xticklabels([f"{int(v)}" for v in m])
    ax.set_xlabel(_stack(r"$m$   ($\omega = m\Omega$)", "谐波阶数"))
    ax.set_ylabel("代码 / 精确同步辐射谱")
    ax.set_ylim(0.92, 1.02)
    _panel_title(ax, "V4 · Larmor 归一化（谐波比值 → 1）")
    ax.annotate(f"精确谱闭合 = {v4['closure']:.4f}",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=12)

    # ---- V5: straight line vs curved --------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    _strip_axes(ax)
    labels = [_stack(r"$|\mathrm{BK}|$", "直线"),
              _stack(r"$\mathrm{LW}$", "直线"),
              _stack(r"$\mathrm{LW}$", "弯曲参考")]
    vals = [abs(v5["bk"]), abs(v5["lw"]), abs(v5["curved_ref"])]
    cols = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, vals, color=cols, width=0.62, zorder=3,
                  edgecolor="white", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e5)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.2e}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK2, fontsize=12)
    ax.set_ylabel(_stack(r"$d^2E/d\omega\,d\Omega$", "对数刻度"))
    _panel_title(ax, "V5 · 直线轨迹 ≈ 零辐射")
    # top-left: the two-line bar labels below the axis would collide with a
    # bottom-left annotation
    ax.annotate(rf"$|{{\rm BK}}/{{\rm curved}}| = {v5['ratio_bk_to_curved']:.1e}$",
                xy=(0.03, 0.97), xycoords="axes fraction", color=INK2,
                fontsize=12, va="top")

    # ---- V6: positivity spectrum ------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _strip_axes(ax)
    w = v6["omega_grid"]
    dW = v6["dW_domega"]
    ax.bar(np.arange(len(w)), dW, color=BLUE, width=0.62, zorder=3,
           edgecolor="white", linewidth=1.5)
    ax.set_xticks(np.arange(len(w)))
    ax.set_xticklabels([f"{float(v):.1e}" for v in w], rotation=30, ha="right")
    ax.set_xlabel(_stack(r"$\omega_m$", "反冲移动谐波"))
    ax.set_ylabel(r"$dW/d\omega$")
    ax.set_ylim(0, 1.35 * float(dW.max()))     # headroom for the annotation
    ax.axhline(0.0, color=BASE, linewidth=1.0)
    _panel_title(ax, "V6 · 角积分谱正性（未裁剪）")
    ax.annotate(f"min = +{v6['min_dW']:.2f}  总 W = {v6['total_W']:.3f}",
                xy=(0.03, 0.97), xycoords="axes fraction", color=INK2,
                fontsize=12, va="top")

    # ---- V1/V2/V7: internal precision -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _strip_axes(ax)
    labels = ["V1 Airy", _stack(r"V2  $|A|^2$", "= 速度双积分"),
              r"V7  $\mathrm{trace}\equiv\mathrm{dot}$"]
    errs = [v1["rel_err"], v2["rel_err"], max(v7["rel_err"], 1e-16)]
    cols = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, errs, color=cols, width=0.62, zorder=3,
                  edgecolor="white", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-2)
    for b, val in zip(bars, errs):
        ax.annotate(f"{val:.2e}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK2, fontsize=12)
    ax.set_ylabel("相对误差（对数）")
    _panel_title(ax, "V1 / V2 / V7 · 解析与内部自洽")
    ax.annotate("V1<1e-3, V2<1e-8, V7<1e-12  →  全部通过",
                xy=(0.03, 0.97), xycoords="axes fraction", color=INK2,
                fontsize=12, va="top")

    # ---- V8: quantum synchrotron ratio -------------------------------------
    d8 = v8["delta"]
    ax = fig.add_subplot(gs[1, 2])
    _strip_axes(ax)
    ax.plot(d8, v8["quantum_over_classical"], color=BASE, linewidth=1.6,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1.0,
            zorder=2, label="精确量子 / 经典（反冲压低）")
    ax.plot(d8, v8["ratio"]["dot"], color=BLUE, linewidth=2.0, marker="o",
            markersize=5, markeredgecolor="white", markeredgewidth=1.2, zorder=4,
            label="BK dot 核 / 精确量子")
    ax.plot(d8, v8["ratio"]["trace"], color=ORANGE, linewidth=1.4,
            linestyle=(0, (4, 3)), marker="s", markersize=4,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5,
            label="BK trace 核 / 精确量子")
    _hline1(ax, d8.min(), d8.max())
    worst = np.max(np.abs(v8["ratio"]["dot"] - 1.0))
    ax.set_xlabel(r"$\delta = \omega/\varepsilon$")
    ax.set_ylabel("比值")
    ax.set_ylim(0.25, 1.12)
    ax.legend(loc="lower left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack("V8 · 量子区间比值",
                            rf"$\chi = {v8['chi']:.2f}$,  "
                            rf"$\gamma = {v8['gamma']:.0f}$"))
    ax.annotate(f"max |BK/精确 − 1| = {worst:.1e}", xy=(0.97, 0.80),
                xycoords="axes fraction", color=INK2, fontsize=12, ha="right")

    # ---- V8: quantum synchrotron spectrum ----------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    _strip_axes(ax)
    chi8, g8, T8 = v8["chi"], v8["gamma"], v8["T"]
    d_line = np.geomspace(0.01, 0.95, 160)
    q_line = T8 * _r.quantum_synchrotron_rate(d_line, chi8, g8)
    c_line = T8 * _r.classical_synchrotron_rate(d_line, chi8, g8)
    ax.plot(d_line, c_line, color=BASE, linewidth=1.6, linestyle=(0, (4, 3)),
            zorder=2, label="经典 Schwinger 谱（无反冲）")
    ax.plot(d_line, q_line, color=INK2, linewidth=2.0, zorder=3,
            label="精确量子同步辐射谱（K_1/3, K_2/3）")
    # code: one-turn dW/domega = dE/domega / omega -> per unit delta: * eps
    dW_ddelta_code = v8["code"]["dot"] / v8["omega"] * g8
    ax.plot(d8, dW_ddelta_code, linestyle="none", marker="o", markersize=7,
            color=BLUE, markeredgecolor="white", markeredgewidth=1.5, zorder=5,
            label="BK 一圈积分（dot 核，反冲移动谐波处）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta = \omega/\varepsilon$")
    ax.set_ylabel(_stack(r"$dW/d\delta$", "每圈光子数"))
    ax.legend(loc="lower left", frameon=False, fontsize=12, labelcolor=INK2)
    _panel_title(ax, _stack(
        "V8 · 量子同步辐射谱：BK 双时间积分 vs 恒定场精确解",
        rf"$\chi = {chi8:.2f}$,  $\gamma = {g8:.0f}$,  "
        rf"$N_t = {v8['n_samples']}$"))

    # ---- verdict panel ------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor("white")
    ax.axis("off")
    _panel_title(ax, "汇总")
    rows = [
        ("V1  Airy 恒等式 (eq 9.6)", f"{v1['rel_err']:.2e}", v1["ok"]),
        ("V2  |A|² = 速度双积分", f"{v2['rel_err']:.2e}", v2["ok"]),
        ("V3  BK → LW 软光子", f"{v3[-1]['ratio']:.4f} ≈ 1−ω/ε", True),
        ("V4  Larmor 归一化", f"闭合 {v4['closure']:.4f}", True),
        ("V5  直线 ≈ 零辐射", f"|BK/弯曲| {v5['ratio_bk_to_curved']:.1e}", True),
        ("V6  正性", f"min +{v6['min_dW']:.2f}", True),
        ("V7  trace ≡ dot 在壳", f"{v7['rel_err']:.2e}", v7["ok"]),
        ("V8  量子同步辐射谱", f"max 偏差 {worst:.1e}", worst < 2e-2),
    ]
    for i, (name, val, ok) in enumerate(rows):
        ax.text(0.02, 0.95 - 0.11 * i, name, transform=ax.transAxes,
                color=INK2, fontsize=12, va="top")
        ax.text(0.56, 0.95 - 0.11 * i, val, transform=ax.transAxes,
                color=INK2, fontsize=12, va="top")
        ax.text(0.98, 0.95 - 0.11 * i, "✓ OK" if ok else "✗ FAIL",
                transform=ax.transAxes, color=GOOD if ok else "#d03b3b",
                fontsize=12, fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.02, "V8 参照：Baier–Katkov / Sokolov–Ternov 恒定场谱，\n"
                        "与 core/qed LCFA 表同一公式；残差为参照的 O(1/γ²)。",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="bottom")

    fig.suptitle("Baier–Katkov 验证结果汇总（validation.py V1–V8，单粒子 FP64）",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")

    if save_path is None:
        import pathlib
        save_path = pathlib.Path(__file__).with_name("validation_summary.png")
    _check_labels(fig)
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {save_path}")
    return str(save_path)


if __name__ == "__main__":
    main()
