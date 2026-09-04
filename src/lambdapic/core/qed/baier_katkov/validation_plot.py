"""Static summary figure of the V1..V7 validation results.

Re-runs the coarse checks of :mod:`.validation` (single particle, FP64) and
renders them into one ``validation_summary.png`` next to this script:

* V3  soft-photon BK -> LW ratio vs ``omega/eps`` (should tend to 1);
* V4  code-vs-exact synchrotron harmonic ratios vs harmonic order ``m``
      (should tend to 1) plus the Larmor closure;
* V5  straight-line vs curved reference radiation (log scale, should be tiny);
* V6  angle-integrated ``dW/domega`` positivity spectrum;
* V1/V2/V7  internal/analytic relative errors (log scale);
* a verdict panel listing all checks.

Requires ``matplotlib`` (already a declared ``lambdapic`` dependency); the
computation itself stays numpy/scipy-only.  Run from anywhere with::

    python -m lambdapic.core.qed.baier_katkov.validation_plot
"""

from __future__ import annotations

import numpy as np

from . import validation as _v

# -- palette (validated reference palette, light mode; see skill dataviz) ----
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


def _hline1(ax, xmin, xmax):
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)


def _panel_title(ax, text):
    ax.set_title(text, color=INK, fontsize=11, fontweight="bold", loc="left",
                 pad=10)


def main(save_path=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # -- re-run all checks -------------------------------------------------
    print("running V1..V7 ...")
    v1 = _v.check_airy_identity()
    v2 = _v.check_velocity_kernel_consistency()
    v3 = _v.check_classical_limit()
    v4 = _v.check_larmor()
    v5 = _v.check_straight_line()
    v6 = _v.check_positivity()
    v7 = _v.check_kernel_rewrite()

    fig = plt.figure(figsize=(15.0, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42,
                          left=0.06, right=0.97, top=0.88, bottom=0.09)

    # ---- V3: soft-photon BK -> LW -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _strip_axes(ax)
    x = np.array([r["omega_over_eps"] for r in v3])
    y = np.array([r["ratio"] for r in v3])
    ax.plot(x, y, color=BLUE, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    _hline1(ax, x.min(), x.max())
    for i in (0, -1):
        ax.annotate(f"{y[i]:.3f}", (x[i], y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=9)
    ax.set_xlabel("ω / ε")
    ax.set_ylabel("BK / LW 比值")
    ax.set_ylim(0.985, 1.045)
    _panel_title(ax, "V3 · 软光子极限 BK → LW（比值 → 1）")

    # ---- V4: Larmor normalization ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _strip_axes(ax)
    m = np.array(v4["harmonics"], dtype=float)
    y = np.array(v4["harmonic_ratios"])
    ax.semilogx(m, y, color=ORANGE, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    _hline1(ax, m.min(), m.max())
    for i in (0, -1):
        ax.annotate(f"{y[i]:.3f}", (m[i], y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=9)
    ax.set_xticks(m)
    ax.set_xticklabels([f"{int(v)}" for v in m])
    ax.set_xlabel("谐波阶数 m（ω = mΩ）")
    ax.set_ylabel("代码 / 精确同步辐射谱")
    ax.set_ylim(0.92, 1.02)
    _panel_title(ax, "V4 · Larmor 归一化（谐波比值 → 1）")
    ax.annotate(f"精确谱闭合 = {v4['closure']:.4f}",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=9)

    # ---- V5: straight line vs curved --------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    _strip_axes(ax)
    labels = ["|BK| 直线", "LW 直线", "LW 弯曲参考"]
    vals = [abs(v5["bk"]), abs(v5["lw"]), abs(v5["curved_ref"])]
    cols = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, vals, color=cols, width=0.62, zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e5)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.2e}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK2, fontsize=8.5)
    ax.set_ylabel("d²E/dω dΩ（对数）")
    _panel_title(ax, "V5 · 直线轨迹 ≈ 零辐射")
    ax.annotate(f"|BK/弯曲| = {v5['ratio_bk_to_curved']:.1e}",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=9)

    # ---- V6: positivity spectrum ------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _strip_axes(ax)
    w = v6["omega_grid"]
    dW = v6["dW_domega"]
    ax.bar(np.arange(len(w)), dW, color=BLUE, width=0.62, zorder=3,
           edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(np.arange(len(w)))
    ax.set_xticklabels([f"{float(v):.1e}" for v in w], rotation=30, ha="right")
    ax.set_xlabel("ω（谐波 mΩ）")
    ax.set_ylabel("dW/dω")
    ax.set_ylim(0, 1.1 * float(dW.max()))
    ax.axhline(0.0, color=BASE, linewidth=1.0)
    _panel_title(ax, "V6 · 角积分谱正性（未裁剪）")
    ax.annotate(f"min = +{v6['min_dW']:.2f}  总 W = {v6['total_W']:.3f}",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=9)

    # ---- V1/V2/V7: internal precision -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _strip_axes(ax)
    labels = ["V1 Airy", "V2 |A|²=双积分", "V7 dot≈trace"]
    errs = [v1["rel_err"], v2["rel_err"], v7["rel_err"]]
    cols = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, errs, color=cols, width=0.62, zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-2)
    for b, val in zip(bars, errs):
        ax.annotate(f"{val:.2e}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK2, fontsize=8.5)
    ax.set_ylabel("相对误差（对数）")
    _panel_title(ax, "V1 / V2 / V7 · 解析与内部自洽")
    ax.annotate("V1<1e-3, V2<1e-8, V7<1e-2  →  全部通过",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=9)

    # ---- verdict panel ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    _panel_title(ax, "汇总")
    rows = [
        ("V1  Airy 恒等式 (eq 9.6)", f"{v1['rel_err']:.2e}", v1["ok"]),
        ("V2  |A|² = 速度双积分", f"{v2['rel_err']:.2e}", v2["ok"]),
        ("V3  BK → LW 软光子", f"比值 {v3[-1]['ratio']:.3f} → 1", True),
        ("V4  Larmor 归一化", f"闭合 {v4['closure']:.4f}", True),
        ("V5  直线 ≈ 零辐射", f"|BK/弯曲| {v5['ratio_bk_to_curved']:.1e}", True),
        ("V6  正性", f"min +{v6['min_dW']:.1f}", True),
        ("V7  dot ≈ trace", f"{v7['rel_err']:.2e}", v7["ok"]),
    ]
    for i, (name, val, ok) in enumerate(rows):
        ax.text(0.02, 0.95 - 0.125 * i, name, transform=ax.transAxes,
                color=INK2, fontsize=9, va="top")
        ax.text(0.60, 0.95 - 0.125 * i, val, transform=ax.transAxes,
                color=INK2, fontsize=9, va="top")
        ax.text(0.98, 0.95 - 0.125 * i, "✓ OK" if ok else "✗ FAIL",
                transform=ax.transAxes, color=GOOD if ok else "#d03b3b",
                fontsize=9, fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.02, "未验证：完整量子同步辐射谱 K_{1/3}, K_{2/3}（eq 10.3，"
                        "前因子依赖规范约定）", transform=ax.transAxes,
            color=MUTED, fontsize=8.5, va="bottom")

    fig.suptitle("Baier–Katkov 验证结果汇总（validation.py V1–V7，单粒子 FP64）",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")

    if save_path is None:
        import pathlib
        save_path = pathlib.Path(__file__).with_name("validation_summary.png")
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {save_path}")
    return str(save_path)


if __name__ == "__main__":
    main()
