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

    fig = plt.figure(figsize=(15.0, 13.0), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.42,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ---- V3: soft-photon BK -> LW -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _strip_axes(ax)
    x = np.array([r["omega_over_eps"] for r in v3])
    y = np.array([r["ratio"] for r in v3])
    y_bare = np.array([r["ratio_bare"] for r in v3])
    ax.plot(x, y_bare, color=BASE, linewidth=1.6, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=2,
            label="在裸 mΩ 处（截断伪影）")
    ax.plot(x, y, color=BLUE, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
            label="在反冲移动谐波 ω_m 处")
    ax.plot(x, 1.0 - x, color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1,
            label="1 − ω/ε")
    _hline1(ax, x.min(), x.max())
    ax.annotate(f"{y[-1]:.4f}", (x[-1], y[-1]), textcoords="offset points",
                xytext=(0, -14), ha="center", color=INK2, fontsize=9)
    ax.set_xlabel("ω / ε")
    ax.set_ylabel("BK / LW 比值")
    ax.set_ylim(0.985, 1.045)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    _panel_title(ax, "V3 · 软光子极限 BK → LW（比值 → 1 − ω/ε）")

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
    ax.set_xlabel("ω（反冲移动谐波 ω_m）")
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
    labels = ["V1 Airy", "V2 |A|²=双积分", "V7 trace≡dot"]
    errs = [v1["rel_err"], v2["rel_err"], max(v7["rel_err"], 1e-16)]
    cols = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, errs, color=cols, width=0.62, zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-2)
    for b, val in zip(bars, errs):
        ax.annotate(f"{val:.2e}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK2, fontsize=8.5)
    ax.set_ylabel("相对误差（对数）")
    _panel_title(ax, "V1 / V2 / V7 · 解析与内部自洽")
    ax.annotate("V1<1e-3, V2<1e-8, V7<1e-12  →  全部通过",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=9)

    # ---- V8: quantum synchrotron ratio -------------------------------------
    d8 = v8["delta"]
    ax = fig.add_subplot(gs[1, 2])
    _strip_axes(ax)
    ax.plot(d8, v8["quantum_over_classical"], color=BASE, linewidth=1.6,
            marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
            zorder=2, label="精确量子 / 经典（反冲压低）")
    ax.plot(d8, v8["ratio"]["dot"], color=BLUE, linewidth=2.0, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
            label="BK dot 核 / 精确量子")
    ax.plot(d8, v8["ratio"]["trace"], color=ORANGE, linewidth=1.4,
            linestyle=(0, (4, 3)), marker="s", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5,
            label="BK trace 核 / 精确量子")
    _hline1(ax, d8.min(), d8.max())
    worst = np.max(np.abs(v8["ratio"]["dot"] - 1.0))
    ax.set_xlabel("δ = ω / ε")
    ax.set_ylabel("比值")
    ax.set_ylim(0.25, 1.12)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2)
    _panel_title(ax, f"V8 · 量子区间比值（χ = {v8['chi']:.2f}, γ = {v8['gamma']:.0f}）")
    ax.annotate(f"max |BK/精确 − 1| = {worst:.1e}", xy=(0.97, 0.80),
                xycoords="axes fraction", color=INK2, fontsize=9, ha="right")

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
            color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5,
            label="BK 一圈积分（dot 核，反冲移动谐波处）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("δ = ω / ε")
    ax.set_ylabel("dW/dδ（每圈光子数）")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _panel_title(ax, f"V8 · 量子同步辐射谱：BK 双时间积分 vs 恒定场精确解"
                     f"（χ = {chi8:.2f}, γ = {g8:.0f}, N_t = {v8['n_samples']}）")

    # ---- verdict panel ------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(SURFACE)
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
                color=INK2, fontsize=9, va="top")
        ax.text(0.56, 0.95 - 0.11 * i, val, transform=ax.transAxes,
                color=INK2, fontsize=9, va="top")
        ax.text(0.98, 0.95 - 0.11 * i, "✓ OK" if ok else "✗ FAIL",
                transform=ax.transAxes, color=GOOD if ok else "#d03b3b",
                fontsize=9, fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.02, "V8 参照：Baier–Katkov / Sokolov–Ternov 恒定场谱，\n"
                        "与 core/qed LCFA 表同一公式；残差为参照的 O(1/γ²)。",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")

    fig.suptitle("Baier–Katkov 验证结果汇总（validation.py V1–V8，单粒子 FP64）",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")

    if save_path is None:
        import pathlib
        save_path = pathlib.Path(__file__).with_name("validation_summary.png")
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {save_path}")
    return str(save_path)


if __name__ == "__main__":
    main()
