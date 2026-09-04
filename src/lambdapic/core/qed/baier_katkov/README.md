# Baier–Katkov reference module / Baier–Katkov 参考模块

This subpackage is a **minimal, physics-first, single-particle** implementation
of the quasi-classical Baier–Katkov (BK) radiation spectrum.  It is fully
decoupled from the online LCFA Monte Carlo pipeline in
`core/qed/radiation.py` / `optical_depth.py` / `inline.py` / `cpu.py`; it
does **not** modify them, and it does **not** mix with them.

本子包是准经典 Baier–Katkov（BK）辐射谱的**最小化、物理优先、单粒子**实现。它与在线
LCFA 蒙特卡洛管道（`core/qed/radiation.py` / `optical_depth.py` / `inline.py` /
`cpu.py`）完全解耦：**不**修改它们，也**不**与它们混用。

A full architecture / implementation document (module layout, per-file roles,
numerical methods, physics principles, known limitations, roadmap) lives at the
repository root: **`Baier-Katkov-architecture.md`**.

完整的架构与实现文档（模块结构、各文件职责、数值方法、物理原理、已知不足与未来
路线）位于仓库根目录：**`Baier-Katkov-architecture.md`**。

## What it computes / 计算内容

For one classical electron trajectory, and one outgoing photon
`k = (omega, omega n)` with `|n| = 1`, the spin-averaged, polarization-summed
single-particle differential spectra (natural units `c = hbar = 1`, `m_e = 1`):

对一条经典电子轨迹，以及一个出射光子 `k = (omega, omega n)`（`|n| = 1`），
自旋平均、极化求和的单粒子微分谱（自然单位 `c = hbar = 1`，`m_e = 1`）：

```
d2E/(dw dO) = (alpha/pi) * w^2 * q^2 * Re[ int dt1 dt2 N e^{-i Phi} ]
d2W/(dw dO) = d2E/(dw dO) / w
```

with / 其中

* `N` the double-time kernel — either the strict trace form (eq. 7.1) or the
  velocity / dot-product form (eq. 7.3), see `kernel.py`;
  `N` 为双时间核 —— 严格迹形式（式 7.1）或速度 / 点积形式（式 7.3），见 `kernel.py`；
* `Phi = (eps/eps') * w * [(t2 - t1) - n.(r2 - r1)]` the recoil phase
  (eq. 5.3), `eps' = eps - w`;
  `Phi = (eps/eps') * w * [(t2 - t1) - n.(r2 - r1)]` 为反冲相位（式 5.3），
  `eps' = eps - w`；
* `q` the charge multiplicity (`= 1` for a single electron).
  `q` 为电荷多重数（单电子时 `= 1`）。

The primary output is the angle-integrated `dW/domega` (and `dE/domega`),
obtained by quadrature over a cone of directions around the initial velocity.

主要输出是对角度积分后的 `dW/domega`（以及 `dE/domega`），通过对初始速度方向附近
一个方向锥做求积得到。

## Assumptions and conventions (read before trusting any number) / 假设与约定（采信任何数字前请先阅读）

* **Single particle**, FP64, direct trapezoid double-time integration.  No
  MPI/GPU/NUFFT/FFT, no formation-length windowing, no importance sampling.
  **单粒子**，FP64，直接梯形双时间积分。无 MPI/GPU/NUFFT/FFT，无形成长度窗口，
  无重要性采样。
* **Spin averaged, polarization summed** (the `Parameters` dataclass records
  this).  Resolved spin/polarization is out of scope.
  **自旋平均、极化求和**（`Parameters` 数据类记录此约定）。分辨自旋/极化不在范围内。
* **Recoil**: `eps' = eps - omega`, `eps/eps'` phase factor.  A single fixed
  incident energy `eps` is used for the whole record; time-dependent local
  energies (sec. 4.2 of `Baier-Katkov.md`) are *not* implemented.
  **反冲**：`eps' = eps - omega`，相位因子 `eps/eps'`。整条记录使用单一固定的入射
  能量 `eps`；随时间变化的局域能量（`Baier-Katkov.md` 第 4.2 节）**未**实现。
* **Finite trajectory endpoints**: the record is integrated as-is, with **no**
  adiabatic switch / asymptotic padding / vacuum-contact subtraction.  The
  velocity (dot) kernel already carries the `-1` vacuum term, which is why it
  is the default; the strict trace kernel still contains the bare contact term
  and is exposed for cross-checking only.  Abrupt record endpoints leave
  boundary radiation, so a straight line is not *exactly* zero for a short
  record — it tends to zero as the record grows (see `validation.py`).
  **有限轨迹端点**：记录按原样积分，**无**绝热开关 / 渐近延拓 / 真空接触项扣除。
  速度（点积）核已自带 `-1` 真空项，故为默认选择；严格迹核仍含裸接触项，仅用于
  交叉检验。记录端点突变会留下边界辐射，因此短记录的直线并非*恰好*为零 —— 随记录
  变长而趋于零（见 `validation.py`）。
* **Classical limit**: as `w/eps -> 0` the BK single-particle spectrum reduces
  to the classical Liénard–Wiechert trajectory spectrum (eq. 7.4), *not* to a
  multi-particle coherent LW total field.
  **经典极限**：当 `w/eps -> 0` 时，BK 单粒子谱退化为经典 Liénard–Wiechert 轨迹谱
  （式 7.4），而**不是**多粒子相干的 LW 总场。
* **Relation to the online LCFA**: completely independent; this module is an
  offline post-processor for recoilless background trajectories.  A trajectory
  that already contains stochastic LCFA recoil must **not** be fed here (double
  counting).
  **与在线 LCFA 的关系**：完全独立；本模块是无反冲背景轨迹的离线后处理器。已包含
  随机 LCFA 反冲的轨迹**不得**输入到这里（会重复计数）。
* **Ensemble**: for `P` independent macro-particles, `S_total = sum_i w_i S_i`
  (with `w_i` the macro-particle weight).  This module computes a single `S_i`;
  the weighted sum is intentionally not implemented here.
  **系综**：对 `P` 个独立宏粒子，`S_total = sum_i w_i S_i`（`w_i` 为宏粒子权重）。
  本模块计算单个 `S_i`；加权求和有意不在本模块实现。

## This is a reference implementation / 这是一个参考实现

It is **not** performance-optimized.  Do not benchmark it, and do not claim it
is faster than LCFA or LW.  Its purpose is to pin down units, phases, recoil,
positivity and the classical limit before any optimization (per
`Baier-Katkov-multiparticle.md` sec. 15, phase 1).

它**未**做性能优化。不要对它做基准测试，也不要声称它比 LCFA 或 LW 更快。其目的是
在任何优化之前先钉牢单位、相位、反冲、正性与经典极限（按
`Baier-Katkov-multiparticle.md` 第 15 节第 1 阶段）。

## Files / 文件

```
baier_katkov/
├── __init__.py        public API / 公共 API
├── units.py           natural units (c=hbar=1, m_e=1) and SI conversion / 自然单位与 SI 换算
├── types.py           Trajectory, Parameters, Spectrum dataclasses / 数据类
├── trajectory.py      linear interpolation of discrete trajectory samples / 离散轨迹样本线性插值
├── phase.py           double-time recoil phase, relative anchoring / 双时间反冲相位、相对锚定
├── kernel.py          trace / dot kernels, classical amplitude, Airy identity / 迹/点积核、经典振幅、Airy 恒等式
├── integrator.py      direct double-time integration -> dW/domega / 直接双时间积分
├── result.py          print / save spectrum / 打印 / 保存谱
├── validation.py      coarse sanity checks (V1..V7) / 粗粒度 sanity 检查（V1..V7）
└── examples/
    └── bk_single_particle.py   minimal end-to-end example / 最小端到端示例
```

## Running / 运行

```bash
# the end-to-end example (prints a spectrum table, writes bk_spectrum.txt)
# 端到端示例（打印谱表，写出 bk_spectrum.txt）
python -m lambdapic.core.qed.baier_katkov

# or the example script directly / 或直接运行示例脚本
python -m lambdapic.core.qed.baier_katkov.examples.bk_single_particle

# the sanity checks / sanity 检查
python -m lambdapic.core.qed.baier_katkov.validation
```

The three commands above work once `lambdapic` is installed (which also pulls in
the full simulation stack via the package's top-level `__init__.py`).  Because
this subpackage is decoupled and needs only `numpy` + `scipy`, it can also be
run **standalone** without importing the rest of `lambdapic`:

上述三条命令在 `lambdapic` 安装后即可工作（安装同时会经包顶层 `__init__.py` 引入
完整模拟栈）。由于本子包已解耦、仅依赖 `numpy` + `scipy`，它也可以**独立**运行，
无需导入 `lambdapic` 的其余部分：

```bash
PYTHONPATH=src python - <<'PY'
import pathlib, sys, types
src = pathlib.Path("src").resolve()
sys.path.insert(0, str(src))
for name in ("lambdapic", "lambdapic.core", "lambdapic.core.qed"):
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = [str(src.joinpath(*name.split(".")))]
    sys.modules[name] = mod
from lambdapic.core.qed.baier_katkov.examples.bk_single_particle import main
main()   # replace with validation.main() for the sanity checks / 运行 sanity 检查时改为 validation.main()
PY
```
