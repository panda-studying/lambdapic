# Baier–Katkov 双时间积分辐射谱模块说明

本文档是 `lambdapic.core.qed.baier_katkov` 子包的完整说明：模块定位、物理原理与数学方法、数值积分方式、架构与数据流、每个文件的职责、每个函数的功能，以及验证体系和已知限制。

---

## 目录

1. [模块定位与范围](#1-模块定位与范围)
2. [物理原理与数学方法](#2-物理原理与数学方法)
3. [数值方法与积分方式](#3-数值方法与积分方式)
4. [单位制](#4-单位制)
5. [架构与数据流](#5-架构与数据流)
6. [各文件与函数详解](#6-各文件与函数详解)
7. [使用示例](#7-使用示例)
8. [运行环境与常用命令](#8-运行环境与常用命令)
9. [验证与回归测试](#9-验证与回归测试)
10. [已知限制与路线图](#10-已知限制与路线图)
11. [相关文档](#11-相关文档)

---

## 1. 模块定位与范围

**本模块做什么**：用 Baier–Katkov（BK）准经典算符方法的**双时间积分**直接计算单粒子在经典外场（如激光、磁约束场）中运动的辐射谱。它与主模拟的 LCFA Monte Carlo 管线完全解耦，是独立的参考实现。它的目标（见 `REVIEW_AND_ROADMAP.md` §8）是给出 LCFA 局部近似给不出的量子结果：LCFA 在每个时刻只用当地场强决定发射率，而 BK 双时间积分保留了两个发射时刻之间的量子干涉（形成长度效应）与量子反冲，能正确处理 LCFA 失效的场景。

**当前范围（phase 1，即本版实现）**：

- 单粒子、FP64、直接双时间积分（无 FFT/NUFFT）；
- 自然单位 $c = \hbar = 1$，电子质量 $m_e = 1$（Compton 单位，SI 映射见 [§4](#4-单位制)）；
- 自旋平均、极化求和的顶点核；
- 角积分谱 $dW/d\omega$（每粒子每单位光子能量的发射概率）；
- 支持**固定入射能量** $\varepsilon$ 和**局域能量** $\varepsilon(t)$ 两种模式（后者为 2026-09-09 完成的 P-2 项，见 [§2.4](#24-局域能量推广-p-2)）。

**明确不含**（当前版本不做）：多粒子系综求和、MPI/GPU、FFT/NUFFT 加速、形成长度窗口截断、重要性采样、LCFA/BK 混合模型、自旋分辨。它是一个**参考实现**，用来钉死单位、相位、反冲、正性、经典极限等约定，再以它为基准优化和集成。

**重要使用前提**：输入轨迹必须是**无辐射反冲的经典背景运动**。如果轨迹记录里已经包含随机 LCFA 反冲（PIC 主模拟的电子轨迹），再喂给本模块就是双重计数——本模块自己处理量子反冲（见 [§2.2](#22-相位-反冲相位与符号约定)）。

---

## 2. 物理原理与数学方法

### 2.1 主公式与前因子

自然单位 $c = \hbar = m_e = 1$、度规 $(+---)$ 下，单粒子在经典轨迹 $\mathbf r(t)$ 上的微分辐射能量谱为（Baier & Katkov 1967；Sokolov–Ternov；Ritus 1985 的准经典算符方法）：

$$\frac{d^2E}{d\omega\,d\Omega} = \frac{\alpha}{4\pi^2}\,\omega^2 q^2\,\mathrm{Re}\iint dt_1\,dt_2\; N(t_1,t_2)\; e^{-i\Phi(t_1,t_2)}$$

$$\frac{d^2W}{d\omega\,d\Omega} = \frac{1}{\omega}\,\frac{d^2E}{d\omega\,d\Omega}$$

各量含义：

| 符号 | 含义 |
|---|---|
| $\omega$ | 光子角频率（自然单位，$m_e$ 作单位；$\omega = 1$ 即光子能量 0.511 MeV） |
| $N(t_1,t_2)$ | 自旋平均、极化求和的**顶点核**（见 [§2.3](#23-核-dottracevelocity)） |
| $\Phi(t_1,t_2)$ | 含反冲的**双时间相位**（见 [§2.2](#22-相位-反冲相位与符号约定)） |
| $q$ | 电荷重数 $\lvert q\rvert/e$（单电子 $=1$；谱按 $q^2$ 标度） |
| $\alpha$ | 精细结构常数（CODATA，约 $1/137.036$） |

前因子 $\alpha/(4\pi^2)$ 是 Liénard–Wiechert 谱的 $e^2/(4\pi^2)$（Jackson 14.67，Gaussian 单位制，$e^2 = \alpha$）。它精确复现 Schwinger 同步辐射谱与 Larmor 功率（validation V4），与主模拟 `core/qed/optical_depth_tables` 中 LCFA 率的归一化一致。⚠️ 不要把 Heaviside–Lorentz 电荷 $e_{HL}^2 = 4\pi\alpha$ 插进这个 Gaussian 公式——那会多出 $4\pi$ 因子（这是历史上 BUG-5 的来源，见 `REVIEW_AND_ROADMAP.md` §2）。

被积函数的实部是物理谱；虚部本应为零（见 [§2.5](#25-厄米对称性与实谱)）。

### 2.2 相位：反冲相位与符号约定

真空中光子的 4-波矢为 $k = (\omega, \omega\mathbf n)$，$\lvert\mathbf n\rvert = 1$，故 $kx = \omega(t - \mathbf n\cdot\mathbf r)$。固定入射能量 $\varepsilon$ 时，电子发射光子后末态能量为

$$\varepsilon' = \varepsilon - \omega$$

双时间反冲相位为（设计文档 `Baier-Katkov.md` eq. 5.3 / 6.1 所指的约定）：

$$\Phi(t_1, t_2) = \frac{\varepsilon}{\varepsilon'}\,\omega\,\Big[(t_2 - t_1) - \mathbf n\cdot(\mathbf r_2 - \mathbf r_1)\Big]$$

- 因子 $\varepsilon/\varepsilon'$ 是**量子反冲**进入相位的方式：它把有效频率重新标定为 $\omega\varepsilon/\varepsilon'$（`phase.recoil_frequency`），而**不改变**光子的实际频率。经典极限（无反冲，$\varepsilon' = \varepsilon$）下该因子退化为 1。
- 要求 $\varepsilon' > 0$，即 $\omega < \varepsilon$——光子能量不能超过入射电子能量（代码中有守卫，BUG-3）。
- 被积函数携带 **$e^{-i\Phi}$**（共轭符号）。这个符号不能靠纯对称积分钉死（V1 说明），是由 BK → Liénard–Wiechert 经典极限（validation V2/V3）验证确定的。

### 2.3 核：dot / trace / velocity

自旋平均、极化求和后有两种等价的 BK 核（`kernel.py`）：

**dot 核（速度形式，eq. 7.3，默认）**，$\gamma = \varepsilon/m$：

$$N_\mathrm{dot} = \frac{(\varepsilon^2 + \varepsilon'^2)\,(\mathbf b_1\cdot\mathbf b_2 - 1) + \omega^2/\gamma^2}{2\,\varepsilon'^2}$$

**trace 核（迹形式，eq. 7.1）**：

$$N_\mathrm{trace} = -\frac{m^2}{\varepsilon\,\varepsilon'} - \frac{\varepsilon^2 + \varepsilon'^2}{4\,\varepsilon'^2}\,(\mathbf b_1 - \mathbf b_2)^2$$

其中 $\mathbf b_i = \boldsymbol\beta(t_i)$ 是无量纲速度 $\mathbf v/c$。

**在壳恒等式（两者逐点相等）**：对同能量的两个在壳速度样本，$\mathbf b_i^2 = 1 - m^2/\varepsilon^2$。利用 $(\mathbf b_1-\mathbf b_2)^2 = \mathbf b_1^2 + \mathbf b_2^2 - 2\mathbf b_1\cdot\mathbf b_2$，trace 形式的常数项合并为

$$\frac{m^2(\varepsilon - \varepsilon')^2}{2\,\varepsilon^2\varepsilon'^2} = \frac{\omega^2}{2\,\varepsilon'^2\gamma^2}$$

正好等于 dot 形式的 $\omega^2/(2\varepsilon'^2\gamma^2)$ 接触项，因此 **$N_\mathrm{dot} \equiv N_\mathrm{trace}$ 逐点成立**（validation V7 在 $\omega/\varepsilon = 0.7$ 的硬光子处验证到 $10^{-12}$）。两个形式的"$-1$"与"$-m^2/(\varepsilon\varepsilon')$"项是同一真空减除，都不需要额外的端点减除。

**velocity 核（经典横向速度核）**：$N_\mathrm{vel} = \mathbf b_1\cdot\mathbf b_2 - (\mathbf n\cdot\mathbf b_1)(\mathbf n\cdot\mathbf b_2) = [\mathbf n\times(\mathbf n\times\mathbf b_1)]\cdot[\mathbf n\times(\mathbf n\times\mathbf b_2)]$，即 Liénard–Wiechert 振幅的模方。它只用于内部一致性检查（validation V2），与 BK 两核相差全导数（边界）项——闭环上在谐波频率处二者一致。

⚠️ **历史教训（BUG-6）**：trace 核曾抄写错系数（把 $( \varepsilon^2+\varepsilon'^2)/(4\varepsilon'^2)$ 写成 $( \varepsilon^2+\varepsilon'^2)/(4\varepsilon\varepsilon')$，差一个 $\varepsilon/\varepsilon'$ 因子），导致"trace 与 dot 相差 $O(\omega)$"的假象。已修复，两核现严格重合。

### 2.4 局域能量推广（P-2，2026-09-09 完成）

固定 $\varepsilon$ 只适用于均匀/恒定能量轨道。PIC 轨迹中电子能量沿途变化（激光等离子体加速，变化可达 $O(1)$），必须在**每个顶点**使用局域能量 $\varepsilon(t_i)$。局域推广把相位和核都拆成逐顶点贡献：

$$\Phi_{ij} = \omega\Big[f_j\,x_j - f_i\,x_i\Big], \qquad x = t - \mathbf n\cdot\mathbf r, \qquad f_i = \frac{\varepsilon_i}{\varepsilon'_i}$$

$$N_{ij} = (A_i + A_j) + (B_i + B_j)\,(\mathbf b_i\cdot\mathbf b_j - 1)$$

逐顶点反冲为 $\varepsilon'_i = \varepsilon_i - \omega$。逐顶点系数为

| | $A_i$ | $B_i$ |
|---|---|---|
| trace | $-\dfrac{m^2}{2\varepsilon_i\varepsilon'_i} + \dfrac{m^2 c_i}{4\varepsilon_i^2}$ | $c_i/4$ |
| dot | $\dfrac{m^2\omega^2}{4\varepsilon_i^2\varepsilon'_i^2}$ | $c_i/4$ |

其中 $c_i = (\varepsilon_i^2 + \varepsilon'_i^2)/\varepsilon'_i^2$。固定能量时（$\varepsilon_1 = \varepsilon_2$）上式精确退化回 [§2.3](#23-核-dottracevelocity) 的形式。

**关键发现**：逐顶点反冲约定下，每个顶点独立满足在壳恒等式 $(\varepsilon_i - \varepsilon'_i)^2 = \omega^2$，因此局域 dot 与局域 trace **仍然逐顶点重合**（回归测试验证到 $10^{-10}$ 相对误差）。旧的"$\varepsilon_1\ne\varepsilon_2$ 时两核不同"的预期只在"全局固定 $\varepsilon'$ 而 $\varepsilon_i$ 变化"的约定下成立；本模块采用逐顶点约定，两核合一。唯一例外是**经典模式**（$\varepsilon'_i = \varepsilon_i$，$f_i = 1$）：此时 trace 形式给出纯 $\mathbf b_i\cdot\mathbf b_j - 1$（无 $\omega$ 接触项），是基本形式；dot 形式保留 $m^2\omega^2$ 接触项。经典模式下谱与能量历史无关（$A_i = 0$、$B_i = 1/2$、$f = 1$ 不依赖 $\varepsilon$），回归测试验证了这一点。

**在壳一致性由调用方负责**：核假定 $\lvert\mathbf b_i\rvert^2 = 1 - m^2/\varepsilon_i^2$，即能量历史与动量历史必须满足 $\varepsilon_i = m\sqrt{1+\lvert\mathbf u_i\rvert^2}$。`Trajectory.local_energy()` 的默认回退值正是这个在壳值；显式传入的 `energy`（如 PIC 能量历史）与 `momentum` 的一致性不在模块内校验。

相位用**差分稳定形式**求值（`phase.local_recoil_phase` / numba 内核）：

$$\Phi_{ij} = \omega\Big[\bar f\,(x_j - x_i) + \Delta f\,\frac{x_j + x_i}{2}\Big], \qquad \bar f = \frac{f_i + f_j}{2},\quad \Delta f = f_j - f_i$$

能量变化只进入小量 $\Delta f$ 项，超相对论小角相位 $\sim(\gamma^{-2} + \theta^2)$ 不会被大数相减吞掉（与固定能量路径同一原理，见 [§3.5](#35-数值稳定性与采样判据)）。

### 2.5 厄米对称性与实谱

$N$ 对 $(t_1,t_2)$ 对称、$\Phi$ 反对称（$1 \leftrightarrow 2$ 交换下 $\Phi \to -\Phi$，局域模式连同能量一起交换），因此整个双时间核满足

$$K(t_2, t_1) = K(t_1, t_2)^*, \qquad K = N e^{-i\Phi}$$

于是双重积分严格为实数。numpy 后端算全方阵、保留虚部作为舍入误差诊断（物理上应为零，量级 $10^{-15}$ 左右）；numba 后端显式利用对称性只算上三角（$j \ge i$，非对角项乘 2），虚部构造性地为零。局域能量模式保持这一性质（回归测试验证虚部/实部 $\sim 10^{-16}$）。

### 2.6 经典极限与软光子对照

- **无反冲极限**（$Parameters.recoil = "classical"$，即 $\varepsilon' = \varepsilon$、相位因子 1）：BK 双时间积分在软光子区（$\omega/\varepsilon \ll 1$）与经典 Liénard–Wiechert 谱重合，偏差 $O(\omega/\varepsilon)$（validation V3 定量验证：比值 $\to 1 - \omega/\varepsilon$）。
- **恒定场对照**（validation V8）：对均匀圆周运动，沿轨迹场强严格恒定，恒定场量子同步辐射公式（$K_{1/3}, K_{2/3}$，与 LCFA 表同一公式）就是本模块双时间积分的解析求值。一圈 BK 谱在反冲移动谐波处与精确解之比 0.997–1.001（残差为参照谱的超相对论误差 $O(1/\gamma^2)$）。

---

## 3. 数值方法与积分方式

### 3.1 双时间梯形积分

积分核 $\mathrm{Re}[N e^{-i\Phi}]$ 在两个时间维度上都用**梯形法则**直接求和（`integrator.trapz_weights`，支持非均匀时间网格）：

$$I = \sum_{i,j} w_i w_j\, N(t_i, t_j)\, e^{-i\Phi(t_i, t_j)}, \qquad w = \text{trapezoid weights}$$

没有用 FFT、没有插值、没有窗口截断——这是最直白、最不容易出错的参考实现。

### 3.2 numpy 后端（`backend="numpy"`，ground truth）

- 按行**分块外积**（`chunk` 参数，默认 256 行/块）控制内存：每块构造 $(N_{chunk}, N_t)$ 的相位与核矩阵，`einsum` 收缩。
- 算**全方阵**，返回复数；虚部即舍入误差诊断（`double_time_integral` 的复数输出、`d2_energy(diagnostics=True)` 的第二个返回值）。
- 速度慢，只作正确性基准与交叉检查用。

### 3.3 numba 后端（`backend="numba"`，默认，快约 140×）

- `@njit(parallel=True, cache=True)`，`prange` 并行。
- **只算上三角**：$j \ge i$，非对角 $(i,j)$ 对贡献乘 2（厄米对称性，[§2.5](#25-厄米对称性与实谱)）。
- **折行负载均衡**：行 $i$ 与行 $N_t-1-i$ 配对进同一任务（两行合计恰好 $N_t - 1$ 个非对角对），每个并行任务工作量相同。
- **批量计算**：一次遍历把所有 $(\omega_k, \mathbf n_d)$ 对同时累加（`_double_sum_batch`）——固定能量模式下相位参数 `fac[k]`、核系数 `a[k], b[k]` 按频率预排；局域模式下逐顶点数组 `A, B, f` 形状 $(N_\omega, N_t)$，相位为 $\omega_k[\bar f\psi_d + \Delta f\,\chi_d]$，其中 $\psi_d = \Delta t - (\mathbf n_d\cdot\mathbf r_j - \mathbf n_d\cdot\mathbf r_i)$（差分形式）、$\chi_d = \bar t - (\mathbf n_d\cdot\mathbf r_j + \mathbf n_d\cdot\mathbf r_i)/2$（能量变化项，反冲因子恒定即 $\Delta f = 0$ 时该项消失）。
- `block` 参数控制每个并行任务的行数，默认 `max(1, min(32, ⌈Nt/2⌉/(4·线程数)))`。
- 与 numpy 后端逐位一致到舍入误差（回归测试断言 $< 10^{-12}$）。

### 3.4 角度积分

角积分谱由方向锥面上的数值求积得到：

- `integrator.cone_directions(axis, theta_max, n_theta, n_phi)`：$\cos\theta$ 用 **Gauss–Legendre** 采样（区间 $[\cos\theta_{max}, 1]$），$\phi$ 均匀采样；返回方向数组 $(n_\theta n_\phi, 3)$ 与每个方向的立体角权重 $d\Omega$（总和恰为锥面立体角 $2\pi(1-\cos\theta_{max})$）。
- `reference.orbit_direction_grid(gamma, ...)`：x-y 平面圆轨道的专用网格——利用绕 z 轴方位对称与轨道平面对称，只取上半球 $\phi = 0$ 的仰角 $\psi$ 方向，权重带 $2\cdot 2\pi\cos\psi\,d\psi$；内锥（默认半宽 $8/\gamma$，同步辐射锥）与外区各用一组 Gauss–Legendre。
- `BKIntegrator.compute_spectrum` 默认锥轴为初始速度方向、默认 $\theta_{max} = 5/\gamma$；对闭环谱通常显式传 `theta_max=np.pi` 做全空间积分。

### 3.5 数值稳定性与采样判据

- **相位差分构造**：所有相位都从差 $t_2 - t_1$、$\mathbf n\cdot(\mathbf r_2 - \mathbf r_1)$ 构造，而不是两个独立的大数相位相减——超相对论小角相位 $\sim(\gamma^{-2} + \theta^2)$ 因此不被消去误差吞掉（`phase.py` 与 numba 内核均如此）。局域能量模式下见 [§2.4](#24-局域能量推广-p-2) 的 $\bar f/\Delta f$ 分解。
- **采样判据（V8）**：双积分中最快相位速率为 $2\omega\varepsilon/\varepsilon' = 2m\Omega$（速度与 $\mathbf n$ 反平行处），梯形求积在 $2m\Omega\,\Delta t > 2\pi$ 时混叠，故一圈采样数需 $N_t > 2m_{max}$（`check_quantum_synchrotron` 默认取 $2m_{max}$ 再乘采样因子 2）。

### 3.6 复杂度与性能现状

- 每条轨迹 $O(N_t^2)$，每张谱 $O(N_t^2 N_\omega N_{dir})$——两个后端是同一方法的不同求值方式，不是不同方法。
- numba 相对 numpy 参考路径约 140× 加速（`REVIEW_AND_ROADMAP.md` §5/§9）。
- $O(N_t N_\omega)$ 的带状截断（利用核随 $\lvert t_1 - t_2\rvert$ 的局域性）在路线图 P2 中，尚未实现。

---

## 4. 单位制

本模块内部统一使用**自然单位（Compton 单位）**：$c = \hbar = 1$ 且 $m_e = 1$。自然单位尺度（`units.py`）：

| 量 | 自然单位 | SI 值（约） |
|---|---|---|
| 能量 $E$ | $m_e$ | $m_e c^2 = 0.511\ \mathrm{MeV}$ |
| 时间 $t$ | $m_e^{-1}$ | $\hbar/(m_e c^2) = 1.29\times 10^{-21}\ \mathrm{s}$ |
| 长度 $x$ | $m_e^{-1}$ | $\hbar/(m_e c) = 3.86\times 10^{-13}\ \mathrm{m}$ |
| 角频率 $\omega$ | $m_e$ | $m_e c^2/\hbar$（$\omega_{nat} = 1$ 即光子能量 0.511 MeV） |

要点：

- `units.py` 同时支持"joule"表象（$c=\hbar=1$ 但能量仍以焦耳计），二者只差一个 $m_e c^2$ 的整体标度，无量纲量（相位、$dW$）不变；模块计算只用 $m_e = 1$ 表象。
- 谱 $dW/d\omega$ 无量纲，$dE/d\omega$ 以 $m_e$ 为单位；$\omega_{nat} \to \mathrm{eV}$ 用 `natural_omega_to_photon_energy_ev`（乘 0.511 MeV）。
- 所有 SI 常数取自 `scipy.constants`（CODATA）。
- ⚠️ 辐射前因子只许用 Gaussian 自然单位 $e^2 = \alpha$，勿混入 Heaviside–Lorentz 电荷（[§2.1](#21-主公式与前因子)、BUG-5）。

---

## 5. 架构与数据流

模块由三个层次组成：**数据结构** → **物理核心（相位/核/积分器）** → **验证与输出**。核心计算链路如下：

```
                输入
   Trajectory(t, r, u, [ε(t)])      Parameters(ε, recoil, kernel, ...)
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
              BKIntegrator / compute_spectrum   (integrator.py)
                         │
        ┌────────────────┼───────────────────────┐
        ▼                ▼                        ▼
   phase.py         kernel.py                 types.Spectrum
   （相位 Φ）       （核 N）                    （结果容器）
        └────────────────┬───────────────────────┘
                         ▼
        double_time_integral[_batch]   (integrator.py)
        ├─ numpy 后端：全方阵复数求和（基准）
        └─ numba 后端：上三角并行求和（默认，~140×）
                         ▼
        I(ω, n) = Re ∬ N e^{−iΦ} dt₁ dt₂
                         ▼
        d²E/dωdΩ = (α/4π²)·ω²·q²·I     d²W/dωdΩ = d²E/dωdΩ / ω
                         ▼
        cone_directions / orbit_direction_grid（Gauss–Legendre 方向网格）
                         ▼
        dW/dω, dE/dω  →  Spectrum  →  result.py（打印 / 存文本）
```

验证侧（不参与计算链路）：

```
validation.py  V1–V8  ──► validation_plot.py ──► validation_summary.png
      │                        （按需）
      └── reference.py（解析参考谱：量子/经典同步辐射、Larmor、Airy）
```

依赖方向：`integrator → kernel/phase/types/units`；`reference` 与 `validation*` 只被验证和示例使用；`trajectory` 只依赖 `types`。核心计算不依赖 matplotlib（仅 `validation_plot` 需要）。

---

## 6. 各文件与函数详解

### 6.1 `types.py` — 数据结构

**`Trajectory`**（dataclass）：单粒子经典轨迹的时间网格采样。字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `time` | `(Nt,)` | 严格递增的采样时刻 |
| `position` | `(Nt, 3)` | 位置 |
| `momentum` | `(Nt, 3)` 或 None | **归一化动量** $\mathbf u = \gamma\boldsymbol\beta$（与 λPIC 的 `ux, uy, uz` 同约定）；无则从位置差分恢复 |
| `energy` | `(Nt,)` 或 None | 局域能量历史 $\varepsilon(t_i)$。**一旦给出即切换到局域能量模式**（[§2.4](#24-局域能量推广-p-2)）：核与相位改用 $\varepsilon(t_1), \varepsilon(t_2)$ 与逐顶点反冲 $\varepsilon'_i = \varepsilon_i - \omega$。这就是把 PIC 能量历史（加速电子）喂进模块的入口。要求全部为正有限值 |
| `mass` | float | 质量（自然单位，电子 $=1$） |
| `charge` | float | 电荷重数 $\lvert q\rvert/e$（谱按 $q^2$ 标度） |

方法/校验：

- `__post_init__()`：类型转换与全部形状/正性校验。
- `n_samples`（property）：采样数 $N_t$。
- `beta()`：有 `momentum` 时用在壳精确式 $\boldsymbol\beta = \mathbf u/\sqrt{1+\lvert\mathbf u\rvert^2}$；否则对 `position` 中心差分（放大采样噪声，生产勿用）。
- `gamma()`：有 `momentum` 时 $\gamma = \sqrt{1+\lvert\mathbf u\rvert^2}$；否则 $1/\sqrt{1-\beta^2}$。
- `local_energy()`：有显式 `energy` 返回之；否则回退在壳值 $m\cdot\gamma$。

**`_beta_from_position(time, position)`**：中心差分求速度，端点用前/后向差分（仅在无 `momentum` 时使用）。

**`Parameters`**：一次 BK 计算的物理与簿记约定。字段：`epsilon`（入射能量；仅在轨迹无 `energy` 时被核/相位使用）、`mass=1.0`、`charge=1.0`、`spin_averaged=True`、`polarization_summed=True`（二者当前必须为 True，未实现其余情形）、`recoil="baier_katkov"`（`"classical"` 为无反冲对照）、`kernel="dot"`（`"trace"` 等价）。方法：

- `epsilon_prime(omega)`：末态能量标签 $\varepsilon'$——BK 模式 $\varepsilon-\omega$（守卫 $\omega<\varepsilon$），经典模式 $\varepsilon$。
- `recoil_factor(omega)`：反冲相位因子 $\varepsilon/\varepsilon'$。

**`Spectrum`**：单粒子微分谱结果容器。字段：`omega`（$(N_\omega,)$）、`dW_domega`（$dW/d\omega$，主输出，无量纲）、`dE_domega`（可选，$dE/d\omega = \omega\,dW/d\omega$）、`metadata`（来源信息字典：$\varepsilon$、是否局域能量、质量、电荷、反冲/核类型、采样数、时间跨度、$\theta_{max}$、方向网格、后端、单位制等）。方法：

- `total_probability()`：$W = \int d\omega\, dW/d\omega$（梯形）。
- `total_energy()`：$E = \int d\omega\, dE/d\omega$。

### 6.2 `trajectory.py` — 轨迹输入与插值

- `as_trajectory(time, position, momentum=None, energy=None, mass=1.0, charge=1.0)`：从裸数组构造 `Trajectory`，校验时间严格递增（$t_0 < t_1 < \dots$）。
- `uniform_trajectory(r_of_t, t0, t1, n_samples, **kwargs)`：把可调用对象 $r(t) \to (3,)$（或 $(N_t,3)$）在均匀网格上采样成轨迹，适合构造解析测试轨迹。
- `linear_interpolate(r, time, t_query)`：把 $(N_t,3)$ 位置历史线性插值到查询时刻（查询点必须在记录区间内）。

模块级约定（docstring 明示）：样本按时间有序；`momentum` 即 $\mathbf u = \gamma\boldsymbol\beta$；无动量时中心差分（噪声放大）；轨迹是**有限记录**，端点效应不在本层去除（[§10](#10-已知限制与路线图)）；**含随机 LCFA 反冲的轨迹不得喂入本模块**（双重计数）。

### 6.3 `phase.py` — 相位因子

全部相位遵守 [§2.2](#22-相位-反冲相位与符号约定) 的符号约定，并从**差分**构造以保数值稳定。

- `classical_phase(t1, t2, omega, n, r1, r2)`：经典双时间相位 $\omega[(t_2-t_1) - \mathbf n\cdot(\mathbf r_2-\mathbf r_1)]$。
- `recoil_frequency(omega, epsilon, epsilon_prime=None)`：有效反冲频率 $\omega\varepsilon/\varepsilon'$（$\varepsilon'$ 默认 $\varepsilon-\omega$；守卫 $\varepsilon'>0$）。**不是**光子频率的改变，而是形成长度积分内部相位的重新标定。
- `recoil_phase(t1, t2, omega, n, r1, r2, epsilon, epsilon_prime=None)`：固定能量 BK 双时间相位 $\Phi = (\varepsilon/\varepsilon')\,\omega[(t_2-t_1) - \mathbf n\cdot(\mathbf r_2-\mathbf r_1)]$。$1\leftrightarrow 2$ 反对称 → 双时间核厄米。
- `local_recoil_phase(t1, t2, omega, n, r1, r2, epsilon1, epsilon2, epsilon_prime1=None, epsilon_prime2=None)`：局域能量相位（[§2.4](#24-局域能量推广-p-2)），差分稳定形式 $\Phi = \omega[\bar f(x_2-x_1) + \Delta f(x_1+x_2)/2]$；顶点标签连同能量交换下反对称 → 厄米性保留。
- `phase_along(t, omega, n, r, t0=None, r0=None)`：单时刻（振幅）相位 $\omega[(t-t_0) - \mathbf n\cdot(\mathbf r-\mathbf r_0)]$，供经典 LW 振幅使用；`t0/r0` 用于把相位锚定在首个样本，避免大数相消。

### 6.4 `kernel.py` — 顶点/旋量核

- `_eps_prime(epsilon, omega, epsilon_prime)`：内部辅助，默认 $\varepsilon' = \varepsilon - \omega$。
- `trace_kernel(beta1, beta2, epsilon, omega, mass=1.0, epsilon_prime=None, epsilon2=None, epsilon_prime2=None)`：trace 核（[§2.3](#23-核-dottracevelocity)）。固定能量时用精确的旧公式；传入 `epsilon2`（局域模式）用对称形式 $(A_1+A_2)+(B_1+B_2)(\mathbf b_1\cdot\mathbf b_2-1)$。
- `dot_kernel(...)`：dot 核，同样支持 `epsilon2` 局域分支。局域模式下与 `trace_kernel` 逐顶点重合（关键发现，[§2.4](#24-局域能量推广-p-2)）。
- `classical_velocity_kernel(beta1, beta2, n)`：经典横向速度核 $\mathbf b_1\cdot\mathbf b_2 - (\mathbf n\cdot\mathbf b_1)(\mathbf n\cdot\mathbf b_2)$（LW 参考谱用）。
- `classical_amplitude(time, position, beta, omega, n)`：经典 LW 远场振幅 $A = \int dt\,\mathbf v(t)\,e^{i\omega[t-\mathbf n\cdot\mathbf r(t)]}$，$\mathbf v = \mathbf n\times(\mathbf n\times\boldsymbol\beta)$，梯形积分；相位锚定首样本。
- `airy_identity(alpha, b)`：Airy 相位积分恒等式 $\int_{-\infty}^{\infty} e^{-i(\alpha\tau + b\tau^3)}d\tau = 2\pi(3b)^{-1/3}\mathrm{Ai}(\alpha/(3b)^{1/3})$——**仅用于 validation V1**，检验相位符号约定与 sec. 9 的三次标定。

### 6.5 `integrator.py` — 双时间积分引擎（核心）

**常量与后端选择**

- `PREFACTOR`：$\alpha/(4\pi^2)$。
- `available_backends()`：本安装可用的后端元组（numpy 恒在，numba 视导入而定）。
- `_resolve_backend(backend)`：`"auto"` → numba 可用则 numba；请求 numba 而不可导入时报错。

**固定能量路径的辅助函数**

- `trapz_weights(time)`：非均匀网格的一维梯形权重（单样本返回零权重，调用方应更早拒绝 $N_t < 2$）。
- `_final_energy(omega, epsilon, epsilon_prime)`：把 $\varepsilon'$ 广播到 $\omega$ 形状；None 表示 BK 反冲 $\varepsilon-\omega$（守卫 $\omega<\varepsilon$）。
- `_phase_factor(phase, omega, epsilon, eps_p)`：相位中 $(t_2-t_1) - \mathbf n\cdot(\mathbf r_2-\mathbf r_1)$ 前面的频率乘子——`"recoil"` 用 $\omega\varepsilon/\varepsilon'$，`"classical"` 用裸 $\omega$。
- `_kernel_coefficients(kernel, omega, epsilon, eps_p, mass)`：把固定能量核写成 $N = a + b\,x$ 的逐频率系数（`dot`：$x = \mathbf b_1\cdot\mathbf b_2 - 1$；`trace`：$x = (\mathbf b_1-\mathbf b_2)^2$；`velocity`：$x = \mathbf b_1\cdot\mathbf b_2 - (\mathbf n\cdot\mathbf b_1)(\mathbf n\cdot\mathbf b_2)$）。

**局域能量路径的辅助函数**

- `_local_vertex_arrays(kernel, omega, energy, phase, mass)`：构造逐（频率, 顶点）数组 `A, B, f`，形状 $(N_\omega, N_t)$（[§2.4](#24-局域能量推广-p-2) 的逐顶点系数；`"recoil"` 相位守卫 $\omega_k < \varepsilon(t_i)$ 对全部样本成立）。
- `_check_local_energy(energy, nt, kernel, epsilon_prime)`：校验能量历史（形状/有限/正），拒绝 velocity 核与非 None 的 `epsilon_prime`（局域模式的 $\varepsilon'$ 由逐顶点约定唯一确定）。

**numpy 求和核心**

- `_double_sum_numpy(time, position, beta, n, factor, kernel, epsilon, omega, eps_p, mass, chunk)`：固定能量全方阵双重和（分块外积，复数）。
- `_double_sum_numpy_local(time, position, beta, n, omega, A, B, f, chunk)`：局域能量全方阵双重和（单频率；`dot = beta @ beta.T`，相位差分稳定形式）。

**numba 求和核心**

- `_accumulate_row(i, ...)`：固定能量版——累加第 $i$ 行（$j \ge i$）到 `acc[k, d]`，非对角项乘 2；`nr[j,d] = \mathbf n_d\cdot\mathbf r_j`、`nb[j,d] = \mathbf n_d\cdot\mathbf b_j` 为预投影。
- `_double_sum_batch(time, w, beta, nr, nb, fac, a, b, kernel_id, block)`：并行上三角批量求和（折行负载均衡，[§3.3](#33-numba-后端backendnumba默认快约-140×)）。
- `_accumulate_row_local(i, ...)` / `_double_sum_batch_local(...)`：局域能量版，多一个 `chi[d]` 能量变化相位项（[§3.3](#33-numba-后端backendnumba默认快约-140×)）。

**公开积分入口**

- `double_time_integral_batch(time, position, beta, omega, dirs, epsilon, mass=1.0, kernel="dot", epsilon_prime=None, phase="recoil", backend="auto", block=None, chunk=256, energy=None)`：主批量入口——对每个 $(\omega_k, \mathbf n_d)$ 对返回 $\mathrm{Re}\iint N e^{-i\Phi}$，形状 $(N_\omega, N_{dir})$。`energy` 给出时切换到局域能量模式（此时 `epsilon`/`epsilon_prime` 不再使用）。所有公开入口（下同）在局域模式下只支持 `dot`/`trace` 核。
- `double_time_integral(time, position, beta, omega, n, epsilon, ..., energy=None)`：单个 $(\omega, n)$ 的**复数**积分 $I$。numpy 后端虚部为舍入诊断；numba 后端虚部构造性为零。
- `d2_energy(time, position, beta, omega, n, epsilon, ..., diagnostics=False, energy=None)`：$d^2E/(d\omega\,d\Omega) = (\alpha/4\pi^2)\,\omega^2 q^2\,\mathrm{Re}\,I$；`diagnostics=True` 时返回 `(值, 虚部)`。
- `d2_probability(...)`：$d^2W/(d\omega\,d\Omega) = d^2E/(d\omega\,d\Omega)/\omega$，同样支持 `diagnostics`。
- `classical_d2_energy(time, position, beta, omega, n, charge=1.0)`：经典 LW 参考谱 $(\alpha/4\pi^2)\,\omega^2\lvert A\rvert^2$（validation 用）。
- `cone_directions(axis, theta_max, n_theta, n_phi)`：锥面方向网格与立体角权重（[§3.4](#34-角度积分)）。
- `_recoil_args(params)`：把 `Parameters.recoil` 映射到积分器的 `(phase, epsilon_prime)` 组合。

**面向用户的包装**

- `BKIntegrator`：捆绑一条轨迹、一组参数与一个后端的便捷类。`__init__(trajectory, params, backend="auto")` 缓存 `time/position/beta/energy`。方法：
  - `d2_energy_batch(omega_grid, dirs)` / `d2_probability_batch(omega_grid, dirs)`：整张 $(N_\omega, N_{dir})$ 网格。
  - `d2_energy(omega, n)` / `d2_probability(omega, n)`：单点。
  - `compute_spectrum(omega_grid, theta_max=None, n_theta=16, n_phi=8, axis=None)`：角积分谱（[§3.4](#34-角度积分)），返回带 metadata 的 `Spectrum`。
- `compute_spectrum(trajectory, params, omega_grid, ...)`：模块级便捷函数（构造 `BKIntegrator` 后直接求谱）。

### 6.6 `reference.py` — 解析参考谱（仅验证用）

均匀圆周运动沿轨迹场强严格恒定，恒定场同步辐射公式就是本模块双时间积分的**解析求值**（Baier & Katkov 1967；Sokolov & Ternov；Ritus 1985），因此可用来检验**实现**（前因子、反冲相位、$\omega^2/\gamma^2$ 项、dot vs trace）而非理论。

- `circle_chi(gamma, rho)`：量子参数 $\chi = \gamma^2\beta^2/\rho$（圆半径 $\rho$）。
- `circle_omega_c(gamma, rho)`：经典临界频率 $\omega_c = \frac{3}{2}\gamma^3\Omega$，$\Omega = \beta/\rho$。
- `recoil_shifted_harmonic(m, Omega, epsilon)`：反冲移动谐波频率

$$\omega_m = \frac{m\Omega}{1 + m\Omega/\varepsilon}$$

  它是 $\omega\varepsilon/(\varepsilon-\omega) = m\Omega$ 的解——一圈 BK 双时间积分在这组频率处被积函数周期化、无截断边界项。物理发射线因此反冲移动，线间距 $\Delta\omega_m = \Omega(\varepsilon'/\varepsilon)^2$；代码在 $\omega_m$ 处求的一圈 $dE/d\omega$ 直接就是连续谱密度 $T\,dP/d\omega$（无 Jacobian）。经典极限 $\varepsilon\to\infty$ 回到 $m\Omega$。
- `quantum_synchrotron_rate(delta, chi, epsilon, form="macdonald")`：量子同步辐射光子数率（单位实验室时间、单位 $\delta = \omega/\varepsilon$）：

$$\frac{dW}{dt\,d\delta} = \frac{\alpha}{\sqrt3\,\pi\,\varepsilon}\Big[\Big(1-\delta+\frac{1}{1-\delta}\Big)K_{2/3}(y) - \int_y^{\infty}\!K_{1/3}(x)\,dx\Big], \qquad y = \frac{2\delta}{3\chi(1-\delta)}$$

  `form="airy"` 给出等价的 Airy 形式（与 `core/qed/optical_depth_tables.py` 的 LCFA 表同一实现）：

$$\frac{dW}{dt\,d\delta} = -\frac{\alpha}{\varepsilon}\Big[\int_z^{\infty}\!\mathrm{Ai}(x)\,dx + \Big(\frac{2}{z} + \delta\chi\sqrt z\Big)\mathrm{Ai}'(z)\Big], \qquad z = \Big(\frac{\delta}{\chi(1-\delta)}\Big)^{2/3}$$

- `classical_synchrotron_rate(delta, chi, epsilon)`：无反冲极限 $\frac{\alpha}{\sqrt3\pi\varepsilon}\int_{y_0}^{\infty}K_{5/3}$，$y_0 = 2\delta/(3\chi)$；对应 Schwinger 谱 $dP/d\omega = \frac{\sqrt3}{2\pi}\frac{\alpha\gamma}{\rho}F(\omega/\omega_c)$。
- `synchrotron_F(x)`：Schwinger 函数 $F(x) = x\int_x^{\infty}K_{5/3}(\xi)d\xi$。
- `synchrotron_power(chi, epsilon, quantum=True)`：辐射功率积分。量子版 $\int_0^1 \varepsilon\delta\,(dW/dt\,d\delta)\,d\delta = \frac{2}{3}\alpha\chi^2 g(\chi)$；经典版积分到 $\infty$ 恒等于 Larmor 功率 $\frac{2}{3}\alpha\chi^2$。
- `erber_g(chi)`：Baier–Katkov–Strakhovenko 量子抑制因子拟合 $g \sim [1 + 4.8(1+\chi)\ln(1+1.7\chi) + 2.44\chi^2]^{-2/3}$。
- `orbit_direction_grid(gamma, n_inner=24, n_outer=8, inner_half_width=None)`：x-y 平面圆轨道方向网格（[§3.4](#34-角度积分)）。
- 内部：`_int_K13`、`_int_K53`、`_int_Ai`（scipy `quad` 无穷限积分，相对容差 $10^{-10}$）。

### 6.7 `result.py` — 结果输出

- `format_spectrum(spectrum)`：人类可读的汇总表（逐行 $\omega, dW/d\omega, dE/d\omega$ + 总概率/总能量 + metadata）。
- `spectrum_table(spectrum)`：$(N_\omega, 3)$ 数组 `[omega, dW/domega, dE/domega]`。
- `save_spectrum_text(spectrum, path)`：三列文本 + 头注释（含单位制与 metadata）。

### 6.8 `units.py` — 单位换算

见 [§4](#4-单位制)。内容：SI 常数（`fine_structure`、`electron_mass`、`electron_mass_energy`）；四个自然单位尺度（`natural_time_unit` 等）；8 个单向换算函数（`si_time_to_natural`/`natural_time_to_si`、`si_length_*`、`si_energy_*`、`si_frequency_*`）；光子能量 eV 换算（`natural_omega_to_photon_energy_ev`/`photon_energy_ev_to_natural_omega`）；批量转换 `si_to_natural(t_si, x_si, p_si, E_si, omega_si)` 与 `natural_to_si(...)`（动量按 $p_{SI}/(m_e c)$ 映射到归一化动量 $\mathbf u$）。

### 6.9 `validation.py` — V1–V8 验证套件

粗粒度、判方向不判精度的自检（任务性质如此，非精度声称）。按验证内容：

| 编号 | 函数 | 验证内容 | 通过标准 |
|---|---|---|---|
| V1 | `check_airy_identity` | Airy 相位积分恒等式（eq. 9.6）与 Macdonald 形式（eq. 9.7/9.8）：相位三次标定与符号体系。半线余弦积分条件收敛，用分步积分尾部修正。注意：对称实积分分不出 $\pm$ 相位符号，符号由 V2/V3 钉死 | $<10^{-3}$ |
| V2 | `check_velocity_kernel_consistency` | $\lvert$经典振幅$\rvert^2$ == velocity 核双时间积分：检验积分器本身与相位符号 $e^{-i\Phi}$ | $<10^{-8}$ |
| V3 | `check_classical_limit` | 闭环上软光子 BK → LW（eq. 7.4），比值 $\to 1-\omega/\varepsilon$；同时展示在裸谐波 $m\Omega$ 处求值的 $O(m\omega/\varepsilon)$ 截断伪影 | 比值趋 1 |
| V4 | `check_larmor` | 绝对归一化：Schwinger 谱积分闭合 == Larmor 功率（纯数学恒等式，钉前因子）；代码在谐波 m=4..64 处 vs 精确谱 $\to 1$ | 闭合 ≈1 |
| V5 | `check_straight_line` | 匀速直线 → 辐射近似零，BK ≈ LW（对照弯曲轨迹同频率辐射 $O(10^2)$） | 比值极小 |
| V6 | `check_positivity` | 闭环反冲移动谐波处角积分 $dW/d\omega$ 严格为正（未裁剪——闭环无边界项，正性是真实检验而非截断） | min > 0 |
| V7 | `check_kernel_rewrite` | dot ≡ trace 在壳逐点相等（硬光子 $\omega/\varepsilon = 0.7$ 处随机方向验证） | $<10^{-12}$ |
| V8 | `check_quantum_synchrotron` | 量子区：一圈 BK 谱（反冲移动谐波处）vs 精确恒定场量子同步辐射谱（$K_{1/3}, K_{2/3}$；即 LCFA 表背后的公式），两核都验 | 比值 0.997–1.001 |

辅助函数：`circle_trajectory(gamma, rho, n_samples, turns)`（x-y 平面均匀圆运动，含归一化动量）、`straight_trajectory(gamma, T, n_samples)`、`harmonics_for_deltas(deltas, Omega, epsilon)`（把目标 $\delta$ 吸附到最近的反冲移动谐波）。`main()` 依次打印 V1–V8 结果与判定。

### 6.10 `validation_plot.py` — 验证汇总图

`main(save_path=None)`：重跑 V1–V8 并渲染成一张 9 面板的 `validation_summary.png`（V3 比值、V4 谐波比值+Larmor 闭合、V5 对数条形、V6 正性谱、V1/V2/V7 相对误差、V8 谱+比值、判定汇总面板）。仅此文件需要 matplotlib（Agg 后端，中文字体 Microsoft YaHei/SimHei）。⚠️ 该 PNG 是二值产物，不要用文本工具打开。

### 6.11 `examples/bk_single_particle.py` — 示例

- `full_circle_trajectory(gamma=10.0, rho=200.0, n_samples=512)`：一圈闭合同步辐射圆轨道（端点位置与归一化动量闭合），返回 `(Trajectory, Omega)`。
- `main()`：在 8 个反冲移动谐波处求全空间角积分谱（x-y 平面轨道绕 z 轴对称，$n_\phi=1$ + Gauss–Legendre $\theta$ 即精确），打印 `format_spectrum` 并保存 `bk_spectrum.txt`。

### 6.12 `__init__.py` / `__main__.py`

- `__init__.py`：导出公共 API——`Trajectory, Parameters, Spectrum, as_trajectory, uniform_trajectory, BKIntegrator, compute_spectrum`，以及子模块 `kernel, phase, units, result, reference`。
- `__main__.py`：`python -m lambdapic.core.qed.baier_katkov` 直接运行上述示例。

---

## 7. 使用示例

**固定能量（闭环同步辐射）**：

```python
import numpy as np
from lambdapic.core.qed.baier_katkov import Trajectory, Parameters, compute_spectrum
from lambdapic.core.qed.baier_katkov.reference import recoil_shifted_harmonic

# 一圈闭合同步辐射圆轨道（x-y 平面），自然单位 c=hbar=m_e=1
gamma, rho, n_samples = 10.0, 200.0, 512
beta = np.sqrt(1.0 - 1.0 / gamma**2)
Omega = beta / rho
T = 2.0 * np.pi / Omega
t = np.linspace(0.0, T, n_samples)
phi = Omega * t
r = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(t)])
u_perp = gamma * beta
u = np.column_stack([-u_perp * np.sin(phi), u_perp * np.cos(phi), np.zeros_like(t)])

traj = Trajectory(time=t, position=r, momentum=u)   # momentum = 归一化动量 u = γβ
params = Parameters(epsilon=gamma)                  # 入射能量 ε = γ·m

# 反冲移动谐波：一圈双时间积分在这些频率处无边界项
omega_grid = recoil_shifted_harmonic(np.arange(1, 9), Omega, gamma)
spectrum = compute_spectrum(traj, params, omega_grid, theta_max=np.pi, n_phi=1,
                            axis=np.array([0.0, 0.0, 1.0]))
print(spectrum.total_probability(), spectrum.total_energy())
```

**局域能量（PIC 加速电子轨迹）**：给 `Trajectory` 传能量历史即可，其余 API 不变：

```python
traj = Trajectory(time=t, position=r, momentum=u, energy=eps_t)  # eps_t: (Nt,)，正
# 要求 eps_t 与 u 在壳一致：eps_t == mass * np.sqrt(1 + |u|²)，且所有 omega < min(eps_t)
spec = compute_spectrum(traj, params, omega_grid, ...)
```

---

## 8. 运行环境与常用命令

- 解释器：`d:/Desktop/lambdapic/.venv/Scripts/python.exe`（Python 3.11，已 `pip install -e ".[test]"`）
- 环境变量：`PYTHONUTF8=1`、`SETUPTOOLS_SCM_PRETEND_VERSION=0.15.1`
- 工作目录：仓库根 `d:/Desktop/lambdapic/lambdapic`（`pytest.ini` 在此，`pythonpath = src`）
- Windows 上 C 扩展未编译（`Simulation.run()` 不可用），但本模块是纯 numpy/scipy/numba，完全可运行；完整 PIC 模拟需 Linux/WSL。

```bash
python -m lambdapic.core.qed.baier_katkov.validation            # V1–V8 验证套件，约 17 s
python -m lambdapic.core.qed.baier_katkov.validation_plot       # 重新生成 validation_summary.png
python -m lambdapic.core.qed.baier_katkov                       # 运行单粒子示例
python -m pytest tests/test_baier_katkov.py -n 0 -q             # 全量回归（含慢速全谱扫描）
python -m pytest tests/test_baier_katkov.py -n 0 -q -m "not slow"   # 跳过全谱扫描，约 10 s
```

---

## 9. 验证与回归测试

- **V1–V8 套件**（[§6.9](#69-validationpy--v1v8-验证套件)）约 17 s，覆盖相位符号、积分器自洽、经典极限、绝对归一化（Larmor）、直线零辐射、谱正性、核恒等式、量子同步辐射谱对照。当前全部通过。
- **pytest 回归**（`tests/test_baier_katkov.py`）共 35 例：
  - 固定能量路径 23 例（含 BUG 修复回归、两后端一致性、V8 类对照、慢速全谱扫描 2 例）；
  - 局域能量路径 12 例（2026-09-09 新增，7 个测试函数）：恒能量退化回固定路径（$10^{-16}$）、局域 dot ≡ trace、numba ≡ numpy（$10^{-15}$）、厄米性（虚部 $10^{-16}$）、谱正性、经典模式能量无关（$10^{-12}$）、入参校验；测试轨迹为精确闭合、在壳、能量变化的圆轨道（$\gamma$ 扫 7→13）。

---

## 10. 已知限制与路线图

按 `REVIEW_AND_ROADMAP.md` 的编号：

- **P-3 有限记录边界辐射（未解决，PIC 应用的拦路虎）**：双时间积分对**硬截断**的有限轨迹产生端点伪影（非谐波频率处谱可为负、角积分谱被边界项污染）。闭环 + 反冲移动谐波处边界项消失（validation V3/V6 利用这点），但一般 PIC 轨迹需要形成长度窗口或渐近补零（路线图 §7-B P1）。
- **P-4 默认锥角**：`compute_spectrum` 默认 $\theta_{max} = 5/\gamma$ 会漏掉高 $\omega$ 尾巴（反冲使大 $\delta$ 处发射锥变宽）；闭环验证中显式用全空间。
- **性能**：$O(N_t^2)$ 双重和；numba 化已完成（约 140×），带状截断 $O(N_t N_\omega)$（§7-C P2）未实现。
- **PIC 集成未做**（§7-D P3/P4）：单位适配器、轨迹记录回调、宏粒子权重系综求和、在线事件采样、自旋分辨核、NUFFT。
- **文档债务（§3 P0，部分缓解）**：代码 docstring 引用的设计文档 `Baier-Katkov.md`（eq. 5.3/7.1/7.3/9.6、sec. 4.2 等编号）**尚不存在**——本 README 补齐了架构与函数说明，但物理推导与文献出处的设计文档仍待写；局域能量（sec. 4.2）的推导记录在 `REVIEW_AND_ROADMAP.md` §4 P-2。
- **调用方责任**：能量历史与动量历史的在壳一致性、轨迹不含 LCFA 反冲（[§1](#1-模块定位与范围)）。

---

## 11. 相关文档

- [REVIEW_AND_ROADMAP.md](REVIEW_AND_ROADMAP.md)：模块审查报告与 PIC 集成路线图（跨会话事实来源；新会话先读其 §9 行动清单）
- [SESSION_SUMMARY_2026-09-08.md](SESSION_SUMMARY_2026-09-08.md)：上一工作会话的详细记录
- [../../../../../tests/test_baier_katkov.py](../../../../../tests/test_baier_katkov.py)：回归测试套件
- [validation_summary.png](validation_summary.png)：V1–V8 验证汇总图（运行 `validation_plot` 重新生成）
