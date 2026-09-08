# Baier–Katkov 模块：2026-09-07/08 工作总结与验证体系说明

> 本文是对这两天在 [baier_katkov/](lambdapic/src/lambdapic/core/qed/baier_katkov/) 子包上所做工作的完整记录，以及对 [validation.py](lambdapic/src/lambdapic/core/qed/baier_katkov/validation.py)（V1–V8 sanity 检查）和 [tests/test_baier_katkov.py](lambdapic/tests/test_baier_katkov.py)（23 个 pytest 用例）逐项的"测什么、为什么、物理意义是什么"的说明。审查结论与路线图另见 [REVIEW_AND_ROADMAP.md](lambdapic/src/lambdapic/core/qed/baier_katkov/REVIEW_AND_ROADMAP.md)。
>
> 单位约定贯穿全文：Compton 自然单位 $c=\hbar=m_e=1$，电子能量 $\varepsilon=\gamma$，光子能量分数 $\delta=\omega/\varepsilon$，末态能量 $\varepsilon'=\varepsilon-\omega$。

---

## 第一部分：这次任务做了什么

### 0. 起点

模块是一个"物理优先、性能为次"的单粒子 Baier–Katkov（BK）参考实现：对一条无反冲经典轨迹 $(t_i,\mathbf r_i,\mathbf u_i)$ 直接做双时间积分

$$\frac{d^2E}{d\omega\,d\Omega}=C\,\omega^2 q^2\,\mathrm{Re}\!\int\!\!\!\int dt_1dt_2\,N(t_1,t_2)\,e^{-i\Phi(t_1,t_2)},\qquad \Phi=\frac{\varepsilon}{\varepsilon'}\,\omega\big[(t_2-t_1)-\mathbf n\cdot(\mathbf r_2-\mathbf r_1)\big]$$

得到单粒子谱，与在线 LCFA 蒙特卡洛管线完全解耦。任务要求：审查错误与可优化点，给出走向完整 PIC 模拟的路线图。

### 1. 审查（第一轮）

通读全部 10 个源文件 + 主模拟的 QED 管线（`radiation.py`、`optical_depth.py`、`inline.py`、`simulation.py`），运行原有 V1–V7，复现疑点。产出 REVIEW_AND_ROADMAP.md。发现：

- **4 个代码缺陷**（BUG-1~4）：`Parameters.recoil` 从未传入计算路径（`recoil="classical"` 被静默忽略且 metadata 谎报）；`kernel` 字符串不校验（`"foo"` 静默按 trace 算）；$\omega\ge\varepsilon$ 时裸 `ZeroDivisionError` / 非物理结果；单样本轨迹 `IndexError`。
- **文档债务**：所有 "eq. 7.1 / sec. 15" 引用指向仓库中不存在的三个设计文档。
- **3 个物理缺口**：量子项从未验证（P-1）、固定 $\varepsilon$ 不适用 PIC（P-2）、有限记录边界辐射（P-3）、默认锥角漏高 $\omega$ 尾巴（P-4）。
- 主模拟为 SI 单位制，`units.py` 已有换算桥；粒子数据结构（`x,y,z,w,ux,uy,uz,_id`）具备轨迹记录所需全部字段。

### 2. 修复 BUG-1~4 与次级问题

- `BKIntegrator` 新增 `_recoil_args()`，把 `recoil` 映射为 `(phase, epsilon_prime)`；`spin_averaged/polarization_summed=False` 抛 `NotImplementedError`（消灭"设了没用"的字段）。
- `kernel` 白名单校验，`"velocity"` 写入文档，dispatch 重构为显式三分支。
- $\omega<\varepsilon$ 守卫（integrator、`phase.recoil_frequency`、`Parameters.epsilon_prime`）。
- `trapz_weights` 单样本返回 `[0.]`，`Trajectory` 拒绝 $N_t<2$ 和非 1-D `time`；`Spectrum` 校验 `dE_domega` 形状。
- `d2_energy/d2_probability` 新增 `diagnostics=True` 返回虚部（兑现 docstring 承诺）；移除 `kernel.py` 里无用的 `fine_structure` 重导出。

### 3. 理论对照问题 → 发现前因子差 $4\pi$（BUG-5）

回答"该和什么公式比较"时，用标准 Schwinger 经典同步辐射谱反算，发现代码所有绝对量比标准大 $4\pi=12.57$ 倍：`units.py` 把 Heaviside–Lorentz 的 $e^2=4\pi\alpha$ 代进了 Jackson 的高斯制公式 $\frac{e^2\omega^2}{4\pi^2 c}|\ldots|^2$，得到 $\alpha/\pi$；正确前因子为 $C=\alpha/(4\pi^2)$。原 V4 的 Larmor 参考碰巧带同一错误，所以"通过"了。四条证据：Jackson 14.67 推导、每圈能量损失 88.5 keV·$E^4/\rho$ 标定、数值 code/standard = 12.45→$4\pi$、与项目 LCFA 表归一化不一致。修复后 V4 比值不变（自洽），V6 的总概率由 0.831 变为 0.0661（恰 $1/4\pi$）。

### 4. Numba 后端

`integrator.py` 重写为双后端：`"numpy"` 分块外积参考路径保留为真值；`"numba"`（默认）`@njit(parallel=True, cache=True)`，显式利用厄米对称性 $K(t_2,t_1)=K(t_1,t_2)^*$ 只算上三角，把一个谱的全部 $(\omega_k,\mathbf n_d)$ 对合并为对 $(t_1,t_2)$ 的一次遍历（核 $N=a_k+b_k x_{ij}$ 在方向间共享，相位宗量 $\psi_{ij,d}=\Delta t-\mathbf n_d\cdot\Delta\mathbf r$ 在频率间共享），行按 $i\leftrightarrow N_t-1-i$ 折叠配对以平衡 `prange` 负载。新增批量接口 `double_time_integral_batch` / `BKIntegrator.d2_energy_batch`。两后端一致到尺度相对 $10^{-14}$；示例 4.2 s → 0.03 s（约 140×），V1–V7 60 s → 1.3 s，$N_t=4000$ 时 0.62 G(对×组合)/s。

### 5. P-1：量子同步辐射交叉验证（本次任务核心）

新建 [reference.py](lambdapic/src/lambdapic/core/qed/baier_katkov/reference.py)（恒定场精确谱、反冲移动谐波、角度网格），validation 新增 V8，pytest 新增 10 例。结果：**dot 核与精确量子谱一致到 0.1–0.3%**（$\gamma=10$），$\gamma=20$ 时 0.07%，残差按 $1/\gamma^2$ 收敛；覆盖 $\chi\in\{0.1,0.5,2\}$、$\delta$ 到 0.7，而同区间量子/经典比值从 0.98 降到 0.38。反冲相位、$(\varepsilon^2+\varepsilon'^2)/2\varepsilon'^2$ 前因子、$\omega^2/\gamma^2$ 项全部正确。

顺带两项发现：

- **BUG-6，trace 核抄错**：$(\mathbf b_1-\mathbf b_2)^2$ 系数应为 $\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon'^2}$，原来是 $\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon\varepsilon'}$（少一个 $\varepsilon/\varepsilon'$），对精确谱偏低 6%→70%。修正后 trace 与 dot **逐点恒等**（$2\times10^{-16}$）；原 docstring 里"两核相差 $O(\omega)$、trace 需真空扣除"整套说法均为此笔误的衍生物，已重写。
- **反冲移动谐波**：闭合轨道上一圈 BK 积分只有在 $\omega\varepsilon/\varepsilon'=m\Omega$ 即 $\omega_m=\dfrac{m\Omega}{1+m\Omega/\varepsilon}$ 处才是周期的（无截断项）。在裸 $m\Omega$ 处评估误差达 ±40%；V3 原先随 $m$ 增长的 0.1%→3% "偏离"正是此伪影，改到 $\omega_m$ 后比值精确落在 $1-\omega/\varepsilon$ 上。V3、V6、示例均已改用 $\omega_m$。

### 6. 文件变更清单

| 文件 | 变更 |
|---|---|
| `integrator.py` | 重写：双后端、批量接口、`PREFACTOR=α/4π²`、校验与守卫、`diagnostics`、`_recoil_args` |
| `kernel.py` | trace 核系数修正；模块文档重写；移除 `fine_structure` 重导出 |
| `phase.py` | `recoil_frequency` 校验 $\varepsilon'>0$ |
| `types.py` | `Trajectory`/`Spectrum` 形状校验；`Parameters` 自旋标志与 $\varepsilon'$ 守卫；kernel 文档 |
| `units.py` | 前因子注释改正 |
| `reference.py` | **新建**：精确谱、经典谱、功率、$g(\chi)$、$\omega_m$、角度网格 |
| `validation.py` | V3 改 $\omega_m$ 并保留裸值；V4 参考去 $4\pi$；V6 改 $\omega_m$；V7 改精确恒等式；**新增 V8** |
| `validation_plot.py` | 3×3 布局，新增 V8 谱图与比值图，V3 双线 |
| `examples/bk_single_particle.py` | 改用 $\omega_m$ |
| `__init__.py` | 导出 `reference` |
| `README.md` | 前因子、双后端、闭合轨道谐波说明、文件表 |
| `REVIEW_AND_ROADMAP.md` | 审查报告与路线图（持续更新） |
| `tests/test_baier_katkov.py` | **新建**：23 例 |
| `validation_summary.png` | 重新生成 |

### 7. 如何运行

```bash
# 在 lambdapic 仓库根目录，使用项目专用 venv
python -m lambdapic.core.qed.baier_katkov.validation        # V1–V8，约 17 s
python -m lambdapic.core.qed.baier_katkov.validation_plot   # 生成 validation_summary.png
python -m pytest tests/test_baier_katkov.py -n 0            # 23 例，约 33 s
python -m pytest tests/test_baier_katkov.py -m "not slow"   # 跳过全谱扫描，约 10 s
```

---

## 第二部分：validation.py —— V1 到 V8 各测什么

`validation.py` 是"粗粒度、看方向"的 sanity 检查：每项打印数值，由人判断；它不设硬阈值（V1/V2/V7 除外），目的是把模块的物理约定一项项钉牢。以下按依赖顺序说明。

### 通用工具

- `circle_trajectory(gamma, rho, n_samples)`：x-y 平面匀速圆周运动一整圈，$\beta=\sqrt{1-1/\gamma^2}$，$\Omega=\beta/\rho$，$T=2\pi/\Omega$，同时给出解析动量 $\mathbf u=\gamma\beta(-\sin\phi,\cos\phi,0)$（避免数值微分）。**为什么用闭合圆环**：它是唯一同时满足"场沿轨迹严格恒定"（LCFA 精确成立，有解析参照）和"无截断边界项"（周期性）的构型。
- `straight_trajectory`：匀速直线，零辐射的参照。
- `harmonics_for_deltas(deltas, Ω, ε)`：把目标 $\delta$ 换成最近的谐波序号 $m=\mathrm{round}\big(\frac{\delta}{1-\delta}\frac{\varepsilon}{\Omega}\big)$。

### V1 · Airy 相位恒等式（`check_airy_identity`）

**测什么**：$\int_{-\infty}^\infty e^{-i(a\tau+b\tau^3)}d\tau=2\pi(3b)^{-1/3}\mathrm{Ai}\big(a/(3b)^{1/3}\big)$，以及 Macdonald 关系 $\mathrm{Ai}(x)=\frac1\pi\sqrt{x/3}\,K_{1/3}(\tfrac23x^{3/2})$、$\mathrm{Ai}'(x)=-\frac{x}{\pi\sqrt3}K_{2/3}(\tfrac23x^{3/2})$。

**物理意义**：BK 方法在恒定场中把相位展开到 $\tau^3$（形成长度内轨迹的三次相位），双时间积分就化为 Airy 函数——这是同步辐射谱里 $K_{1/3},K_{2/3}$ 的来源。此检查确认相位符号约定 $e^{-i\Phi}$ 下的三次重标定与 scipy 的 Airy/Bessel 约定一致。数值上半直线余弦积分条件收敛，用截断 + 分部积分尾项 $-\sin g(L)/g'(L)$ 稳定求值。**结果** 6.9e-7；Macdonald 关系 < 1e-10。

### V2 · 速度核双积分 ≡ $|A|^2$（`check_velocity_kernel_consistency`）

**测什么**：经典 Liénard–Wiechert 振幅 $\mathbf A=\int dt\,\mathbf n\times(\mathbf n\times\boldsymbol\beta)e^{i\omega(t-\mathbf n\cdot\mathbf r)}$ 的 $|\mathbf A|^2$，与用经典横向速度核 $N=\boldsymbol\beta_1\!\cdot\!\boldsymbol\beta_2-(\mathbf n\!\cdot\!\boldsymbol\beta_1)(\mathbf n\!\cdot\!\boldsymbol\beta_2)$ 做双时间积分的结果相等。

**物理意义**：$|\int f|^2=\int\!\!\int f_1 f_2^*$ 是纯数学恒等式，但它同时验证了三件事：积分器的梯形权重与外积组织正确；双时间相位 $\Phi=\omega[(t_2-t_1)-\mathbf n\cdot(\mathbf r_2-\mathbf r_1)]$ 的差分构造（避免大数相消）正确；**$e^{-i\Phi}$ 的符号与单时间振幅的 $e^{+i\omega(\cdots)}$ 互为共轭**——符号反了这个恒等式就不成立。**结果** 2.5e-12。

### V3 · 软光子极限 BK → LW（`check_classical_limit`）

**测什么**：$\gamma=10$、$\rho=200$（$\chi=0.5$）圆环，谐波 $m=1..4$（$\omega/\varepsilon\approx0.0005$–$0.002$），全球面角积分后 BK（dot 核 + 反冲相位）与经典 LW 谱的比值。BK 在反冲移动谐波 $\omega_m$ 处评估，LW 在 $m\Omega$ 处评估；同时报告 BK 在裸 $m\Omega$ 处的比值 `ratio_bare`。

**物理意义**：量子谱在 $\omega/\varepsilon\to0$ 时必须回到经典谱，且领头修正是 $-\delta$（量子/经典 $\approx1-\delta$，来自反冲对相空间的压缩）。**结果**：$\omega_m$ 处比值 0.9995、0.9990、0.9985、0.9980，精确落在 $1-\omega/\varepsilon$ 上；裸 $m\Omega$ 处 0.999→1.030——这条曲线保留下来是为了**记录截断伪影**：相位失配仅 $2\pi m\,\omega/\varepsilon\approx0.05$ rad 就造成 3% 误差，是 P-3（PIC 截断轨迹）问题严重性的直接量化。

### V4 · Larmor 绝对归一化（`check_larmor`）

**测什么**：$\gamma=30$、$\rho=10^4$（$\chi=0.09$，经典区）圆环。(a) 闭合：Schwinger 谱 $\frac{dP}{d\omega}=\frac{\sqrt3}{2\pi}\frac{\alpha\gamma\beta}{\rho}F(\omega/\omega_c)$，$F(x)=x\int_x^\infty K_{5/3}$，对 $\omega$ 积分应等于 Larmor 功率 $P=\frac23\alpha\gamma^4\beta^4/\rho^2$（纯数学恒等式，与代码无关，用来锁定参考公式本身的归一化）。(b) 代码在谐波 $m=4..64$ 处的角积分 $dE/d\omega$ 与 $T\cdot dP/d\omega$ 的比值。

**物理意义**：这是模块**绝对归一化**的唯一独立标定——它决定输出的光子数、辐射能量是否可信。$\frac23\alpha\gamma^4/\rho^2$ 对应实验上熟知的每圈能量损失 88.5 keV·$E[\mathrm{GeV}]^4/\rho[\mathrm m]$。BUG-5 就是在此暴露：原参考带 $4\pi$ 错误与代码相互掩护。**结果**：闭合 0.996（$\omega$ 网格截断所致）；谐波比值 0.946→0.991 随 $m$ 单调趋 1——低次谐波偏低是 Schott 离散谐波与连续谱近似的固有差异，非代码问题。

**为何不在连续 $\omega$ 上做梯形积分**：硬截断圆环在非谐波频率的谱被边界/混叠项主导，会高估几个量级；只有谐波处是干净的。

### V5 · 直线轨迹 ≈ 零辐射（`check_straight_line`）

**测什么**：$\gamma=50$ 匀速直线，$\omega=1$，一个斜方向 $\mathbf n$；BK 与 LW 的 $d^2E/d\omega d\Omega$，以及同频率下弯曲轨迹的参照值。

**物理意义**：匀速运动不辐射。但有限记录的硬截断端点会产生"边界辐射"（等价于粒子突然出现/消失），所以不是严格零；检查的是它比弯曲参照小许多个量级，且 BK 与 LW 都如此。$\omega\Delta t\ll1$ 保证不是混叠。**结果** $|BK/弯曲|=9.8\times10^{-10}$。

### V6 · 角积分谱正性（`check_positivity`）

**测什么**：示例配置（$\gamma=10$、$\rho=200$、$N_t=512$），前 8 个反冲移动谐波，全球面角积分 $dW/d\omega$ 的最小值与总概率。

**物理意义**：单粒子谱是发射概率密度，必须非负；BK 双时间积分的被积函数本身有正有负，正性只在角积分后、无边界项时成立。闭合轨道 + $\omega_m$ 是"无边界项"的构型，所以这里的正性是真检验而非裁剪（四分之一弧在非谐波频率会出现负值，见 P-3）。总概率 $W=0.066$ 与经典估算一致（每圈光子数 $\frac{5\pi}{\sqrt3}\alpha\gamma\approx0.66$，前 8 个谐波占其中约一成）。**结果** min $+1.12$。

### V7 · trace 核 ≡ dot 核（`check_kernel_rewrite`）

**测什么**：随机在壳速度对（$|\boldsymbol\beta|=\sqrt{1-1/\gamma^2}$），硬光子 $\omega/\varepsilon=0.7$，两种核逐点相对差 < 1e-12。

**物理意义**：BK 自旋平均、极化求和的顶点因子有两种等价写法——迹形式 $-\frac{m^2}{\varepsilon\varepsilon'}-\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon'^2}(\mathbf b_1-\mathbf b_2)^2$ 与点积形式 $\frac{\varepsilon^2+\varepsilon'^2}{2\varepsilon'^2}(\mathbf b_1\!\cdot\!\mathbf b_2-1)+\frac{\omega^2}{2\varepsilon'^2\gamma^2}$，由 $(\mathbf b_1-\mathbf b_2)^2=2-2/\gamma^2-2\mathbf b_1\!\cdot\!\mathbf b_2$ 连接，常数项合并为 $\frac{(\varepsilon-\varepsilon')^2m^2}{2\varepsilon^2\varepsilon'^2}=\frac{\omega^2}{2\varepsilon'^2\gamma^2}$。原 V7 只在 $\omega=10^{-3}$ 软极限以 1% 容差检验，因此没能发现 BUG-6；现在在硬光子处以机器精度检验。只有将来做局域能量核（$\varepsilon_1\ne\varepsilon_2$）时两者才会分道，届时含 $b_i^2=1-m^2/\varepsilon_i^2$ 的迹形式是基本形式。**结果** 6.2e-16。

### V8 · 量子同步辐射交叉验证（`check_quantum_synchrotron`）

**测什么**：$\gamma=10$、$\chi=0.5$（$\rho=\gamma^2\beta^2/\chi=198$），目标 $\delta\in\{0.05,0.1,0.2,0.3,0.5\}$ 各取最近的反冲移动谐波 $\omega_m$；一圈 BK 双积分（dot 与 trace 两核）角积分后的 $dE/d\omega$，与精确恒定场量子同步辐射谱 $T\cdot\frac{dP}{d\omega}$ 的比值。同时打印量子/经典比值以显示"这确实是量子区间"。

**参照公式**（`reference.quantum_synchrotron_rate`，Baier–Katkov 1967 / Sokolov–Ternov / Ritus 1985，$y=\frac{2\delta}{3\chi(1-\delta)}$）：

$$\frac{dW}{dt\,d\delta}=\frac{\alpha}{\sqrt3\,\pi\,\varepsilon}\left[\Big(1-\delta+\frac{1}{1-\delta}\Big)K_{2/3}(y)-\int_y^\infty K_{1/3}(x)\,dx\right],\qquad\frac{dP}{d\omega}=\delta\,\frac{dW}{dt\,d\delta}$$

**物理意义**——这是整个模块最关键的一项：

1. *为什么它是正确的参照*：匀速圆周运动中场沿轨迹严格恒定，恒定场公式是同一个 BK 双时间积分的**解析结果**（不是另一套理论），因此比较的是实现而非理论，且 LCFA 在此没有"局域恒场近似是否成立"的模糊性。它也正是项目 `optical_depth_tables.py` 所用的公式，验证通过即建立了 BK 与在线 LCFA 的桥。
2. *检验了哪些量子要素*：$\delta=0.5$ 时量子/经典 = 0.44，也就是说反冲相位因子 $\varepsilon/\varepsilon'$、前因子 $(\varepsilon^2+\varepsilon'^2)/2\varepsilon'^2$、$\omega^2/\gamma^2$ 项中任何一个错一个因子，比值都会偏离百分之几十——此前它们从未被检验过（V3/V4 只覆盖 $\delta<0.002$）。
3. *为什么在 $\omega_m$ 处比较且不需要雅可比*：一圈积分的被积函数以 $\tilde\omega=\omega\varepsilon/\varepsilon'$ 为周期，$\tilde\omega T=2\pi m$ 时无边界项。谱线在 $\omega$ 上的间距为 $\Delta\omega_m=\Omega(\varepsilon'/\varepsilon)^2$，每圈线能量 $E_m=\Delta\omega_m\cdot\frac{d^2E}{d\omega}$，两个 $(\varepsilon'/\varepsilon)^2$ 相消，代码一圈的 $dE/d\omega|_{\omega_m}$ 直接就是连续谱密度 $T\,dP/d\omega$。
4. *采样数*：双积分中最快的相位变化率是 $2\tilde\omega$（速度反平行于 $\mathbf n$ 处），梯形和在 $2\tilde\omega\Delta t>2\pi$ 时混叠，故 $N_t>2m_{max}$；默认取 2 倍余量。角度求积用 `orbit_direction_grid`：$\psi=\theta-\pi/2$ 在 $[0,8/\gamma]$（同步辐射锥）与 $[8/\gamma,\pi/2]$ 两段 Gauss–Legendre，利用方位对称与镜面对称。
5. *残差的意义*：参照公式是超相对论渐近式，误差 $O(1/\gamma^2)$；$\gamma$ 10→20 时残差 0.3%→0.07% 正是这个标度，$N_t$ 与角度网格加倍不改变结果，说明数值已收敛、剩余差异属于参照而非代码。

**结果**：dot 0.9971、0.9993、1.0006、1.0010、1.0007；trace 逐位相同。

---

## 第三部分：test_baier_katkov.py —— 23 个用例各守护什么

pytest 用例是 validation 的"硬阈值版"，任何一条失败都意味着某个已钉牢的约定被破坏。按主题分组：

### A. 数值实现正确性（8 例）

| 用例 | 内容 | 守护的约定 |
|---|---|---|
| `test_numba_matches_numpy` ×6 | 3 种核 × {反冲相位, 经典相位}，4 个 $\omega$ × 3 个方向，numba 批量结果与 numpy 参考路径逐点相差 < 1e-12·max\|ref\| | numba 内核（上三角对称、方向/频率批量、折叠并行）与参考双重求和完全等价。用尺度相对而非逐点相对，因为闭合圆环在某些 $(m,\mathbf n)$ 处有精确零 |
| `test_full_square_is_real` | numpy 全方阵路径的虚部 < 1e-10·实部 | $K(t_2,t_1)=K(t_1,t_2)^*$ 对称性成立，积分为实数——这是 numba 路径只算上三角的前提 |
| `test_velocity_kernel_equals_amplitude_squared` | V2 的 pytest 版，1e-8 | 积分器组织与 $e^{-i\Phi}$ 符号 |

### B. 绝对归一化与核（3 例）

| 用例 | 内容 | 守护的约定 |
|---|---|---|
| `test_prefactor_constant` | `PREFACTOR == α/(4π²)` | BUG-5 不复发 |
| `test_absolute_normalization_schwinger` | $\gamma=30$、$\rho=10^4$、$m=64$，经典相位 BK 的角积分 $dE/d\omega$ 与 $T\cdot$Schwinger 谱之差 < 3% | 绝对归一化对标 88.5 keV/圈 的标准；走的是 `BKIntegrator(recoil="classical")` 路径，同时覆盖 BUG-1 的修复 |
| `test_trace_kernel_equals_dot_kernel_on_shell` | 三组 $(\varepsilon,\omega)$ 含 $\delta=0.7$ 硬光子，两核 rtol 1e-12 | BUG-6 不复发；两核为同一函数 |

### C. 解析参照的三重锚定（5 例）

参照公式若错，V8 的通过就毫无意义，所以先独立锚定它。

| 用例 | 内容 | 守护的约定 |
|---|---|---|
| `test_reference_macdonald_airy_and_lcfa_table_agree` ×3 | $\chi\in\{0.05,0.5,2\}$，$\delta\in[0.01,0.9]$：Macdonald 形式、Airy 形式、项目 `gen_photon_prob_rate_for_delta(chi)(delta)·t_{unit}/\gamma` 三者相差 < 1e-5·峰值 | 我写的参照与项目 LCFA 表是同一个函数（LCFA 表是每单位固有时的 SI 速率，乘 $\hbar/mc^2$ 再除 $\gamma$ 化为实验室时间自然单位）；Airy↔Macdonald 换算恒等式 $2+\delta\chi z^{3/2}=(1-\delta)+\frac1{1-\delta}$ 正确 |
| `test_reference_classical_limit_and_power` | (i) $\chi=10^{-3},10^{-4}$，固定 $y_0=0.2$ 时量子/经典 → 1（偏差 < $3\chi$）；(ii) $\delta\cdot$经典速率 ≡ Schwinger 的 $\frac{\sqrt3}{2\pi}\frac{\alpha\gamma\beta^2}{\rho}F(y_0)$ 到 1e-8；(iii) 经典功率 ≡ Larmor $\frac23\alpha\chi^2$ 到 1e-6；量子功率/Larmor $=g(\chi)\in(0,1)$ 且与 BKS 拟合 $[1+4.8(1+\chi)\ln(1+1.7\chi)+2.44\chi^2]^{-2/3}$ 相差 < 3% | 参照公式的经典极限、归一化、以及总功率的量子压低（$\chi=1$ 时 $g=0.18$，即辐射反作用被压低 5 倍——这是"量子相关结果"最直观的一个数） |
| `test_direction_grid_covers_sphere` | 权重和 $=4\pi$，方向单位模 | 角度求积完备 |

### D. 量子区间交叉验证（4 例）

| 用例 | 内容 | 守护的约定 |
|---|---|---|
| `test_quantum_synchrotron_quick` | $\chi=0.5$，$\delta=0.3,0.5$（量子/经典 < 0.75），dot 与 trace 两核，容差 1%，约 7 s | P-1 的最小守护：反冲相位、前因子、$\omega^2/\gamma^2$ 项 |
| `test_quantum_synchrotron_spectrum[0.5]` (slow) | $\delta=0.05$–$0.5$ 全扫描，1% | 软尾到硬光子的整条谱形 |
| `test_quantum_synchrotron_spectrum[2.0]` (slow) | $\chi=2$，$\delta$ 到 0.7，2%（$\gamma=10$ 时 $\rho=50$，$1/\gamma^2$ 效应更显著） | 深量子区 $(1-\delta)^{-1}$ 项主导时仍正确 |
| `test_bare_harmonic_evaluation_is_wrong` | $\delta=0.1,0.2$：在 $\omega_m$ 处 < 1%，在裸 $m\Omega$ 处存在 > 20% 偏差 | 把"陷阱"固化为断言：谁若把评估点改回 $m\Omega$，测试立刻失败；同时是 P-3 边界效应量级的回归记录 |

### E. 接口约定与输入校验（3 例）

| 用例 | 内容 | 守护的约定 |
|---|---|---|
| `test_recoil_option_is_honoured` | `Parameters(recoil="classical")` 与显式 `phase="classical", epsilon_prime=ε` 逐位一致，且与 BK 反冲结果不同 | BUG-1 不复发 |
| `test_spectrum_consistency` | `compute_spectrum` 输出 $dE=\omega\,dW$、闭合轨道 $dW>0$、metadata 如实 | 谱对象自洽；V6 正性 |
| `test_invalid_input_rejected` | 非法 kernel/phase/backend、$\omega\ge\varepsilon$、$N_t=1$、非法 recoil、`spin_averaged=False` 各抛预期异常 | BUG-2/3/4 与次级问题不复发 |

---

## 第四部分：现在可以相信什么、还不能相信什么

**已钉牢**（有硬测试守护）：相位符号；双时间积分器；两后端等价；绝对归一化 $\alpha/4\pi^2$；两种核恒等；反冲相位与全部量子项在恒定场中到 $O(1/\gamma^2)$；闭合轨道谐波的正确评估点。

**尚未覆盖**（路线图 §7）：
- **P-3 有限记录边界项**：任何非闭合、非周期评估点的轨迹目前都带 $O(m\omega/\varepsilon)$ 级伪影，PIC 轨迹尤甚——需要形成长度窗口/绝热开关，这也是把复杂度从 $O(N_t^2)$ 降到 $O(N_tN_\omega)$ 的同一件事。
- **P-2 局域能量** $\varepsilon(t)$：现在整条轨迹用单一 $\varepsilon$；能量沿轨迹变化 $O(1)$ 的 PIC 场景需要 $\varepsilon_1,\varepsilon_2$ 版本的迹核。
- **P-4 自适应锥角**：`compute_spectrum` 默认 $5/\gamma$ 锥在高 $\omega$ 处漏尾巴（V8 用的是自建全球面网格，不受此影响）。
- 变场、非圆轨道、自旋/极化分辨、系综加权、与 `Simulation` 的对接——均未开始。
