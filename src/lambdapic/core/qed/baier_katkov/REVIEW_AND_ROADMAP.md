# Baier–Katkov 模块审查报告与 PIC 集成路线图

> 审查日期：2026-09-07；更新：2026-09-08（§2 缺陷全部修复、前因子 4π 修正、Numba 后端、pytest 回归）。范围：[baier_katkov/](lambdapic/src/lambdapic/core/qed/baier_katkov/) 全部源码（约 1400 行）+ 主模拟 QED 管线（[radiation.py](lambdapic/src/lambdapic/core/qed/radiation.py)、[optical_depth.py](lambdapic/src/lambdapic/core/qed/optical_depth.py)、[inline.py](lambdapic/src/lambdapic/core/qed/inline.py)、[simulation.py](lambdapic/src/lambdapic/simulation/simulation.py)）。
> 所有"已确认"的缺陷都在本机（lambdapic 专用 venv，Python 3.11.9）上实际运行复现过。

---

## 1. 现状总结

模块是一个**物理优先、性能为次**的单粒子参考实现，定位清晰：

- 对一条**无反冲经典轨迹** $(t_i, r_i, u_i)$，直接做双时间积分，给出单粒子微分谱：

$$ \frac{d^2E}{d\omega\, d\Omega} = \frac{\alpha}{\pi}\,\omega^2 q^2\,\mathrm{Re}\int\!\!\!\int dt_1 dt_2\, N(t_1,t_2)\, e^{-i\Phi(t_1,t_2)}, \qquad \frac{d^2W}{d\omega\, d\Omega} = \frac{1}{\omega}\frac{d^2E}{d\omega\, d\Omega} $$

- 核 $N$ 有严格迹形式（eq. 7.1）与速度/点积形式（eq. 7.3）两个版本；相位 $\Phi = (\varepsilon/\varepsilon')\,\omega[(t_2-t_1) - n\cdot(r_2-r_1)]$ 含 BK 反冲因子（eq. 5.3）。
- 单位制为 Compton 自然单位（$c=\hbar=1$，$m_e=1$），`units.py` 提供 SI 双向换算。
- 角度积分用初始速度方向附近的 Gauss–Legendre 锥积分。

**验证套件 V1–V7 全部通过**（`python -m lambdapic.core.qed.baier_katkov.validation`，实际运行结果）：

| 检查 | 内容 | 结果 |
|---|---|---|
| V1 | Airy 相位恒等式（eq 9.6） | rel_err = 6.9e-7 ✓ |
| V2 | $\|A\|^2$ = 速度核双时间积分（验证积分器与 $\exp(-i\Phi)$ 相位符号） | rel_err = 2.5e-12 ✓ |
| V3 | 软光子 BK → LW（闭合圆环，**反冲移动谐波** $\omega_m$ 处） | 比值 0.9995 → 0.9980，即 $1-\delta$ 的 $O(\omega/\varepsilon)$ 量子修正 ✓（在裸 $m\Omega$ 处评估则 0.999→1.030，为截断伪影，见 BUG-6/P-3） |
| V4 | Larmor 绝对归一化 | 精确谱闭合 0.996；谐波比值 0.946→0.991 收敛 ✓ |
| V5 | 直线 ≈ 零辐射 | \|BK/弯曲\| = 9.8e-10 ✓ |
| V6 | 角积分谱正性（闭合轨道，$\omega_m$ 处） | min(dW/dω) = +1.12，total_W = 0.0660 ✓ |
| V7 | trace ≡ dot 在壳恒等式（$\omega/\varepsilon=0.7$ 硬光子） | rel_err = 6.2e-16 ✓ |
| V8 | **量子同步辐射交叉验证**（$\gamma=10$，$\chi=0.5$，$\delta=0.05$–$0.5$，$N_t=7960$） | BK/精确 = 0.997–1.001（dot 与 trace 相同），而量子/经典 = 0.96→0.44 ✓ |

虚部诊断：numpy 参考路径实测 $\mathrm{Im}/\mathrm{Re} \lesssim 10^{-10}$（舍入级），numba 路径显式利用厄米对称性，虚部恒为 0。回归测试见 [tests/test_baier_katkov.py](lambdapic/tests/test_baier_katkov.py)（含参照公式三重锚定与量子交叉验证）。

**结论：模块的骨架、相位符号和经典极限*形状*是可靠的；但存在 6 个已复现的代码缺陷（含一个使所有绝对量偏大 $4\pi$ 的前因子错误 BUG-5，和一个 trace 核抄写错误 BUG-6）、1 个文档债务问题，以及 3 个会直接影响 PIC 应用的物理缺口（见下）。缺陷已于 2026-09-08 全部修复；量子区间已由 V8 对精确谱验证到 0.1–0.3%。**

---

## 2. 已确认的代码缺陷（均已复现，2026-09-08 已全部修复）

> 修复后回归：V1–V7 结果逐位不变；每项修复均有复现脚本验证（见各条"修复"注记）。

### BUG-1（严重，语义错误）：`Parameters.recoil` 从未进入计算路径 ✅ 已修复

[integrator.py:201-213](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L201-L213) 中 `BKIntegrator.d2_probability/d2_energy` 只把 `params.epsilon/mass/charge/kernel` 传给底层函数，**从不传 `recoil`、`spin_averaged`、`polarization_summed`**。而 [types.py](lambdapic/src/lambdapic/core/qed/baier_katkov/types.py) 的 `Parameters` 却暴露这些字段、`epsilon_prime()`/`recoil_factor()` 方法，并在 metadata 里记录 `recoil=self.params.recoil`。

复现：`Parameters(epsilon=γ, recoil="classical")` 与默认 `"baier_katkov"` 给出**逐位相同**的结果（-0.0041752842...），而显式传 `phase="classical"` 的正确结果为 -0.0041810726。即：

- 用户设置 `recoil="classical"` 会被**静默忽略**，计算仍用 BK 反冲；
- 更糟的是输出的 `Spectrum.metadata` 会**谎报** `recoil: classical`。

修复（2026-09-08）：新增 `_recoil_args()`，`BKIntegrator.d2_probability/d2_energy` 现在把 `params.recoil` 映射为 `phase`/`epsilon_prime` 传入（`"classical"` → `phase="classical", epsilon_prime=epsilon`）。回归验证：`recoil="classical"` 与显式传参结果逐位一致（diff = 0），且与 BK 反冲结果不同。另在 `Parameters.__post_init__` 中对 `spin_averaged=False` / `polarization_summed=False` 抛 `NotImplementedError`（当前只有自旋平均/极化求和核）。

### BUG-2：`kernel` 字符串不校验，非法值静默回退 ✅ 已修复

[integrator.py:98](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L98)：`kern = dot_kernel if kernel == "dot" else trace_kernel`。实测 `kernel="foo"` 不报错，静默按 trace 核计算。此外 `"velocity"`（validation V2 内部使用）有效但**未写入 docstring**，属于隐藏 API。

修复（2026-09-08）：`double_time_integral` 现校验 `kernel in ("dot", "trace", "velocity")` 否则抛 `ValueError`；`"velocity"` 已写入 docstring；dispatch 重构为显式三分支（`"velocity"` 不再先赋 trace 核再被覆盖的死代码路径）。

### BUG-3：`omega == epsilon` 除零 / `omega > epsilon` 无守卫 ✅ 已修复

`epsilon' = epsilon - omega`，[integrator.py:90](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L90) 及 `phase.recoil_frequency` 中 `omega * epsilon / epsilon_prime`。实测 `omega == epsilon` 抛裸 `ZeroDivisionError`；`omega > epsilon` 时 $\varepsilon'<0$，反冲因子变负、相位反号，产生**非物理但有限**的结果，无任何警告。

修复（2026-09-08）：`double_time_integral` 在 `epsilon_prime=None`（BK 反冲默认）时校验 `omega < epsilon`；`phase.recoil_frequency` 校验 $\varepsilon' > 0$；`Parameters.epsilon_prime()` 同步守卫。均抛带具体数值信息的 `ValueError`（经典模式 `epsilon_prime=epsilon` 不受限，符合物理）。

### BUG-4：`trapz_weights` 单样本崩溃 ✅ 已修复

[integrator.py:50](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L50)：`time` 长度为 1 时 `dt` 为空数组，`w[0] = dt[0]/2` 抛 `IndexError`。实测复现。

修复（2026-09-08）：`trapz_weights` 对单样本返回 `[0.0]`（不再 `IndexError`）；`Trajectory.__post_init__` 拒绝 `Nt < 2`（`ValueError: at least two time samples are required`），从源头挡住无意义的双时间积分。

### BUG-5（严重，归一化错误）：辐射前因子混用单位制，所有绝对量偏大 $4\pi$ ✅ 已修复

第一轮审查漏掉，原因是 V4 "Larmor 归一化通过"——但 V4 的参考公式与代码带着**同一个**错误，比值类检验（V3/V4/V5）对前因子全部免疫。证据链：

1. **推导**：Liénard–Wiechert 谱 $\frac{e^2\omega^2}{4\pi^2 c}\left|\int \mathbf n\times(\mathbf n\times\boldsymbol\beta)\,e^{i\omega(\cdots)}dt\right|^2$（Jackson 14.67）是**高斯制**，$\hbar=c=1$ 时 $e^2=\alpha$，前因子应为 $\alpha/(4\pi^2)$。原 [units.py](lambdapic/src/lambdapic/core/qed/baier_katkov/units.py) 注释把 Heaviside–Lorentz 制的 $e^2=4\pi\alpha$ 代进高斯公式得到 $\alpha/\pi$。BK 公式 $dw=\frac{\alpha}{(2\pi)^2}\frac{d^3k}{\omega}\int\!\!\int\cdots$ 同样是 $\alpha/(4\pi^2)$。
2. **独立标定**：同步辐射每圈能量损失 $\frac{4\pi}{3}\frac{\alpha\gamma^4}{\rho}$ ⇒ 88.5 keV·$E[\mathrm{GeV}]^4/\rho[\mathrm m]$，只有 $P=\frac23\alpha\gamma^4\beta^4/\rho^2$ 对得上；原 V4 用的 $\frac{8\pi}{3}$ 多了 $4\pi$（高斯 Larmor 的 $\frac23 e^2$ 配了 HL 的 $e^2$），其 $dP/d\omega=2\sqrt3\,\alpha\gamma/\rho\,F$ 亦然（标准 Schwinger 谱是 $\frac{\sqrt3}{2\pi}\alpha\gamma/\rho\,F$）。
3. **实测**（$\gamma=30$、$\rho=10^4$、$N_t=4000$，对标准 Schwinger 谱）：code_LW/standard = 12.29 ($m$=16) → 12.45 ($m$=64)，趋向 $4\pi=12.57$；BK 同样。
4. **与项目自身不一致**：LCFA 表 [optical_depth_tables.py:124-131](lambdapic/src/lambdapic/core/qed/optical_depth_tables.py#L124-L131) 的 Airy 公式（前因子 $\alpha/\varepsilon$）经典极限恰给出标准 Schwinger 谱，即本模块此前所有 $dW/d\omega$、$W$、$E$ 比项目自己的 LCFA 大 12.57 倍。

修复：`integrator.PREFACTOR = alpha/(4 pi^2)`，`d2_energy` 与 `classical_d2_energy` 统一使用；V4 的 $P$、$dP/d\omega$ 去掉 $4\pi$；units.py 注释、模块 docstring、README 公式同步改正。回归：V4 闭合仍 0.996、谐波比值仍 0.946→0.991（比值不变，说明修正自洽）；V6 total_W 由 0.831 变为 0.0661（恰为 $1/4\pi$）；新增 `test_absolute_normalization_schwinger` 在 $m=64$ 处对标准 Schwinger 谱守护到 3% 以内。

### BUG-6（物理错误）：trace 核抄写错误，"dot 与 trace 相差 $O(\omega)$"是假象 ✅ 已修复

由 V8 量子交叉验证发现（§4 P-1）。原 [kernel.py](lambdapic/src/lambdapic/core/qed/baier_katkov/kernel.py) 的 trace 核为 $-\frac{m^2}{\varepsilon\varepsilon'}\left[1+\frac{\varepsilon^2+\varepsilon'^2}{4m^2}(\mathbf b_1-\mathbf b_2)^2\right]$，即 $(\mathbf b_1-\mathbf b_2)^2$ 的系数为 $\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon\varepsilon'}$；对精确量子同步辐射谱它系统性偏低（$\delta=0.05\to0.5$ 时 0.94→0.29），而 dot 核一致到 0.1%。推导在壳恒等式：用 $(\mathbf b_1-\mathbf b_2)^2 = 2-2/\gamma^2-2\mathbf b_1\!\cdot\!\mathbf b_2$，系数取 $\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon'^2}$ 时常数项合并为 $\frac{(\varepsilon-\varepsilon')^2 m^2}{2\varepsilon^2\varepsilon'^2}=\frac{\omega^2}{2\varepsilon'^2\gamma^2}$，trace 与 dot **逐点恒等**（任意 $\omega$，数值差 2e-16）。原系数少了一个因子 $\varepsilon/\varepsilon'$。

修复：kernel.py `trace_kernel` 与 integrator.py `_kernel_coefficients` 改为 $-\frac{m^2}{\varepsilon\varepsilon'}-\frac{\varepsilon^2+\varepsilon'^2}{4\varepsilon'^2}(\mathbf b_1-\mathbf b_2)^2$；kernel.py 模块文档重写（删除"受控改写、相差 $O(\omega)$"的错误说法，记录验证史）；V7 改为在硬光子 $\omega/\varepsilon=0.7$ 处检验恒等式到 1e-12（原来只在 $\omega=10^{-3}$ 软极限检验到 1e-2，因此没能发现）；V8 中两核结果逐位相同。**后果**：dot/trace 之争终结，两者是同一函数；只有将来做局域能量核（$\varepsilon_1\ne\varepsilon_2$，P-2）时，含 $b_i^2=1-m^2/\varepsilon_i^2$ 的 trace 形式才是基本形式。

### 次级问题 ✅ 已全部修复（2026-09-08）

- **虚部诊断被静默丢弃**：[integrator.py:130](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L130) `d2_energy` 只返回 `I.real`，与模块 docstring "imaginary part is kept as a numerical diagnostic, never dropped silently" 的承诺矛盾。→ `d2_energy`/`d2_probability` 新增 `diagnostics=False` 开关，为 `True` 时返回 `(value, value_imag)`；默认行为不变。实测 Im/Re ≈ 1e-16。
- **`Trajectory.__post_init__` 不校验 `time.ndim == 1`**（`as_trajectory` 校验了），直接构造 2-D time 会在积分器里得到难以理解的错误。`Spectrum.__post_init__` 同理不校验 `dE_domega` 形状。→ 两处校验均已补上。
- **`kernel.py` 里 `fine_structure` 只是重新导出**（[kernel.py:35](lambdapic/src/lambdapic/core/qed/baier_katkov/kernel.py#L35)），模块内未使用，建议从 `__all__` 移除以免误导。→ 已移除（全模块 grep 确认无人从 `kernel` 导入它）。
- **`double_time_integral` 中 `kern` 对 `"velocity"` 会先赋成 trace 核再被覆盖**（[integrator.py:98-115](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L98-L115)），属于死代码路径，重构 dispatch 时一并清理。→ 随 BUG-2 一并清理。

---

## 3. 文档债务：设计文档全部悬空

模块中几乎所有物理公式的引用（"eq. 7.1"、"eq. 5.3"、"sec. 15"、"sec. 4.2"、"eq. 10.3"…）都指向三个不存在的文件：

- `Baier-Katkov.md`（核与相位的推导、eq. 编号来源）
- `Baier-Katkov-architecture.md`（README 声称在仓库根目录）
- `Baier-Katkov-multiparticle.md`（路线图 sec. 15）

已在 `d:\Desktop\lambdapic` 全树（排除 .venv）搜索确认这三个文件**不存在**。后果：

1. 迹核/点积核的具体形式、$(\alpha/\pi)$ 前因子、$\varepsilon/\varepsilon'$ 反冲相位约定**无法独立审计**——模块的物理正确性目前只能靠 V1–V7 的内部自洽与经典极限支撑；
2. 点积核与迹核之间存在 $O(\omega/\varepsilon)$ 的差别（前因子 $(\varepsilon^2+\varepsilon'^2)/(2\varepsilon'^2)$ vs 有效 $(\varepsilon^2+\varepsilon'^2)/(2\varepsilon\varepsilon')$，差一个因子 $\varepsilon'/\varepsilon$），**哪一个更接近真实量子谱，目前没有任何外部依据**（见 §5 物理缺口 P-1）；
3. 新贡献者无法理解 "eq. 10.3 量子同步辐射谱" 指什么。

**建议**：把设计文档（哪怕只是方程清单 + 推导要点）纳入仓库 `docs/`，或将方程编号直接内联进模块 docstring；至少为每个 eq. 引用补上文献出处（Baier–Katkov 原始论文 / Baier–Katkov–Strakhovenko 专著）。

---

## 4. 物理层面的问题（影响 PIC 应用的关键项）

### P-1：量子项（反冲/ω² 项）从未被独立验证 ✅ 已解决（2026-09-08，validation V8）

原状：V3/V4 只在 $\omega/\varepsilon \to 0$ 软光子区验证了经典极限；点积核中的 $\omega^2/\gamma^2$ 项、反冲相位因子 $\varepsilon/\varepsilon'$、前因子 $(\varepsilon^2+\varepsilon'^2)/(2\varepsilon'^2)$ 从未对照量子同步辐射谱验证过——而量子反冲恰恰是这个模块相对 LCFA 的核心卖点。

**结果**（[reference.py](lambdapic/src/lambdapic/core/qed/baier_katkov/reference.py) + `validation.check_quantum_synchrotron`，公式与对接方式见 §7-B）：

| $\gamma$ | $\chi$ | $\delta=\omega/\varepsilon$ | 量子/经典 | BK(dot)/精确 |
|---|---|---|---|---|
| 10 | 0.1 | 0.02 – 0.3 | 0.98 → 0.38 | 0.999 – 1.001 |
| 10 | 0.5 | 0.05 – 0.5 | 0.96 → 0.44 | 0.997 – 1.001 |
| 10 | 2.0 | 0.05 – 0.7 | 0.97 → 0.54 | 0.988 – 1.001 |
| 20 | 0.5 | 0.05 – 0.5 | 0.96 → 0.44 | 0.9993 – 1.0002 |

$\gamma$ 从 10 到 20 时残差由 0.3% 降到 0.07%，符合参照公式本身的 $O(1/\gamma^2)$ 精度；角度网格 (24,8) 与 (48,16)、$N_t$ 6000 与 12000 结果到 4 位相同（已收敛）。**反冲相位、前因子、$\omega^2/\gamma^2$ 项全部正确**；同时发现并修正了 trace 核抄写错误（BUG-6）。参照公式本身经三重锚定：Macdonald 形式 ≡ Airy 形式 ≡ 项目 LCFA 表 `gen_photon_prob_rate_for_delta`（≤ 5.5e-6）；$\chi\to0$ → Schwinger 谱；经典功率 ≡ Larmor，量子功率/Larmor 与 BKS 拟合 $g(\chi)$ 相符（≤ 1.4%）。

### P-2：固定入射能量 ε —— 对 PIC 轨迹不成立 ✅ 已解决 2026-09-09

整条轨迹使用单一 $\varepsilon$（`Parameters.epsilon`），$N(t_1,t_2)$ 和 $\Phi$ 均如此。PIC 中电子能量沿轨迹随激光场振荡并因辐射损失变化，正确形式是局部能量 $\varepsilon(t_1), \varepsilon(t_2)$（模块自认"sec. 4.2"）。对激光尾场加速（能量变化 $O(1)$）或长记录，固定 ε 会同时错相位和核。

**已实现**：`Trajectory.energy` 可选字段（逐采样 `eps(t)`，正数校验），`Trajectory.local_energy()` 从动量历史恢复 `eps = m·γ`。核与相位切换到逐顶点形式（`kernel.py` 的 `epsilon2` 关键字 + `integrator.py` 的 `energy=` 入口，numpy/numba 双后端）：

- 相位 $\Phi_{ij}=\omega\left[f_j(t_j-\mathbf n\cdot\mathbf r_j)-f_i(t_i-\mathbf n\cdot\mathbf r_i)\right]$，$f_i=\varepsilon_i/\varepsilon'_i$，用差分稳定形式 $\Phi=\omega[\bar f\,\Delta x + \Delta f\,\bar x]$ 求值（$\bar f$、$\Delta f$ 为逐对平均/差，$x=t-\mathbf n\cdot\mathbf r$），能量变化只经小项 $\Delta f$ 进入，保留反演反对称性 → 厄米上三角求和仍成立。
- 迹核（基本形式）：$N_{ij}=(A_i+A_j)+(B_i+B_j)(\mathbf b_i\cdot\mathbf b_j-1)$，$A_i=-\frac{m^2}{2\varepsilon_i\varepsilon'_i}+\frac{m^2 c_i}{4\varepsilon_i^2}$，$B_i=\frac{c_i}{4}$，$c_i=\frac{\varepsilon_i^2+\varepsilon_i'^2}{\varepsilon_i'^2}$。

**关键发现**：逐顶点反冲 $\varepsilon'_i=\varepsilon_i-\omega$ 下，在壳恒等式 $(\varepsilon_i-\varepsilon'_i)^2=\omega^2$ **逐顶点成立**，故对称化的局域 dot 与 trace 核**严格重合**（回归测试到 $10^{-10}$）。即 kernel.py 文档旧预期"ε1≠ε2 时两核不同"只在一个固定全局 $\varepsilon'$ 的约定下成立；本项目采用 roadmap 的逐顶点 $\varepsilon'_i=\varepsilon_i-\omega$ 约定，两核合一。经典模式（$\varepsilon'_i=\varepsilon_i$，$f=1$）则完全不含能量历史（trace 退化为纯 $\mathbf b_1\cdot\mathbf b_2-1$），已作为"能量无关性"测试固定。**注**：sec. 4.2 局部形式因 `Baier-Katkov.md` 缺失（§3 文档债）而由对称化推导并靠测试钉死（恒能量退化 ≡ 固定路径到 $10^{-16}$；能量变化轨道上谱正性、厄米性、numba≡numpy 到 $10^{-11}$）。

pytest 新增 8 个用例（12 个参数化实例）：`test_local_energy_degenerates_to_fixed`、`_kernels_coincide_on_shell`、`_matches_numpy`、`_is_hermitian`、`_spectrum_positive`、`_classical_is_energy_independent`、`_validation`。固定路径未动，23 例原回归 + 35 例全量 + V1–V8 全部通过。

### P-3：有限记录边界辐射 —— PIC 应用的拦路虎

实测：闭合圆环在**非谐波**频率 $\omega/\varepsilon = 0.5$ 处，宽锥角积分的 dW/dω = **-4922**（大负值）；这正是 V6 文档中所说的"硬截断端点辐射"。V8 进一步量化：即使在闭合轨道上，只要评估频率偏离周期条件（裸 $m\Omega$ 而非反冲移动的 $\omega_m$，相位失配仅 $2\pi m\,\omega/\varepsilon$），角积分谱就偏离 ±40%（`test_bare_harmonic_evaluation_is_wrong` 守护此事实）。PIC 记录的轨迹天然是任意截断的非闭合片段，边界项会污染整个谱。修复需要：

- 端点绝热开关（adiabatic switch / 渐近延拓）或
- 形成长度窗口（见 §6.3）——把 $|t_2 - t_1|$ 截断在形成长度 $l_f$ 内，物理上同时消除边界污染并降复杂度，是首选。

### P-4：默认锥角 $\theta_{max}=5/\gamma$ 漏掉高 ω 尾巴

BK 的特征发射角随光子能量按 $(\varepsilon/\varepsilon')\,/\gamma$ 展宽（ω→ε 时发散）。默认固定锥 5/γ 对高 ω 会系统性低估角度积分。修复：`theta_max(ω) = c·(\varepsilon/\varepsilon')/\gamma` 自适应，或对 theta_max 做收敛性检查。

---

## 5. 性能现状与优化方向

**2026-09-08 起 [integrator.py](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py) 提供双后端**：`"numpy"`（分块外积参考路径，保留为真值）与 `"numba"`（默认；`@njit(parallel=True, cache=True)`，显式利用 $K(t_2,t_1)=K(t_1,t_2)^*$ 只算上三角，并把一个谱的全部 $(\omega,\mathbf n)$ 对合并为对 $(t_1,t_2)$ 的一次遍历——核在方向间共享、相位宗量在频率间共享；行按 $i \leftrightarrow N_t-1-i$ 折叠配对以平衡 prange 负载）。两后端对三种核 × 两种相位一致到尺度相对 $\lesssim 10^{-14}$。

实测（16 线程）：

| 配置 | numpy | numba |
|---|---|---|
| 示例：$N_t=512$，8 ω × 16 方向 | 4.2 s | **0.03 s**（约 140×） |
| $N_t=4000$，4 ω × 16 方向 | — | 0.83 s ≈ 0.62 G(对×组合)/s |
| validation V1–V7 全套 | ~60 s | **1.3 s** |
| 首次编译（之后走磁盘缓存） | — | ~3.7 s |

外推：PIC 轨迹 $N_t\approx4096$、50 ω × 32 方向 → 单粒子约 **20 s**（修复前估算 1.7 小时）；§7-B 的量子同步辐射交叉验证（$\gamma=10$、$N_t=2\times10^4$、15 ω × 16 方向）约 80 s，已可行。

仍是 $O(N_t^2 N_\omega N_{dir})$——numba 只是同一方法的快速求值，不改变标度。后续按性价比：

1. **形成长度带状截断**（复杂度降到 $O(N_t\,N_\omega)$）：只对 $|t_2-t_1| \lesssim 2 l_f(\omega)$ 的带内求和，同时天然解决 P-3 的边界污染——现在是最大的杠杆；
2. **FFT/NUFFT**（远期）：固定 ε 时相位是 $e^{if(\omega,\mathbf n)(t_2-t_1)}e^{ig(\omega,\mathbf n)\cdot\Delta\mathbf r}$ 结构，单时间谱可用 NUFFT 降到 $O(N_t\log N_t)$，但核 $N(t_1,t_2)$ 非可分（含 $\boldsymbol\beta_1\cdot\boldsymbol\beta_2$），需低秩/多项式展开才能完全 FFT 化。

---

## 6. 与主模拟的集成点（已勘察）

主模拟内部是 **SI 单位**（场 $E_0 = a_0 m_e c \omega_0/e$，见 [laser.py:339](lambdapic/src/lambdapic/callback/laser.py#L339)；χ 计算用 SI 常数，见 [inline.py:5-13](lambdapic/src/lambdapic/core/qed/inline.py#L5-L13)）。而 BK 模块是 Compton 单位。`units.si_to_natural` 已覆盖 t/x/E/ω 与 $p \to u$ 的换算，**单位桥已具备**，缺的是一个把 Simulation 数据流式转成 `Trajectory` 的适配器。

关键集成点（[simulation.py](lambdapic/src/lambdapic/simulation/simulation.py) 的 run 循环）：

| 阶段 | 用途 |
|---|---|
| `_push_position_1` / `_interpolator` 之后 | 每步读取 $x,y,u_x,u_y,u_z$（SI + 归一化动量），记录轨迹样本 |
| `radiation`（`update_chi()` + `event()`）之后 | 读取 χ、δ、event，用于 BK/LCFA 对照与事件标记 |
| callback 机制 `@callback(stage=...)` | 轨迹记录器/谱累加器的自然挂载点 |

颗粒数据（[particles.py](lambdapic/src/lambdapic/core/particles.py)）已具备集成所需的全部字段：`x,y,z`（SI）、`w`（宏粒子权重）、`ux,uy,uz`（= $u=\gamma\beta$，与 `Trajectory.momentum` 约定一致）、`_id`（64 位稳定粒子 ID，跨 patch/MPI 追踪轨迹必需）。MPI 下按 rank 收集、按 `_id` 归并即可。

**双时间积分不需要插值/重采样**——梯形权重支持非均匀网格（[integrator.py:45](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L45)），PIC 步进的时间网格可直接使用；但若将来走 FFT 路径，则需 `linear_interpolate` 重采样到均匀网格。

---

## 7. 路线图：从参考实现到完整 PIC 模拟

### 阶段 A — 修复与文档补全（先做，1–2 天）

1. ✅ 修 BUG-1 ~ BUG-5（§2）；
2. 把 `Baier-Katkov.md` 的方程清单与推导要点补进仓库（至少为每个 eq. 引用补文献出处），或删除悬空引用；
3. ✅ pytest 回归 [tests/test_baier_katkov.py](lambdapic/tests/test_baier_katkov.py)：numba≡numpy、虚部、$|A|^2$ 恒等式、Schwinger 绝对归一化、`recoil` 选项生效、谱自洽、非法输入拒绝（13 例，约 4 s）。

### 阶段 B — 物理验证闭环（核心，决定模块可信度）

4. ✅ **P-1 量子同步辐射交叉验证**（2026-09-08 完成：`reference.py` + validation V8 + pytest `test_quantum_synchrotron_*`；结果见 §4 P-1）。参照物是**恒定场量子同步辐射谱**（Sokolov–Ternov / Baier–Katkov 1967 / Ritus 1985，也就是 LCFA 速率），且正是项目 LCFA 表 [`gen_photon_prob_rate_for_delta`](lambdapic/src/lambdapic/core/qed/optical_depth_tables.py#L124-L131) 所用的公式——它是同一理论在恒定场中把双时间积分解析做完的结果，所以检验的是**实现**（前因子、反冲相位、$\omega^2/\gamma^2$ 项、dot/trace 谁对）而非理论；圆轨道上场沿轨迹严格恒定，LCFA 精确成立，对照没有模糊地带。模块单位制（$c=\hbar=m_e=1$，$\varepsilon=\gamma$）下，记 $\chi=\gamma^2\beta^2/\rho$，$\delta=\omega/\varepsilon$，$y=\dfrac{2\delta}{3\chi(1-\delta)}$，单位实验室时间的光子数谱为

$$\frac{dW}{dt\,d\delta} = \frac{\alpha}{\sqrt3\,\pi\,\varepsilon}\left[\left(1-\delta+\frac{1}{1-\delta}\right)K_{2/3}(y) - \int_y^\infty K_{1/3}(x)\,dx\right],\qquad \frac{dP}{d\omega}=\frac{\omega}{\varepsilon}\frac{dW}{dt\,d\delta}$$

   等价的 Airy 形式（项目表所用；$z=[\delta/(\chi(1-\delta))]^{2/3}$，$y=\tfrac23 z^{3/2}$）：$\dfrac{dW}{dt\,d\delta}=-\dfrac{\alpha}{\varepsilon}\left[\displaystyle\int_z^\infty\!\mathrm{Ai}(x)dx+\left(\frac2z+\delta\chi\sqrt z\right)\mathrm{Ai}'(z)\right]$，由 $\mathrm{Ai}(z)=\frac1\pi\sqrt{z/3}\,K_{1/3}(y)$、$\mathrm{Ai}'(z)=-\frac{z}{\pi\sqrt3}K_{2/3}(y)$ 及 $2+\delta\chi z^{3/2}=(1-\delta)+\frac1{1-\delta}$ 可证恒等。经典极限 $\chi\to0$ 退化为 Schwinger 谱 $\frac{dP}{d\omega}=\frac{\sqrt3}{2\pi}\frac{\alpha\gamma}{\rho}F(\omega/\omega_c)$、总功率 $\frac23\alpha\chi^2$（V4 与 pytest 已验证此极限）；积分量检验可用 $P=P_{cl}\,g(\chi)$，$g\approx[1+4.8(1+\chi)\ln(1+1.7\chi)+2.44\chi^2]^{-2/3}$。

   对接方式：闭合一圈的角积分 $\left.dE/d\omega\right|_{\omega_m}$ 对应 $T\cdot dP/d\omega$（V4 的用法）。**必须在反冲移动后的谐波 $\omega_m = \dfrac{m\Omega}{1+m\Omega/\varepsilon}$ 处评估**：BK 双积分的周期性由 $\omega\varepsilon/\varepsilon'$ 决定，无边界项的条件是 $\omega\varepsilon/\varepsilon'=m\Omega$；在裸 $m\Omega$ 处评估会带入 $\propto m\,\omega/\varepsilon$ 的边界项——V3 中随 $m$ 增长的 0.1%→3% 偏离（远大于 $O(\omega/\varepsilon)=0.2\%$）疑为此伪影而非量子修正，交叉验证时一并澄清。参数：$\rho=\gamma^2/\chi$，$\chi\in\{0.05,0.2,0.5,1\}$，$\delta\in[0.01,0.6]$，$N_t\gtrsim10\,m_c$（$m_c=\tfrac32\gamma^3$，$\gamma=10$ 时 1500）。$\gamma=10$ 的 $1/\gamma^2$ 修正约 1%，足以判定 dot/trace（两者主项差因子 $1-\delta$，$\delta=0.3$ 时差 30%）。
5. ✅ **P-2 局部能量核**（2026-09-09 完成）：把 $N(t_1,t_2)$、$\Phi$ 推广到 $\varepsilon(t_1),\varepsilon(t_2),\varepsilon'_i = \varepsilon_i - \omega$（改动在 kernel.py + phase.py + integrator.py + types.py，`Trajectory` 增加可选能量历史字段 `energy`；局部形式由对称化推导、靠测试钉死，详见 §4 P-2）；
6. **P-3 边界处理**：优先实现形成长度窗口 $W(|t_2-t_1|/l_f)$（兼顾性能），辅以端点绝热开关；用"截断非闭合圆弧 + 与闭合圆环对比"做收敛性检验；
7. **P-4 自适应锥角** `theta_max(ω)` 或收敛检查。

### 阶段 C — 性能（在 B 的验证之上，避免优化错误代码）

8. ✅ Numba 化（2026-09-08，见 §5）；形成长度带状截断待做（现为最大杠杆，兼治 P-3）；
9. ✅ 多 $(\omega,\mathbf n)$ 批量接口 `double_time_integral_batch` / `BKIntegrator.d2_energy_batch` 已随 numba 后端提供；多**轨迹**批量（共享 ω/n 网格）待做。

### 阶段 D — PIC 集成

10. **单位适配器** `SimUnitAdapter`：Simulation SI 量 → Compton 单位，封装 `units.si_to_natural`；
11. **轨迹记录回调** `TrajectoryRecorder`：`@callback` 挂 `_interpolator` 阶段，按 `_id` 追踪选中粒子（含跨 patch 迁移），输出 `Trajectory`（记录无反冲背景——注意：带 LCFA 反冲的轨迹不能喂给本模块，模块 README 已声明防双计）；
12. **系综求和**：$S_{total} = \sum_i w_i S_i$（`Spectrum` 加权加法器 + 按权重归一）；
13. **BK/LCFA 混合策略**（本模块在 PIC 中的实际用法，两种模式）：
    - *离线后处理模式*（推荐先做）：跑一次无反冲 PIC（关掉 radiation），录轨迹，BK 出谱——用于研究低 ω 红外区、形成长度效应、离散谐波结构；
    - *在线事件模式*（高级、可选）：从 BK 微分速率 $dW/d\omega$ 构造光学深度采样器替换 LCFA 表。注意 BK 是非局域理论，直接采样需要"形成长度内局域化"近似；多数代码的做法是保留 LCFA 事件生成、BK 只做谱后处理与 LCFA 表校准。

### 阶段 E — 工程化

14. CI 中跑 pytest 化的 V1–V7 + 一个端到端 PIC 轨迹 → BK 谱冒烟测试；
15. 文档：修复 §3 悬空引用，README 增补"如何从 Simulation 得到一条 Trajectory"的最小教程。

---

## 8. BK 相对传统（LCFA）方法的物理价值定位

路线图服务于用户目标"得到传统方式得不到的量子相关的结果"。明确本模块能给出的、LCFA 管线给不出的东西：

1. **红外/软光子区精确谱**：LCFA 表在 $\omega \lesssim$ 特征红外截断处失效（且采样表有 $\chi_{min}$ 截断，见 [optical_depth.py:35](lambdapic/src/lambdapic/core/qed/optical_depth.py#L35)），BK 双时间积分在此天然有限；
2. **形成长度非局域效应**：LCFA 假设场在形成长度内不变；BK 对有限长度/变场结构精确，可定量给出 LCFA 的适用边界；
3. **离散谐波与干涉结构**：wiggler 类结构化场、长脉冲尾场的谐波谱，LCFA 连续近似看不到；
4. **反冲的非微扰处理**：$\varepsilon/\varepsilon'$ 相位因子 + $\varepsilon'=\varepsilon-\omega$ 全阶保留；
5. **PIC 级对比基准**：作为 LCFA 表的离线校准/验证工具（χ 边界处）。

同时也应写清 BK 的适用边界：准经典方法假定 $\chi \lesssim 1$ 且光子自旋/电子自旋已求和（`spin_averaged`、`polarization_summed` 固定为 True）——自旋分辨谱需要新的核，是更远的扩展。

---

## 9. 行动清单速查

| 优先级 | 事项 | 工作量 | 对应章节 |
|---|---|---|---|
| ~~P0~~ ✅ | ~~修 BUG-1（recoil 静默忽略）~~ 已完成 2026-09-08 | — | §2 |
| ~~P0~~ ✅ | ~~修 BUG-2/3/4（校验与守卫）~~ 已完成 2026-09-08 | — | §2 |
| ~~P0~~ ✅ | ~~修 BUG-5（前因子 4π）+ pytest 回归~~ 已完成 2026-09-08 | — | §2, §7-A |
| P0 | 补设计文档/文献出处（架构/函数说明已补：README.md 重写为中文详细文档 2026-09-09；剩余：物理推导设计文档 `Baier-Katkov.md` 与文献引用） | 0.5–1 d | §3 |
| ~~P1~~ ✅ | ~~量子同步辐射交叉验证~~ 已完成 2026-09-08（BK/精确 0.997–1.001；顺带修 BUG-6 trace 核） | — | §4 P-1, §7-B |
| P1 | 形成长度窗口 + 边界处理 | 1–2 d | §7-B |
| ~~P2~~ ✅ | ~~局部能量核 ε(t)~~ 已完成 2026-09-09（`Trajectory.energy` + 逐顶点核/相位，双后端；固定路径未动） | — | §7-B |
| ~~P2~~ ✅ | ~~Numba 化~~ 已完成 2026-09-08（约 140×） | — | §5 |
| P2 | 带状截断（$O(N_t N_\omega)$） | 1–2 d | §7-C |
| P3 | 单位适配器 + 轨迹记录回调 + 系综求和 | 2–3 d | §7-D |
| P4 | 在线事件采样 / 自旋分辨核 / NUFFT | 周级 | §7-D/E |

