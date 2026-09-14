# Baier–Katkov 模块审查报告与 PIC 集成路线图

> 审查日期：2026-09-07；更新：2026-09-08（§2 缺陷全部修复、前因子 4π 修正、Numba 后端、pytest 回归）、2026-09-11（P-1/P-2 收尾、P-3 重新诊断 + 适用性守卫、**P-4 自适应锥角 + 两板网格 + 角度守卫，原文方向判断被实测推翻**）、**2026-09-12（第二轮独立审查：基线全绿复核 + Test B 复现；新增 BUG-7/8/9 与一批输入校验、文档、验证债问题，全部本机复现，集中在 §2.5；其中代码级问题同日全部修复，回归 51→58 例）**、**2026-09-14（§8.2 带状截断判据实验：结论是"不做"——滞后分布不衰减、截断曲线小于整条记录都不收敛；§8.1 的"钥匙"说法作废，§5 P2 该项关闭）**。范围：[baier_katkov/](lambdapic/src/lambdapic/core/qed/baier_katkov/) 全部源码（约 1400 行）+ 主模拟 QED 管线（[radiation.py](lambdapic/src/lambdapic/core/qed/radiation.py)、[optical_depth.py](lambdapic/src/lambdapic/core/qed/optical_depth.py)、[inline.py](lambdapic/src/lambdapic/core/qed/inline.py)、[simulation.py](lambdapic/src/lambdapic/simulation/simulation.py)）。
> 所有"已确认"的缺陷都在本机实际运行复现过：第一轮在 Windows 专机（venv，Python 3.11.9），**第二轮在 Linux 集群（`/home/panda/lambdapic/.venv`，Python 3.12.7，numpy 2.4.4）**。

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

### 第二轮审查新发现（2026-09-12，全部已在本机复现）

> **2026-09-12 同日修复**：本小节标题与"次级""文档债"两条列出的**代码级问题已全部修掉**（BUG-7/8/9 + 输入校验 + metadata 谎报），并补 7 个 pytest 用例钉住行为。回归 **58 例全过**（原 51 例全部保持通过），V1–V8 数值**逐位不变**（含 V6 的 `total_W = 6.599e-02`）。**未修**的只剩纯文档项：`kernel.airy_identity` 无调用者/无测试、V5 文档说过头、README §8 环境仍是 Windows 旧值、`phase.*` 公开函数不在计算路径上。修复清单见各行 ✅。
>
> 环境：Linux 集群，`/home/panda/lambdapic/.venv`（Python 3.12.7，numpy 2.4.4）。基线先行核实：V1–V8 全过（数值与本文档记录一致）、`tests/test_baier_katkov.py` **51 例全过**、单粒子示例端到端正常、Test B 复现三个主标度（$n^2$ 律偏差仍为 $+0.30\%$、$\Gamma=n$、线宽 $\propto1/n$）。

#### BUG-7（中等，诊断报错位置）✅ 已修复：`angular_edge_fraction` 指向靠轴的节点，不是锥边缘节点

[integrator.py:964](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L964) 用 `d2E[:, -n_phi:]` 取"最外侧 $\theta$ 节点"，docstring 断言"directions 按内板在前、外板在后排列，最后 `n_phi` 个样本是离 `theta_max` 最近的节点"。**这个断言与 `cone_directions` 的实际排序相反**：`cos_theta` 由 `roots_legendre` 升序节点构造，故 $\theta$ **降序**——索引 0 才是离 `theta_max` 最近的节点。

复现（$n_\theta=4,\theta_{max}=1.0$）：各节点 $\theta=[0.9616,0.8065,0.5580,0.2533]$；只有索引 0 有值时返回 `0.0`，只有最后节点有值时返回 `1.0`——**判据完全反了**。真实谱上（$\gamma=10$ 圆、$\theta_{max}=0.6$、$n_\theta=8,n_\phi=4$）metadata 报 `angular_edge_fraction=0.033`，而真正锥边缘节点（$\theta=0.594=\theta_{max}$）承担 **38.6%** 的角积分。

后果：README §3.7 与本文档 P-4(d) 把它推荐为"锥角够不够"的**免费判据**（"大 ⇒ 该抬高 `theta_max`"）。在闭环这种固定锥本就不收敛的情形（P-4(e)）它本应报警，却报出一个小值，**恰好掩盖了唯一该被它暴露的信号**。两板模式下标量取的是贴近 `split` 的节点，既不是边缘也不是轴，语义更含糊。

**修复**：`angular_edge_fraction(d2E, dom, n_phi, offset=0)` 改取 `slice(offset, offset + n_phi)`；`compute_spectrum` 用新增的 `cone_bands` 算 `offset`（单板 `0`、两板 `n_inner * n_phi`）并传入；`cone_directions` 与 `cone_bands` 的 docstring 写明"板内 $\theta$ **递减**，离 $\theta_{max}$ 最近的节点是最后一板的**头** $n_\phi$ 个"。实测：修复后 metadata 的 `angular_edge_fraction` **精确等于**最宽 $\theta$ 节点的真实份额（同轴网格下两边都是 0.04932，相对误差 $10^{-12}$），而旧值 0.03265 是轴节点。新增 `test_angular_edge_fraction_points_at_the_cone_edge`（合成输入：边缘节点独占→1、轴节点独占→0、两板 `offset`）+ `test_angular_edge_fraction_metadata_matches_the_widest_node`（与真实谱最宽节点对拍，并断言两者份额之差 $>10^{-2}$，使断言有牙齿）。

#### BUG-8（低-中，静默 NaN / 裸异常）✅ 已修复：$\omega=0$ 无守卫

`d2E` 在 $\omega=0$ 处恰为 0（前因子 $\omega^2$），而 `dW = dE/omega`（[integrator.py:1324](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L1324)、733、1190）给出 $0/0$：实测 `dW_domega[0] = nan`、`total_probability() = nan`（NaN 静默传播到整张谱的积分），`d2_probability(omega=0.0)` 抛裸 `ZeroDivisionError`。`_final_energy` 与 `Spectrum.__post_init__` 都不拦 $\omega=0$、也不拒 NaN。BUG-3 只守了 $\omega\ge\varepsilon$ 这一端。

**修复**：`compute_spectrum` 的网格校验 + `d2_probability`/`BKIntegrator.d2_probability`/`d2_probability_batch`/`Spectrum.__post_init__` 四处守卫，$\omega\le0$ 一律抛带解释的 `ValueError`（$\omega=0$ 的 `dE/d\omega` 本身有定义且为 0，仍可用 `d2_energy`）。附带修好 `formation_time`：$\omega\le0$ 处干净返回 $\inf$（物理上 $\tau_f\to\infty$），不再触发除零 RuntimeWarning；`record_adequacy` 原有逻辑把 $\inf$ 映射为"判据不适用"，正是 $\omega=0$ 应有的行为。新增 `test_zero_and_negative_omega_are_rejected`（含 `Spectrum` 拒 NaN/inf）。

#### BUG-9（低-中，公开 API 危险）✅ 已修复：`Trajectory.mass` / `Trajectory.charge` 从不进入计算路径

`Trajectory` 的 `mass`/`charge` 字段（[types.py:53-54](lambdapic/src/lambdapic/core/qed/baier_katkov/types.py#L53-L54)）被 README §6.1 当作有效字段列出，但积分器只读 `Parameters` 的对应字段（[integrator.py:1174](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L1174) 等）。实测：`Trajectory(charge=3.0)` 与默认**逐位相同**（3.655714e-07），只有 `Parameters(charge=3.0)` 生效（3.290142e-06，恰 ×9）；`Trajectory(mass=5.0)` 同样无效。`Trajectory.local_energy()` 全包无人调用。

后果：阶段 D 的单位适配器若按 README 把电荷放在轨迹上，会**静默按 $q=1$ 计算**。与 BUG-1（`recoil` 静默忽略）同型。

**修复**：选择"单一事实来源"而不是"让死字段生效"——`Trajectory` 的 `mass`/`charge` 字段与 `as_trajectory` 的对应参数已**删除**（`local_energy()` 的在壳回退随之改为 `gamma()`，自然单位下 $m_e=1$）。原来的写法留了两个能设电荷的地方，正是产生这个陷阱的原因；现在写在轨迹上会立刻 `TypeError`。`README §6.1` 同步说明。新增 `test_trajectory_carries_no_inert_mass_or_charge`（含 `Parameters.charge` 的 $q^2$ 比率验证）。

#### 次级（输入校验缺失，2026-09-12）✅ 已全部修复

- ✅ **`Trajectory` 不校验 `time` 严格递增**：`as_trajectory` 校验（[trajectory.py:45](lambdapic/src/lambdapic/core/qed/baier_katkov/trajectory.py#L45)），`Trajectory` 直接构造不校验。实测非单调时间被接受，`trapz_weights` 返回乱权重（`[0.1,0.05,0.05,0.15,0.05]`），谱给出**貌似合理的错值**而不报错。README §7 与示例都直接构造 `Trajectory`。→ 校验下沉到 `Trajectory.__post_init__`（报出具体是哪一对样本违反），`as_trajectory` 改为纯转发。
- ✅ **有限性只对 `energy` 校验**：`position`/`momentum`/`time` 里一个 NaN 被接受，整条谱变 NaN，且 `SamplingWarning` 与 `RecordLengthWarning` **都不触发**（NaN 比较为 False，`np.any(nan >= 1)` 为假）——完全静默。PIC 里的丢失粒子会产生这种记录。→ `time`/`position`/`momentum` 全部加有限性校验。
- ✅ **`split` 非法时静默退回单板，metadata 照记**：`cone_directions` 在 `not 0 < split < theta_max` 时丢弃 `split`/`n_inner`（[integrator.py:770](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L770)），但 `compute_spectrum` 仍写 `metadata["theta_split"]`（[integrator.py:1367](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L1367)）。实测 `split=5.0, theta_max=1.0` 返回单板 4 方向，而 metadata 报 `theta_split=5.0, n_inner=8`——**metadata 描述了没被使用的网格**。`compute_spectrum(..., n_inner=…)` 不带 `split` 时同样被静默忽略。→ 新增 `cone_bands` 作为两板规则的唯一事实来源：`split`/`n_inner` 必须成对、`split` 必须落在 $(0,\theta_{max})$，否则 `ValueError`；metadata 改记 `cone_bands` 的**实际**结果。
- ✅ **size-1 数组 `omega` 抛 TypeError**（numpy ≥ 2）：`double_time_integral` 的 `omega = float(omega)`（[integrator.py:650](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L650)）对 `np.array([0.5])` 抛 `TypeError: only 0-dimensional arrays can be converted to Python scalars`；标量与 0-d 数组正常，batch 路径（`np.atleast_1d`）正常。逐点迭代 `grid[i:i+1]` 的调用方会撞上，numpy 1.x 下曾可用。→ 新增 `_as_scalar_omega`（接受标量/0 维/size-1 数组，尺寸 >1 报错），三个单点入口统一走它。

#### 文档与验证债（2026-09-12）

- **`kernel.airy_identity` 无人调用、零测试覆盖**：它在 `__all__` 里、被 README §6.4 与 docstring 说成"仅用于 validation V1"，但全树 grep 无调用者——`validation.check_airy_identity` 是**另一个函数**，公式内联重推（[validation.py:66](lambdapic/src/lambdapic/core/qed/baier_katkov/validation.py#L66)）。公开 API 与测试各自演化的典型入口。
- **V5 文档说过头**：docstring/README §6.9 说直线记录上"BK ≈ LW"，实测 `BK=-2.83e-07` vs `LW=+1.86e-04`，比值 $1.5\times10^{-3}$，**两者并不一致**。物理无问题（dot 核的真空减除消掉直线项，有限记录 LW 振幅不消），是那句文档描述错误。
- **`test_record_adequacy_tracks_the_open_arc_deficit` 没有检验它声称的物理**：三条轨迹是同一圆的 (1.0 圈,1200)、(0.5,600)、(0.25,300)——采样密度恒为 1200/圈、`_curvature_rate` 相同，故 $L/\tau_f$ 严格 $\propto$ 圈数（实测 1.5996/0.7998/0.3999），断言退化成一个阈值区间加一个构造上必然成立的单调性；docstring 里那张谱亏损表（0.89@0.9 圈 … $-0.19$@0.4 圈）**从未被执行**。
- **`test_prefactor_constant` 是变更探测器**：断言 `PREFACTOR == fine_structure/(4π²)`，即 `integrator.py:123` 的定义式本身，不检验归一化（真正钉绝对归一化的是 V8 与 Schwinger 对照）。
- **README §6.5 漏参数**：`BKIntegrator.__init__` 漏 `checks`，`compute_spectrum` 漏 `checks`/`split`/`n_inner`。
- **README §8 环境整节是 Windows 旧值**（`d:/Desktop/...Scripts/python.exe`、Python 3.11、`PYTHONUTF8=1`、"C 扩展未编译"、V1–V8 约 17 s / pytest 约 10 s）；本文档第 4 行的"Python 3.11.9"同。本机实为 Linux、Python 3.12.7、C 扩展 16 个 `.so` 全编译、V1–V8 约 6 s、51 例 pytest 约 11 s。
- **`phase.classical_phase`/`recoil_phase`/`local_recoil_phase` 不在计算路径上**：README §2.4/§6.3 把它们当作"计算所用的相位"，实际积分器用内联副本（[integrator.py:290](lambdapic/src/lambdapic/core/qed/baier_katkov/integrator.py#L290)、328 及 numba 内核）。实测两者一致（$\le10^{-10}$），故**当前不是错误，是漂移风险**；`Parameters.epsilon_prime`/`recoil_factor` 同理未被计算使用。

#### 已核查干净（负面结果，限定上条清单的边界）

numpy/numba 双后端在**非均匀网格**上 × 3 核 × 2 相位 × 固定/局域能量一致到 $10^{-14}$；奇/偶 $N_t$ 配 `block ∈ {1,2,3,8,1000}` 的折行配对覆盖每行恰好一次；逐 $\omega$ 数组 `epsilon_prime`（套件未覆盖）两后端一致；局域核/相位的 numpy 与 numba 公式代数与数值同一；用文档函数（`phase.*`/`kernel.*`）独立重写的朴素 $\sum_{ij}w_iw_jNe^{-i\Phi}$ 能复现 `double_time_integral`；`kernel.trace_kernel`/`dot_kernel` 中"死"的局域分支与 `_local_vertex_arrays` 逐位一致；`trapz_weights`、`_beta_from_position`、两套方向网格的权重守恒、`_kernel_coefficients` 与核函数、`Parameters.recoil` → `(phase, epsilon_prime)` 映射均自洽。**V8 对 scipy Macdonald/Airy、V2 的精确代数恒等式、V7 的在壳恒等式、与项目 LCFA 表的交叉核对都确实独立，未发现循环论证。**

#### 项目级（非代码）

- **仓库位置（2026-09-14 更正）**：git 仓库在 `/home/panda/lambdapic/lambdapic`（分支 `main`），**不是**外层目录——外层 `/home/panda/lambdapic` 放的是虚拟环境、用户配置、参考书 PDF 等不入库的东西，所以在外层 `git log` 会报"不是仓库"。第二轮审查时据此误判为"无版本控制"，记录错误，此处更正。
- **提交状态（2026-09-14）**：此前 09-11 与 09-12 两个会话的成果都只在工作区（最后一次提交是 2026-09-10 的 `253a0ac`；合并起来 10 个文件 +1703/−99，含 2 个未跟踪文件）。已按内容拆成两个提交落库：`411a63f`（09-11 的 P-3 守卫 / P-4 自适应锥角 / Test B，该快照单独跑 51 例全过）与 `2563766`（本轮的 BUG-7/8/9 与输入校验，58 例全过）。拆分方式：先整体回退本轮改动得到可验证的 09-11 快照并提交，再从备份恢复本轮状态提交——**中间态与终态各自都跑过 pytest 与 V1–V8**，用于确认回退没有过推或漏推。
- **Test B 线能量列复现偏差**：本次复跑 $n=2,8,16$ 得 2.0029 / 7.9177 / 15.7285，本文档 §8.1 记的是 1.981 / 8.021 / 15.97（偏差 1.1–1.5%），而 §8.1 声称"$n\ge2$ 后线性到 0.2% 内"。三个主标度（$n^2$、$\Gamma$、线宽）逐位重现，只有"线能量"这个窗口积分量在 1% 量级上不可复现——不影响结论，但该行声称的精度需要下调。

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

**2026-09-11 进展：已与 BKS 专著逐式核对，编号对照建立。** 用 Baier–Katkov–Strakhovenko 专著（*Electromagnetic Processes at High Energies in Oriented Single Crystals*, World Scientific）第 55–77 页（§2.2–§2.4，含 §2.1 的方法论前提）核对，确认模块公式就是该书结果，编号对照如下：

| 模块 docstring 引用 | 实际出处（BKS 专著） | 内容 |
|---|---|---|
| eq. 5.3 / 6.1（相位） | **(2.25a)/(2.25b)**，经 (2.40) 的 $k'_\mu=k_\mu\varepsilon/(\varepsilon-\hbar\omega)$ 进入 | 反冲相位因子 $\varepsilon/\varepsilon'$ |
| eq. 7.3（dot 核） | **(2.42) 第一式** | $[(\varepsilon^2+\varepsilon'^2)(v_1v_2-1)+\hbar^2\omega^2/\gamma^2]/(2\varepsilon'^2)$ |
| eq. 7.1（trace 核） | **(2.42) 第二式** | $-\frac{m^2}{\varepsilon\varepsilon'}\big[1+\frac{(\varepsilon^2+\varepsilon'^2)\gamma^2(v_1-v_2)^2}{4\varepsilon\varepsilon'}\big]$ |
| 主公式（前因子 $\alpha/4\pi^2$、$\omega^2$、双时间积分、$e^{-i\Phi}$） | **(2.40)**（能量谱）＋ (2.13) $d\mathcal E=\hbar\omega\,dw$（概率→能量） | 自旋平均/极化求和对应 (2.42) 前的 "summation over spin states of the final electron and averaging over ... initial electron" |

数值核对（$m=1,\varepsilon=10,\omega=3,\mathbf v_1\!\cdot\!\mathbf v_2=0.9$，在壳）：模块 dot/trace 两核与 (2.42) 两式**逐位相等**，差 $0.00\mathrm{e}{+}00$；相位与 (2.40) 的 $\frac{\varepsilon}{\varepsilon'}(kx_2-kx_1)$ 差 $0.00\mathrm{e}{+}00$。**BUG-6 修复所补的 $\varepsilon/\varepsilon'$ 因子，正是使 trace 核回到 (2.42) 的关键**（修复前系数 $(\varepsilon^2+\varepsilon'^2)/(4\varepsilon\varepsilon')$ 与专著不符）。

三处**方法差异**（均非错误，需明确记录）：

1. **局域能量 P-2 超出专著范围**：专著 §2.3 处理非定常场时只保留 (2.33) T-指数展开的**首项** (2.34)，即"能量在形成长度内视为常数"，相位仍用单一 $\varepsilon$；本模块 P-2 用逐顶点 $\varepsilon_i,\varepsilon'_i$ 的对称化形式，是**超出该近似的推广**（专著适用域内两者一致）。这解释了为何 P-2 只能"由对称化推导、靠测试钉死"——它本就不在专著里。
2. **角度积分路线不同**：专著用超相对论小角展开 (2.43)–(2.45) 解析积掉发射角，经高斯积分 (2.47) 得 (2.46)；本模块保留完整角依赖、按锥面 Gauss–Legendre **数值**积分（README §3.4）。物理等价，模块更通用但不利用专著的小角约化。
3. **未实现自旋/极化分辨**：专著 (2.39)/(2.41) 给出含自旋密度矩阵与光子极化矢量 $e$ 的分辨核；模块只实现求和/平均后的 (2.42)（`spin_averaged`/`polarization_summed` 固定 True）。模块已声明此限制。

`eq. 9.6`（Airy 恒等式）不在 p.55–77 范围内，尚未核对出处。

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

### P-3：有限记录 —— 已重新诊断（2026-09-11），守卫已加，物理修正未做

> 原诊断（"硬截断端点辐射污染整个谱"，例：闭合圆环 $\omega/\varepsilon=0.5$ 处 $\mathrm dW/\mathrm d\omega=-4922$）**经复核不成立**。以下为实测结论（环境：Python 3.12 venv；脚本见对话记录）。

**(a) $-4922$ 是采样混叠，不是边界项。** 闭合圆环、$\gamma=10,\chi=0.5$（$\rho=198$）、$\delta=0.5$ 处反冲移动谐波 $m\approx1990$，Nyquist 判据要求 $N_t>4m=7960$。实测 $\mathrm dE/\mathrm d\omega$ 与精确值之比：

| $N_t$ | 512 | 2000 | 3980（$=2m$） | 7960（$=4m$） | 15920 |
|---|---|---|---|---|---|
| 比值 | **2126** | **−82** | 2.16 | **1.0007** | 1.0007 |

充分采样后闭环在该频率**没有**边界问题（V8 通过用的正是 $N_t=7960$）。原 $-4922$ 用的是 V6 的 $N_t=512$，欠采样 8 倍。**附带修正**：V8 docstring 的判据 $N_t>2m$ 偏松一倍（只允许每振荡一个采样点，$N_t=2m$ 实测仍差 116%）；正确判据是 $N_t>4m$。

**(b) 真正的失败模式在低 $\omega$／记录短于形成时间，且局域在端点。** 不欠采样、用同一二维角度网格对比"开弧 vs 整圈"（圆、一次谐波，比值相对 $L\,\mathrm dP/\mathrm d\omega$）：

| $L/\tau_f$ | 1.60（闭环） | 1.44 | 1.28 | 1.12 | 0.96 | 0.80 | 0.64 |
|---|---|---|---|---|---|---|---|
| 比值 | 1.000 | 0.886 | 0.719 | 0.480 | 0.209 | **−0.032** | **−0.187** |

**整数圈精确**（2.00 圈→2.000，3.00 圈→3.000），**非整数圈按缺失的偏圈亏损**（1.25 圈多出的 1/4 圈只贡献 0.1645＝额定的 66%，与独立 1/4 弧的 61% 吻合）。亏损**局域在记录两端的 $\sim\tau_f$ 邻域**，不是"污染整个谱"。$\tau_f=\big(6\varepsilon'/(\omega\Omega_{\rm eff}^2)\big)^{1/3}$ 随 $\omega$ 减小而增大——低 $\omega$ 最严重，恰是本模块承诺的软光子区。直线记录 $\Omega_{\rm eff}=0\Rightarrow\tau_f=\infty$（判据不适用，真空减除已消去直线部分）。

**(c) 形成长度窗口（原文"首选"）不是它的解。** 窗口把被积函数乘以 $\le1$ 的因子，而有限记录下的自然权重是 $(L-|\tau|)$，窗口只会进一步削减积分。实测：$\tau_w=100$ 把 $\delta=0.5$ 的 $-62.8$ 压到 $-0.057$（好），但同时把一次谐波的 $+4.10$ 压到 $-0.139$（**杀掉物理信号**）。窗口的正确定位是性能（带状截断，§5）与非均匀场退相干建模。

**已实现（2026-09-11）**：两个运行时守卫，所有入口默认 `checks="warn"`（可 `"raise"`/`"ignore"`）——`SamplingWarning`（`sampling_margin ≥ 1`，Nyquist）与 `RecordLengthWarning`（`L/\tau_f < 1.5`）；`BKIntegrator.adequacy()` 与 `Spectrum.metadata` 给出逐 $\omega$ 数值。V1–V8 数值逐位不变，回归 35→40 例全过。

**未做**：端点亏损的物理修正。前置问题：$L\lesssim\tau_f$ 时"弧长×常数场辐射率"**本身就不是正确靶值**，需先做收敛性研究定靶，再考虑端点渐近延拓或 $(L-|\tau|)$ 权重尾部修正。

**(d) 收敛性研究（2026-09-11）：$\tau_f/L$ 标度**未获证实**，且该问题在可用测试台上是**提法不成立**的。**

原计划是在一条非周期轨迹上量"亏损 vs $L$"，验证 $\text{deficit}=c/L$。实测结论：

1. **在锐谱线处 $d\mathrm E/d\omega \propto L^2$**，与 $\Delta t$ 无关（每圈 80/200/400 采样给出逐位相同结果，margin 0.05→0.01），整数圈与非整数圈皆然。这是**正确的物理**（谱线：振幅 $\propto L$，密度 $\propto L^2$；线宽 $\propto1/L$，故线能量 $\propto L$）。后果：`reference.py` 里"一圈 $d\mathrm E/d\omega$ = 连续谱密度 $T\,\mathrm dP/\mathrm d\omega$"的对应**只对一圈成立**；对多圈记录拿 $L\cdot\mathrm dP/\mathrm d\omega$ 作靶是**范畴错误**——我最初的"亏损表"正是踩了这个坑。**后续认识（2026-09-11，见 [§8.1](#81-圈间相干一个已经测到但当时标错了名的-lcfa-判别信号2026-09-11)）**：同一个 $L^2$ 标度换到"BK vs LCFA"的视角下，就是**圈间相干的直接观测**——LCFA 管线（局域速率、蒙卡事件无相位）给不出它，定量预言是"$n$ 圈时谱线中心密度／LCFA 连续谱密度 $=n$"。同一条物理，当时按记账问题记下了。
2. **换到真正宽带（chirped）轨迹后**（$\Omega(t)=\Omega_0(1+0.5\sin(\omega_s t))$，慢调制，$\chi\in[0.5,0.75]$，用绝热靶 $\int_0^L\mathrm dP/\mathrm d\omega(\chi(t))\mathrm dt$），$d\mathrm E/\mathrm d\omega \propto L^{1.07}$（确为线性，无 $L^2$），但**亏损不服从 $c/L$**：

   | $L/\tau_f$ | 2.42 | 4.84 | 7.27 | 9.70 | 12.13 | 14.55 | 19.41 | 24.26 |
   |---|---|---|---|---|---|---|---|---|
   | deficit | 0.143 | −0.078 | 0.089 | −0.062 | 0.087 | 0.198 | −0.025 | −0.003 |
   | $\text{deficit}\cdot L/\tau_f$ | 0.35 | −0.38 | 0.65 | −0.60 | 1.05 | 2.87 | −0.48 | −0.08 |

   亏损**变号**（有时过冲），且 $\text{deficit}\cdot L/\tau_f$ 在 $-0.6$ 到 $2.9$ 之间游走，**不是常数**。主导 $L$ 依赖的是谱分辨率/相干结构，不是光滑的 $c/L$ 尾巴。
3. 唯一稳健、可复现的现象仍是最初那条：**$L\lesssim\tau_f$ 时谱塌缩（可变负）**——守卫的 1.5 阈值正是据实测塌缩边界定的，不依赖任何标度模型。

**结论**：**不要基于 $c/L$ 模型做端点修正**。若要继续，需要一条真正非重复（宽带/随机或真实 PIC）轨迹，且靶值必须由 $L\to\infty$ 收敛性定义，而不是解析速率；在拿到那种测试台之前，守卫（已实现）是恰当的终点。

### P-4：默认锥角 —— 已解决（2026-09-11），且**原文的方向判断有误**

> 原文："BK 的特征发射角随光子能量按 $(\varepsilon/\varepsilon')\,/\gamma$ 展宽（ω→ε 时发散）；默认固定锥 $5/\gamma$ 对**高 $\omega$** 会系统性低估角度积分。"**这两句都不成立。**以下为实测（$\gamma=10,\chi=0.5$ 圆、整数反冲移动谐波、BK 核、收敛角网格）。

**(a) 发射半角在低 $\omega$ 处变宽，方向与原文相反。** 相位角向项的驻相标度给出

$$\theta_c=\Big(\frac{4\varepsilon'\Omega_{\rm eff}}{\omega\varepsilon}\Big)^{1/3}=\Big(\frac{4}{m}\Big)^{1/3},\qquad m=\frac{\omega\varepsilon}{\varepsilon'\Omega_{\rm eff}},$$

实测半能角 $\approx0.35\,\theta_c$，且 $\theta_{50}\gamma m^{1/3}=4.6$–$6.2$ 在 $\delta=0.02$–$0.7$ 内近似恒定（$m^{-1/3}$ 标度成立）：

| $\delta$ | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|---|---|
| $\theta_c\gamma$ | 4.62 | 3.37 | 2.62 | 2.00 | 1.67 | 1.26 | 0.95 |
| $\theta_{50}\gamma$（实测） | 1.79 | 1.26 | 0.98 | 0.70 | 0.60 | 0.37 | 0.33 |

即**软光子端锥最宽**（$\delta=0.02$ 处 $\theta_c\approx4.6/\gamma$），而 $5/\gamma$ 在 $\delta\gtrsim0.5$ 处反而宽裕。被截掉的正是本模块承诺的软光子区（§8.1）。

**(b) 第二个、独立得多的原因：速度摆幅。** 角积分绕**固定轴**（默认 $\beta(0)$）做锥面积分，但记录上每一段辐射都绕**瞬时**速度方向。发射方向最远到 $\text{swing}+\theta_c$，其中 $\text{swing}=\max_t\angle(\boldsymbol\beta(t),\boldsymbol\beta(0))$——闭环轨道是 $2\pi$，固定锥只捕到约 1%。**任何固定小锥都无效**，与 $\omega$ 无关。

**(c) 实测总账**（$\gamma=5,\chi=0.5$、$0.75$ rad 弧、$N_t=2\times10^4$、经典 LW 角分布 + 二维 $96\times48$ 收敛参考；LW 的角结构与 BK 相同——核不依赖 $\mathbf n$，$\mathbf n$ 只进相位）：

| $\delta$ | $\theta_c\gamma$ | 自适应 $\theta_{max}$ | 旧默认 $5/\gamma$ | 新默认（自适应 + 两板） |
|---|---|---|---|---|
| 0.10 | 2.64 | 2.333 | **−99.69%** | −0.79% |
| 0.30 | 1.68 | 1.760 | **−92.92%** | −1.02% |
| 0.50 | 1.27 | 1.511 | **−91.19%** | −1.58% |

**(d) 已实现（2026-09-11）**：

1. `integrator.default_theta_max` $=\min(\pi,\ \text{swing}+3\max_\omega\theta_c)$，下限为历史 $5/\gamma$（**绝不比过去更紧**）；`theta_max=None` 时由 `compute_spectrum` 自动使用。取 $\omega$ 网格上的最大值，是为了保住"一份方向网格在频率间共享"的 numba 批处理路径。
2. `cone_directions` 增加**两板**选项（`split`/`n_inner`）：$1/\gamma$ 内芯单独用一组 Gauss–Legendre。这是**必需**而非优化——单板跨全空间时 $n_\theta=16$ 在 $\delta=0.5$ 处角积分错 **97%**、$n_\theta=32$ 仍错 13%，要到 $n_\theta\approx64$ 才收敛；分板后 $24+8$ 个节点即与收敛值逐位相同。
3. 第三个运行时守卫 `AngularConvergenceWarning`（默认 `checks="warn"`）：把 `n_theta`/`n_phi`/内板节点数加倍，在**承载谱量最多**的 $n_{probe}$ 个频率上重算角积分（按 $|\mathrm dE/\mathrm d\omega|$ 选点），**相对峰值**的变化 $>2\%$（`ANGULAR_RTOL`）即报警；开销约 $4n_{probe}/N_\omega$。阈值与实测精度边界吻合（$\gamma=5,\chi=0.5$、$\delta=0.3$、`axis=z`：$n_\theta=12$ → 角积分高 4.6% → 报警；$n_\theta=16$ → 高 0.0% → 静默）。**探针与判据在 2026-09-11 改过**：最初是"min/median/max 三点 + 逐点相对"，在 Test B（闭环线谱）上被线间相消残差带出 $>140\%$ 的**误报**（线中心只动 $1.9\times10^{-5}$），故改为"权重最大探针 + 峰值相对"。已知局限见 README §3.7（线谱需由驱动器自带针对线中心的检查）。另有**免费**诊断 `metadata["angular_edge_fraction"]`（最外层节点承担的份额）及 `["velocity_swing"]`、`["emission_half_angle"]`。
4. **V1–V8 数值逐位不变**，回归 40→48 例全过。原因：自适应与两板都只在 `theta_max=None` 的默认路径启用；显式传 `theta_max` 的调用（全部验证与示例都是如此）走历史单板路径。

**(e) 边界（诚实记录，未做）**：固定轴锥面只在**摆幅远小于锥角**时有效。闭环轨道（发射是绕轴的环）即使自适应锥开到 $\pi$、$n_\theta=16$ 也不收敛（$\mathrm dE/\mathrm d\omega$ 是精确值的 2.39 倍），此时须显式传 `axis=`（对称轴/束轴，如 V6 传 `axis=[0,0,1]`）或更大的 `n_theta`。新守卫会**报警**而不是静默返回错值（旧默认在此情形下静默返回约 1% 的量）。要根治需要沿速度路径自适应布置方向网格，属于新方法而非参数调整。

**方法论附带发现**：角积分的收敛性研究**必须在整数反冲移动谐波上做**。用非整数 $m$ 时闭合轨道出现端点项，角积分看起来随节点数乱跳、不收敛（实测：$m=852.8$、$N_t=6823$ 时 $n_\theta$ 从 8 增到 128 结果在 $\pm20\%$ 间振荡；换成整数 $m=853$ 后 $n_\theta\ge24$ 即收敛到 1e-5）。

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

1. ~~**形成长度带状截断**（复杂度降到 $O(N_t\,N_\omega)$）：只对 $|t_2-t_1| \lesssim 2 l_f(\omega)$ 的带内求和——现在是最大的**性能**杠杆。注意它**不是 P-3 的解**（§4）：窗口只会进一步削减积分、加重端点亏损；它解决的是代价。~~ ⛔ **2026-09-14 判定不做**，见 §8.2：判据实验显示被积函数的滞后分布不衰减，截断掉的是相消项而非尾巴，在闭合与非周期两种记录上都不收敛。当前最大的性能杠杆改为**周期性分块**（仅适用于封闭/准周期轨迹）。
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

   **V8 的定位与盲区（2026-09-11 澄清，避免重复困惑）**：V8 的对照物 `reference.quantum_synchrotron_rate` **就是项目的 LCFA 速率**——恒定场里两者是同一个函数（Macdonald ≡ Airy ≡ `optical_depth_tables.gen_photon_prob_rate_for_delta`，实测差 $10^{-15}$），这不是巧合而是 LCFA 的定义。因此 V8 **不可能**区分 BK 与 LCFA，它检验的是**实现**（前因子、反冲相位、$\omega^2/\gamma^2$ 接触项、dot/trace 核），不是理论差异；它的价值在于"两者必须一致区"过关，否则 BK 与 LCFA 的任何差别都分不清是物理还是 bug（BUG-6 就是这样被抓出来的）。V8 看不见 LCFA 缺失的东西，原因见 [§8.1](#81-圈间相干一个已经测到但当时标错了名的-lcfa-判别信号2026-09-11)：场恒定 + 只用一圈 + 对照点取在谐波中心。另外要分清两个层次——V8 对照的是 LCFA 的**速率**（解析闭式），不是 LCFA 的**用法**（沿轨迹积分／蒙卡事件采样）；恒定 $\chi$ 下两者数值等价，所以 V8 对后者同样无话可说。
5. ✅ **P-2 局部能量核**（2026-09-09 完成）：把 $N(t_1,t_2)$、$\Phi$ 推广到 $\varepsilon(t_1),\varepsilon(t_2),\varepsilon'_i = \varepsilon_i - \omega$（改动在 kernel.py + phase.py + integrator.py + types.py，`Trajectory` 增加可选能量历史字段 `energy`；局部形式由对称化推导、靠测试钉死，详见 §4 P-2）；
6. ✅ **P-3 重新诊断 + 适用性守卫**（2026-09-11）：原例 $-4922$ 实为**采样混叠**（欠采样 8 倍）；真实失败模式是低 $\omega$ 的端点亏损。已加 `SamplingWarning`/`RecordLengthWarning`（§4 P-3）。**未做**：端点亏损的物理修正——先做收敛性研究定靶（$L\lesssim\tau_f$ 时"弧长×常数场辐射率"**本身就不是正确靶值**），再考虑端点渐近延拓或 $(L-|\tau|)$ 尾部修正；
7. ✅ **P-4 自适应锥角 + 角度守卫**（2026-09-11 完成）：`default_theta_max` = $\min(\pi,\text{swing}+3\max_\omega\theta_c)$、`cone_directions` 两板规则、`AngularConvergenceWarning`。**原文"高 $\omega$ 漏尾巴"的方向判断被实测推翻**——是软光子端被截（§4 P-4）。未做：沿速度路径自适应布网格（闭环情形的根治）。

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

### 8.1 圈间相干：一个**已经测到、但当时标错了名**的 LCFA 判别信号（2026-09-11）

**LCFA 缺失的东西只有一个根源：相位。** BK 保留完整的双时间相位

$$\iint \mathrm dt_1\mathrm dt_2\;N(t_1,t_2)\,e^{-i\Phi(t_1,t_2)},$$

而 LCFA 管线是局域、非相干的 $\int \mathrm dt\;\dfrac{\mathrm dW_{\rm LCFA}}{\mathrm dt\,\mathrm d\omega}\big(\chi(t)\big)$——蒙卡事件生成尤其如此，**事件之间没有任何相位关系**。这一个差别在三种情形下显形：形成长度内的场变化、离散谐波／圈间干涉、有限磁铁的边缘辐射。§8 上面第 2、3 条说的就是它们。

**圆轨道上的具体形式。** $n$ 圈记录的谱线宽度与间距分别是

$$\Delta\omega_{\rm line}=\frac{2\pi}{L}=\frac{\Omega}{n},\qquad \Delta\omega_m=\Omega\left(\frac{\varepsilon'}{\varepsilon}\right)^2
\quad\Longrightarrow\quad \text{分辨条件 }n\gg\left(\frac{\varepsilon}{\varepsilon'}\right)^2=\frac{1}{(1-\delta)^2}.$$

$\delta=0.5$ 处需要几十圈；$\delta=0.05$ 处一圈就够。

**这个效应我们已经测到过——就是 P-3(d) 第 1 条那个 $\mathrm dE/\mathrm d\omega\propto L^2$。** 当时它的标签是"拿 $L\,\mathrm dP/\mathrm d\omega$ 作靶是范畴错误"（定靶失败），但同一个标度换到"BK vs LCFA"的视角下就是**圈间相干的直接证据**：谱线密度 $\propto L^2$（振幅 $\propto L$ 相干叠加），线宽 $\propto1/L$，线能量 $\propto L$。两条说法都对——它是一个真实的物理效应，只是当时被当成记账问题记下了。**注意"$n\to\infty$ 收敛到连续谱"这句要拧紧**：固定 $\omega$ 处的密度是 $\propto n^2$ **发散**的，收敛的只是**包络**（或按线能量归一后的密度），这正是当初的范畴错误的来源。

**干净的定量预言**（可直接检验）：

$$\frac{\text{BK 谱线中心密度}}{\text{LCFA 连续谱密度}}=n,$$

$n=1$ 时比值为 1——那正是 V8 的结果。

**V8 看不见它的三条原因**（按重要性）：(i) `circle_trajectory` 默认**一圈**，而 $\delta=0.5$ 处线宽 $\Omega$ 比线间距 $0.25\Omega$ 还**大 4 倍**，相邻谱线完全重叠，离散结构早已糊掉；(ii) 对照点选在**谐波中心**，而"一圈密度在谐波中心恰等于连续谱密度"就是 [reference.py](lambdapic/src/lambdapic/core/qed/baier_katkov/reference.py) 那条"无 Jacobian"论断本身——**这是两理论按构造相等的唯一一点**；(iii) 靶取的是连续谱包络。

**成本与钥匙。** $n$ 圈要求 $N_t\propto n$，而双重积分是 $O(N_t^2)$，故代价 $\propto n^2$（20 圈 $=400\times$，约每点一小时）。~~**带状截断**（核按 $\lvert t_1-t_2\rvert\lesssim\tau_f$ 局域，而 $\tau_f\ll T$）把它降成 $O(N_t)$，即 $\propto n$——20 圈从一小时变三分钟。这是带状截断值得排在 PIC 集成之前的**第二个**理由（第一个是多粒子成本，见 §7-D 第 12 条）。~~

> ⛔ **上面这段已被 §8.2 的判据实验推翻（2026-09-14）**：$\tau_f\ll T$ 恰恰是问题所在——相干要求保留跨周期配对，而被积函数的滞后分布根本不衰减，截断掉的不是尾巴而是相消项。带状截断不但不能解锁本扫描，反而正是破坏它的那件事。$n$ 圈扫描的候选工具改为**周期性分块**（§8.2 结论 3）。

**必须诚实的两点边界**：

1. $n$ 圈相干叠加要求**这 $n$ 圈中间不发生辐射**——一旦某个圈发了一个光子，$\varepsilon$ 变了，后续圈的相位关系就断了。所以 $n^2$ 增强是**理想化上界**，真实上限由每圈发射概率、能量展宽与场的不完美共同压制，圆轨道上还有辐射损失导致的轨道收缩。这个效应真正有观测意义的场所是 **undulator**（$N_w$ 有限但可控，谐波结构实打实被测到）。
2. **LCFA 速率本身没有错。** 在恒定场里它与精确同步辐射谱逐位相同（实测 $10^{-15}$，且项目 LCFA 表用的就是同一个函数）。错的是把它当成能给出谱线结构的工具；只要观测分辨率粗于线间距，LCFA 完全够用。

**判据实验：已做（2026-09-11），三个标度同时成立。** 见新文件 [coherence.py](lambdapic/src/lambdapic/core/qed/baier_katkov/coherence.py)（测量）+ [coherence_plot.py](lambdapic/src/lambdapic/core/qed/baier_katkov/coherence_plot.py)（图，独立于 `validation_summary.png`）。参数：$\gamma=5,\chi=0.5,\delta=0.5$（$\rho=48$、$m=245$），$n=1,2,4,8,16$，$N_t=4.4\,m\,n$。**取 $\gamma=5$ 而非 10**，因为 $m\propto\gamma^3$（245 vs 1990），而线宽以线间距为单位时与 $\gamma$ 无关——代价降一个量级而信号不变。

| $n$ | 线中心密度 /（$T\,\mathrm dP/\mathrm d\omega$） | $n^2$ | $\Gamma$ | $n$ | FWHM/线间距 | $1/n$ | 线能量比 | $n$ |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0030 | 1 | 1.0030 | 1 | 不可测 | 1.000 | 0.834 | 1 |
| 2 | 4.0120 | 4 | 2.0060 | 2 | 0.478 | 0.500 | 1.981 | 2 |
| 4 | 16.0480 | 16 | 4.0120 | 4 | 0.227 | 0.250 | 4.009 | 4 |
| 8 | 64.1920 | 64 | 8.0240 | 8 | 0.1115 | 0.125 | 8.021 | 8 |
| 16 | 256.7679 | 256 | 16.0480 | 16 | 0.0555 | 0.0625 | 15.97 | 16 |

**结论**：

1. **$n^2$ 律成立，指数精确为 2**：偏差恒为 $+0.30\%$ 且**与 $n$ 无关**（1.0030 / 4.0120 / 16.0480 / 64.1920 / 256.7679），这个常数是角积分量子化因子（对所有 $n$ 相同），不是标度偏离。$\Gamma=n$ 随之成立（$=1.0030\,n$）。
2. **线宽 $\propto1/n$**，实测常数 $0.444(\varepsilon/\varepsilon')/n$。**朴素 Dirichlet 估计给它的两倍**（$|\sin(ny)/y|^2$ 半高在 $ny=1.3915$），该因子 $2$ 未解——故只报实测常数，不作解析断言。
3. **线能量 $\propto n$**（$n=1$ 处 0.834 偏小，因为一圈线宽超过谱线间距、邻线溢出窗口；$n\ge2$ 后线性到 0.2% 内）。

**方法论收获（两条，都是踩过的坑）**：

- **$n=1$ 的 Dirichlet 核恒等于 1**（$\sin(nx)/\sin x$ 在 $n=1$ 时为恒等），所以一圈记录**根本没有线结构**——$\delta=0.5$ 处一圈线宽 $\Omega$ 比线间距 $0.25\Omega$ 大 4 倍，谱是彻底糊掉的连续谱。这才是 V8 看不见此效应的**根本原因**，而不是"对照点没选好"。V8 是这条测量在 $n$ 轴上的零端点（$\Gamma(1)=1.003$ 正是 V8 的比值）。
- **角守卫在线谱上会误报。** 通用 `AngularConvergenceWarning` 按 $|\mathrm dE/\mathrm d\omega|$ 选探针，而线谱的**线间**值是相消残差：实测那里角网格微扰带来 $>140\%$ 的变化，而线中心只动 $1.9\times10^{-5}$。驱动器因此**只屏蔽这一条警告，并用 `coherence.verify_angular_convergence`（针对线中心的收敛检查）替代**——不是放宽阈值掩盖。守卫本身也已从"逐点相对 + min/max/median 探针"改成"**峰值相对 + 权重最大探针**"（§4 P-4 (d) 第 3 条）。

**未做**：带状截断。$\gamma=5$ 下 $\sum n^2$ 的代价可承受（全扫描约 5 分钟），所以本实验**没有**验证"带宽取 $3\text{–}5\tau_f$ 不破坏 $n^2$ 增强"——那条仍然悬着，等带状截断实现后补测。

---

## 8.2 带状截断的判据实验（2026-09-14）：**结论是"不做"**

> 起因：§5 P2 与 §8.1 把"形成长度带状截断"列为下一个工作项，§8.1 更断言它是"解锁 $n$ 圈相干扫描的钥匙"（把 $\sum n^2$ 降成 $\propto n$，20 圈从一小时到三分钟）。做之前先验前提。**测量推翻了这条断言，并且比预期更强：带状截断在测过的每一种情形下都不是受控近似，不应实现。**
>
> 方法：模块外一次性脚本（`/tmp/band_probe.py`、`/tmp/band_curve.py`、`/tmp/band_block.py`，均未入库），用模块公开件自己算带掩码的双时间和。参数同 Test B：$\gamma=5$、$\chi=0.5$、$\delta=0.5$，$T=307.81$、$\tau_f=24.66$、$\tau_f/T=0.0801$、$m=245$、$\omega_m=2.500260$。**harness 自校验：无掩码时复现 `compute_spectrum` 到 $1.17\times10^{-12}$**，且 $n=1,2,4,8$ 的全和比值逐位复现路线图的 1.0030 / 4.0120 / 16.0480 / 64.1920。

**(a) 滞后分布根本不衰减（决定性）**。在每个反冲移动谐波处，把第二个顶点整体前进 $d$ 个周期的块积分
$$M(d)=\sum_{i,j\in\text{一圈}}w_iw_j\,N\,e^{-i\Phi(t_i,\,t_j+dT)}$$
实测 $|M(d)|/|M(0)|=\mathbf{1.000000}$、相位 $=0$ 精确成立，$d=0\dots15$。这是解析必然：轨迹严格周期使 $\Phi(t_1+T,t_2+T)=\Phi(t_1,t_2)$，而谐波条件 $\omega\varepsilon/\varepsilon'=m\Omega$ 给出块相位 $\Delta\Phi=2\pi m d$（与方向 $\mathbf n$ 无关）。**即 $n^2$ 律就是 $n^2$ 个等幅同相块的相干叠加**，"远端配对可以丢"与它就是同一句话的反面。

**(b) 截断曲线 $C(W)/C(\infty)$——小于整条记录都不收敛**（$C(W)$ 为限制 $|t_j-t_i|\le W$ 的角积分谱；列名中 τ 为单位 $\tau_f$，T 为单位周期）：

| 记录 | $1\tau$ | $3\tau$ | $5\tau$ | $12\tau$ | $0.5T$ | $1T$ | $2T$ | $3T$ |
|---|---|---|---|---|---|---|---|---|
| 圆 $n=1$，线中心 | 0.103 | 0.305 | 0.516 | 0.909 | 0.877 | 1.000$^*$ | — | — |
| 圆 $n=4$，线中心 | 0.152 | 0.453 | 0.766 | 1.348 | 1.301 | **0.996** | **1.168** | **3.452** |

$^*$ $n=1$ 的记录只跨 $1T$，$W\ge T$ 即全和。$n=4$ 时 $W=3T$ 已覆盖记录的 75%，截断和仍是全和的 **3.45 倍**，且 $C(W)$ 随 $W$ 上下乱跳（有符号部分和，非单调）。**被截掉的不是尾巴，是相消项**——闭合轨道上每个周期贡献完全相同，求和只靠整条记录上的相消才收敛。

**(c) 朴素带破坏 $n^2$ 律，且不是"差一个系数"**。$\Gamma=$ 带和/$(n\cdot\text{LCFA})$ 在固定 $B$ 下**与 $n$ 无关**（$B=3$：1.72 / 1.64 / 1.60；$B=2$：5.22 / 5.57 / 5.75），即带和 $\propto n$、全和 $\propto n^2$。判据是"$\Gamma$ 变成 $n$-无关"，**不是 $\Gamma\to1$**——$B=3$、$n=1$ 时 $\Gamma=1.72$ 已经接近教科书式的 1，极易被误读为"收敛良好"。同时 $\Gamma$ 强烈依赖 $B$（$1.6$–$5.7$），说明结果由带宽任意决定。

**(d) 梳状带 $|t_2-t_1-kT|\le B\tau_f$ 既不准也不省**。$k$ 取 $0\dots n-1$ 的窗口并集：`comb/full` $=3.06$（$B=1$）、$1.59$（$B=3$）、$2.45$（$B=5$）——**比全和还大**，随 $B$ 增大才降向 1（因为宽窗才把相消项放回来）。每行必须访问 $n$ 个窗口，保留的对数占比 16% / 47% / 79%（$B=1/3/5$），对全和的加速比是**常数** $T/(4B\tau_f)=3.12/B$——$B=3$ 时已无收益，$B=5$ 时**比全和还慢**。原写法只取 $k\ge0$、破坏厄米对称，需折到最近周期。

**(e) 非周期记录同样不收敛（但测试台本身有限）**。chirped 记录（$\Omega(t)=\Omega_0(1+0.5\sin\omega_s t)$）上 `banded/full` 在 $B\le10$ 内于 $-0.95$ 到 $5.14$ 之间乱跳、含负值。诚实边界：该记录只有 $4T$ 长，而 $T/\tau_f=12.5$，$B\le10$ 的带宽必然短于一个周期，**这个测试台无法充分判定宽带情形**——但也没有给出任何支持截断的证据。

**(f) 附带发现：线中心的双时间和是量级 $10^5$ 的相消**。把单圈块的首样本权重从 $\Delta'/2$ 改成 $\Delta/2$（相对变化 $\sim5\times10^{-4}$）后块和由 $+2.08\times10^{-2}$ 翻成 $-4.2\times10^{-3}$。$\sum|\text{terms}|/|\text{value}|\approx2.5\times10^5$（独立复现）。**任何把双时间和拆成子和的方案（截断、分块）都会继承这个放大因子**：全和稳定是因为相消发生在求和内部，部分和不是。

**结论与后续**：

1. **不实现带状截断**。它既不能解锁 $n$ 圈扫描（那正是它破坏的东西），在测过的非周期情形下也拿不出可信带宽；而"取够大的 $B$"会在 $B\gtrsim3$ 时把收益吃光。
2. §8.1 的"钥匙"说法**作废**，§5 P2 的该项**关闭**。若将来仍要做 PIC 加速，必须先在一个**真正宽带、且长度 $\gg T$ 的**测试台上重新做 (b) 的截断曲线——那是本项重启的前置条件。
3. $n$ 圈扫描的候选工具只剩**周期性分块**（$I_n=\sum_d(n-|d|)M(d)$，$O(n)$ 次单圈积分，连续时间下严格）。但 (f) 是新出现的前置障碍：单圈块和本身就是相消量，直算不稳；上网格还必须与 $T$ 可约（否则块相位差 $2\pi m/(N-1)$，$n=2$ 时 $1.43$ rad）。**在动这个工具之前，先解决块和的数值稳定性**。
4. 方法学：本项目已连续三次出现"方向判断被实测推翻"（P-3、P-4、本节）。**凡涉及"某个近似可以丢"的判断，先做单变量截断曲线，再谈实现。**

---

## 9. 行动清单速查

| 优先级 | 事项 | 工作量 | 对应章节 |
|---|---|---|---|
| ~~P0~~ ✅ | ~~修 BUG-1（recoil 静默忽略）~~ 已完成 2026-09-08 | — | §2 |
| ~~P0~~ ✅ | ~~修 BUG-2/3/4（校验与守卫）~~ 已完成 2026-09-08 | — | §2 |
| ~~P0~~ ✅ | ~~修 BUG-5（前因子 4π）+ pytest 回归~~ 已完成 2026-09-08 | — | §2, §7-A |
| P0 | 补设计文档/文献出处（架构/函数说明已补：README.md 重写为中文详细文档 2026-09-09；**文献出处已补 2026-09-11：与 BKS 专著 (2.25)/(2.40)/(2.42) 的编号对照已建立，含三处方法差异记录，见 §3**；剩余：物理推导设计文档 `Baier-Katkov.md`） | 0.5–1 d | §3 |
| ~~P1~~ ✅ | ~~量子同步辐射交叉验证~~ 已完成 2026-09-08（BK/精确 0.997–1.001；顺带修 BUG-6 trace 核） | — | §4 P-1, §7-B |
| ~~P1~~ ✅ | ~~采样守卫 + 记录长度诊断（P-3 重新诊断）~~ 已完成 2026-09-11（`SamplingWarning`/`RecordLengthWarning`，所有入口默认开；$-4922$ 实为采样混叠，Nyquist 判据 $N_t>4m$） | — | §4 P-3, §7-B |
| ~~P1~~ ✅ | ~~P-4 自适应锥角 + 两板网格 + 角度守卫~~ 已完成 2026-09-11（原文方向判断有误：软光子端被截而非高 $\omega$；旧默认在 $0.75$ rad 弧上漏 92–99.7%，新默认 ←1.6%。V1–V8 逐位不变，回归 40→48 例） | — | §4 P-4, §7-B |
| P2 | 端点亏损修正：**收敛性研究已做（2026-09-11）——$\tau_f/L$ 标度未获证实，$c/L$ 模型不可用**（§4 P-3 (d)）。若继续，需先建宽带/真实 PIC 非重复测试台，靶值由 $L\to\infty$ 收敛性定义 | 3–5 d | §4 P-3 |
| ~~P2~~ ✅ | ~~局部能量核 ε(t)~~ 已完成 2026-09-09（`Trajectory.energy` + 逐顶点核/相位，双后端；固定路径未动） | — | §7-B |
| ~~P2~~ ✅ | ~~Numba 化~~ 已完成 2026-09-08（约 140×） | — | §5 |
| ~~P2~~ ⛔ | ~~带状截断（$O(N_t N_\omega)$）~~ **已判定不做（2026-09-14）**：判据实验（§8.2）显示滞后分布不衰减（$|M(d)|/|M(0)|=1$ 精确）、截断曲线小于整条记录都不收敛（$W=3T$ 时偏离 245%）、朴素带破坏 $n^2$ 且 $\Gamma$ 由 $B$ 任意决定、梳状带既不准也不省（加速比 $T/(4B\tau_f)=3.12/B$）。原"是相干扫描钥匙"的说法作废 | — | §8.2 |
| P2 | 周期性分块（$O(N_tN_1)$ 的 $n$ 圈扫描）：**§8.2 后 $n$ 圈扫描仅剩的候选**。前置障碍：单圈块和是量级 $10^5$ 的相消量、直算不稳定（§8.2 (f)）；上网格须与 $T$ 可约 | 2–3 d（含稳定性） | §8.2 |
| ~~P2~~ ✅ | ~~圈间相干判据实验~~ 已完成 2026-09-11（新文件 `coherence.py` + `coherence_plot.py`）：$n^2$ 律（偏差恒 $+0.30\%$，与 $n$ 无关）、$\Gamma=n$、线宽 $\propto1/n$、线能量 $\propto n$ **三个标度同时成立**。**未做**：带状截断不破坏 $n^2$ 的验证（$\gamma=5$ 下代价可承受，未启用截断） | — | §8.1 |
| ~~P1~~ ✅ | ~~第二轮审查新缺陷~~ **已完成 2026-09-12**：BUG-7 角度边界诊断指向靠轴节点（实测把承担 38.6% 的边缘节点报成 0.033，掩盖了唯一该报警的信号）、BUG-8 $\omega=0$ 静默 NaN、BUG-9 `Trajectory.charge/mass` 死字段（字段已删，改 `TypeError`）、`Trajectory` 缺 time 单调性与有限性校验、metadata 在 `split` 非法时谎报网格。补 **7 个 pytest 用例**钉住行为，回归 **51→58 例**全过、V1–V8 数值逐位不变 | 0.5 d | §2.5 |
| ~~P1~~ ✅ | ~~提交未落库的成果~~ 已完成 2026-09-14：拆成 `411a63f`（09-11 成果）+ `2563766`（本轮审查修复）两个提交，工作区干净 | — | §2.5 |
| P3 | 单位适配器 + 轨迹记录回调 + 系综求和 | 2–3 d | §7-D |
| P4 | 在线事件采样 / 自旋分辨核 / NUFFT | 周级 | §7-D/E |

