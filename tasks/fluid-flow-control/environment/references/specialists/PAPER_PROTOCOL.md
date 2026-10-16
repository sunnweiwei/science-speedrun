# HydroGym 论文依据与复现协议核实

更新：2026-09-06。状态：**证据清单已固定，完整可执行评测协议尚未认证，未宣称复现成功。**

本轮只核实、记录和测试清单一致性；没有启动新训练、AutoResearch、
修改已有实验或改写冻结数据。配套机器可读记录是
[paper-targets.json](paper-targets.json)，精确覆盖现有 47 个工况。

后续用户已批准[分阶段 baseline 计划](STAGED_BASELINES.md)：先复现有对应论文
结果的工况，成功后把验证的方法适配到其他工况，作为我们自己的 reference。
下文“是否接受新 reference”的讨论项因此已得到方向性确认；具体测试窗口等
未恢复细节仍不能冒充作者设置。本文件其余部分保留核实当时的证据与未决信息。

## 1. 已固定的边界与本次结论

- 数据沿用 `hydrogym-public-500gb-v1`：47 个资源变体，730 个文件，
  488,770,756,805 字节。训练和测试共享这份来源数据；不扩展或删减工况。
- 原始数据内容及默认 YAML 是冻结的来源证据，不等于“已经确认的论文实验配置”。
  下一步的纸面复现覆盖参数必须单独记录，不覆盖这些原件。
- 先做 organizer 的单工况训练、单工况固定策略回放。之后的 AutoResearch
  才要求一个共同训练的 checkpoint 覆盖所有工况；不能用专家模型分发器代替。
- 保留现有 `hydrogym-policy/v1` 交付接口。独立评估中模型和训练得到的
  normalizer 不更新，允许策略使用显式 episode 内状态。
- 当前 benchmark 的主指标仍为每个工况的平均未折扣 native episode return。
  这是我们冻结的统计口径，不自动等于论文训练图纵轴；没有来源明确、同口径的
  reference anchors 前，不建立论文分数归一化总榜。

47 个工况的证据覆盖：

| 类别 | 数量 | 含义 |
| --- | ---: | --- |
| 找到同工况学习曲线，setup/统计口径仍需对齐 | 5 | 2D Pinball Re100、3D Pinball Re150、2D Cylinder Re1000/Re3900、2D SquareCylinder Re200 |
| 找到同系列曲线，但具体工况配置未确定 | 1 | Kolmogorov：Fig.3b 的 Reynolds、奖励权重等未定位 |
| 未定位到该精确变体的可对应学习曲线 | 41 | 包含只有物理验证、其他 solver/控制方式或其他 Reynolds 的相关结果 |
| 有同维度/Re 的精确无控制流场表格数值 | 15 | 与上三类重叠；不是 RL 分数，也不是对当前网格/重启文件的认证 |
| 已获得完整同口径、可用于自动验收的论文 RL 标量目标 | 0 | 所有 acceptance target / tolerance 保持 null，不伪造精确分数 |

这不是“其余工况没有价值”，也不是已证明作者没有这些结果；
而是在本次检查的公开资料中，尚不能给它们填入可审计的论文策略分数。
环境资源列表和论文实际展示的控制实验不是同一张表。

## 2. 证据来源与优先级

以最终 Nature 正文及补充材料为先，不把早期 arXiv 图号/叙述混入最终版。
下文 PDF 页码从封面开始计数；补充材料正文印刷页码通常比 PDF 页码小 1。

- [Nature 正文与附件入口](https://www.nature.com/articles/s41586-026-10917-6)：
  正文 17 页；补充材料 74 页。阅读正文 Methods、Fig.2/3、Extended Data
  Fig.2/3，以及 SI 4A–4D、5A–5N 中相关环境段落；图和数值表另行目视核对。
- [固定平台代码](https://github.com/dynamicslab/hydrogym/tree/4ab9854dea3d84e38a59c25e0f5835a00cf8225f)；
  [固定数据版本](https://huggingface.co/datasets/dynamicslab/HydroGym-environments/tree/cf5ef6748443ae72e9647f398487e98d421a7243)。
- [作者早期 TorchRL 示例](https://github.com/cl126162/TheHydroGymProject/tree/20a483ecc6ea711f6b380fff05f212ca4d7e2c67)：
  可作为实现证据，不能自动认定就是最终论文的训练命令。
- [相关 Kolmogorov 实现](https://github.com/smokbel/Controlling-Kolmogorov-Flow/tree/45563a836b2eb3ad8ba319ec873b7b2df4802585)：
  是相关实现，不是已经证实的 Fig.3b 原始程序。

PDF SHA256：

```text
article.pdf       ba9a67f3507ea353655f236824e88a95ddde8f709032f2b86766d65acf2b1e09
supplementary.pdf e985c2916810e795aeaf65f9a8581f806ab1f76647e46deced4ba369889cbbf7
```

本地原件保留在 `/mnt/workspace/hydrogym/paper/`，不需要复制进 Git。
本次 `hf datasets info` 核实了固定数据 revision。
Nature 页面在网页读取工具中遇到身份跳转，正文核实使用已保存的完整最终版 PDF。
早先对公开附件、Zenodo 代码包和 Hub 文件清单的检查记录保留在
[REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md#public-source-recheck-2026-09-06)：
未定位完整原始学习曲线、逐种子测试分数和策略权重归档，不能表述为它们不存在。

## 3. 论文真正报告了哪些可比结果

以下训练图终点是目视量级，**不是精确数值、误差容限或独立测试分数**。
范围只用于防止量纲误读；正式验收不读取这些估计值。
图中 episode 横轴范围也不等于已经恢复了完整 collector 配置。

| 记录 ID | 论文位置与工况 | 方法 | 训练图覆盖 | 训练图终点粗读 | 不能省略的限定 |
| --- | --- | --- | --- | --- | --- |
| ed2a | Extended Data Fig.2a；2D Pinball Re100 | PPO/DDPG/TD3 | 约 500 episodes | 各方法约 -0.2 至 0 | 正文称约 90% 减阻；未给出与我们 episode 平均一致的精确窗口 |
| fig2a | Fig.2a；3D Pinball Re150 | PPO/DDPG/TD3 | 约 500 | 各方法约 -0.35 至 0 | 不能从重叠曲线可靠判定最佳方法；不能当原始 return |
| ed2c | Extended Data Fig.2c；2D Cylinder Re3900 | PPO/DDPG/TD3 | 约 2,500 | 约 -430 至 -390 | 纵轴有 ×10²，不能读成 -4 |
| ed3a | Extended Data Fig.3a；2D Cylinder Re1000 | 从头 PPO / transfer PPO | 约 2,000 | 从头 PPO 约 -380 至 -320 | 单工况复现对应从头训练线，不把 Re200 预训练算成相同训练资源 |
| ed3c | Extended Data Fig.3c；2D SquareCylinder Re200 | 从头 PPO / transfer PPO | 约 3,000 | 从头 PPO 约 -245 至 -225 | 图内标签明确是 Re200，不是 Re1000 |
| fig3b | Fig.3b；Kolmogorov 增强混合 | PPO/GPPO | 约 500 | PPO 约 25–35；GPPO 约 40–50 | 只展示一个 training seed；具体 Re、权重和统计量未恢复 |

Extended Data Fig.2/3 的图注明确：
训练曲线是 3 次独立训练的均值与 ±1 标准差；测试图是单条策略 rollout，
不是三个独立测试种子的均值。Fig.2 也报告 3 个训练种子，但不能据此编造测试
重复次数。Fig.3b 明确只有一个展示种子，其“65% 样本减少”不是测试回报提升 65%。

必须排除的错误对应：

- Fig.2c 的 gust 控制是 **3D Re1000、AOA20、G=2**，正文约 20% 载荷波动降低；
  这一资源变体不在当前 500 GB 选择中。不能移给所选 3D Re10000/50000、AOA12。
- Fig.2b 是 **3D Cavity Re7500**；Extended Data Fig.2b 是 **2D Re4200**。
  两者都不是所选 2D Re7500。
- Fig.3c–d 的 Cylinder Re3900 约 8% 减阻使用多段 spanwise MARL；
  所选资源 YAML 是一个动作输入，不能继承该结果。
- Extended Data Fig.3d 的 2D→3D transfer 明确是 **Re1000**，
  不是所选的 3D Re200/Re3900。
- 论文 38% 是 Re200000 机翼的**局部 skin-friction**降低，训练环境是
  Reτ206 open channel；不能当作所选 Reτ180 channel 的总阻力分数。
- SI Table 5 的 cube 数值是 Re300/1000；所选 Re3700 没有对应表格数值。
- SI Nozzle Fig.23 是指定注入条件下的数值/实验验证，不是 RL controller leaderboard。

## 4. 主目标、物理诊断与实现差异

所有主指标最终仍按每个工况固定的 native reward 累加再对预先规定的 rollout 求均值。
下表记录的是论文目标/代码现状；不把未对齐的公式强行合并成同一个任务。
CD、CL、CS 分别表示阻力、升力和另一个横向力系数。

| Profile | 论文奖励/目标 | 需要一起保存的物理量 | 已核实的关键问题 |
| --- | --- | --- | --- |
| cylinder / square | -abs(CD)-ω abs(CL) | 平均 CD、升力 RMS/振幅、St | Cylinder 的 ω 在 Re100/200/1000/3900 分别 .1/.2/1/1；Square 引用同框架，需确认实际系数 |
| pinball | -abs(ΣCD)-abs(ΣCL) | 总阻力、逐柱升力、f0 | 论文 ω=1，所有所选 MAIA YAML ω=0；需按论文显式覆盖并记录 |
| cavity | -Σ((obs-mean)/std)² | 探针波动能量；可用时另存场 TKE | 代码用绝对误差；不能与论文平方误差或 Firedrake 场能量混用 |
| cube / sphere | 引用 cylinder/square 的减阻与抑升力框架 | CD、两个横向力、St | 代码另加 CS 惩罚，论文未单列该项；Sphere 8 jets 与 YAML 10 inputs 不一致 |
| naca_gust | -abs(CL-CLref)-.25 abs(CD-CDref) | 载荷偏差/峰值/波动、控制量 | G=2、1-cosine、3 个前缘喷口各长 .03c；gust 时序和 reference 平均窗口不明 |
| naca_steady | 对该 MAIA 非 gust 变体未定位完整论文 RL 目标 | CL、CD、L/D | 当前代码 CL/CD，drag <1e-6 或 lift<0 时 -100；这是代码规则 |
| dra_jet | 全局净功率节省 | 阻力、升力、驱动能耗 | 代码 jet 的 CP=0、奖励 -CD；论文 1 自由度，YAML 10 inputs |
| kolmogorov | Fig.3b 增强混合/TKE，并惩罚控制幅度 | TKE、耗散、极端事件、动作幅度 | 默认 α=0 只罚动作，不是增强混合；具体 Re 和 GPPO 实现未定 |
| jax_channel | 无控制/受控壁面剪切应力幅值之比 | 壁面剪切、速度统计、控制量 | 当前代码 -wall_shear，求解器实现亦不同 |
| nek_channel | 1-τw_ctrl/τw_ref | 壁面剪切、速度统计 | 论文控制案例 Reτ206，当前所选 Reτ180 不同 |
| nozzle | 负的推力向量角度跟踪误差范数 | 实际/目标角、壁压、动作 | 原生代码弧度；验证图角度。3D actuator 数、目标轨迹和时域未固定 |
| fd_cavity / fd_cylinder / fd_pinball | SI 3F 明确为 flow objective 的负时间积分；各工况完整训练配方未定位 | 原生 objective：扰动动能 / CD / ΣCD | 必须验证 wrapper 是否实现此积分规则，并单独绑定动作/传感器和物理时域 |

“控制量平方和”等只能先作为控制 effort 诊断，**不是已计算的物理驱动功率**。
TKE 也不能把局部探针、全场积分、空间均值、全流场动能与脉动动能视作同一量。
CL 振幅、RMS、标准差必须分别记录。时间平均的 CL/CD 不等于平均 CL 除以平均 CD。
所有诊断的空间/时间窗口及参考尺度需写入正式 protocol 后再比较。

### 无控制流场的确切数值

`paper-targets.json` 中 `physics_validation` 保存原表的 St/f0、CD 和可用的
升力统计，共 15 个 MAIA 工况。下方总表展示 CD，JSON 保留完整数值和原表位置。

这些值用于先排查网格、边界条件和初始流场，不是训练分数。
例如 3D Pinball Re150 的论文总 CD=2.949；
当前其他窗口得到不同数值时，先比较相同无控制物理量，
不能直接宣布训练模型落后几个百分点。

## 5. 47 个工况的逐项固定清单

H/K 表示“每 episode 控制步数 / 每次动作间 CFD 子步数”。
SI 列只是论文环境默认值，**不是已认证的测试时长**；
Hub 列只是来源配置文本，**不是证明运行时实际采用了这些值**。
JAX/NEK 的 horizon 字段还有独立语义问题。
? 是尚未固定，不是零。FD 不继承 MAIA 的 CFD 迭代数。
同一 K 在不同网格层级/求解器上不代表同样的物理时间。

| 精确 case ID | Metric profile | 学习曲线证据 | SI H/K | Hub H/K | 论文无控制 CD |
| --- | --- | --- | --- | --- | ---: |
| `Cavity_2D_Re7500` | cavity | — | ? / 400 | 200 / 150 | — |
| `Cavity_2D_Re7500_fine_FD` | fd_cavity | — | ? / ? | ? / ? | — |
| `Cavity_2D_Re7500_medium_FD` | fd_cavity | — | ? / ? | ? / ? | — |
| `Channel_3D_Retau180` | jax_channel | — | ? / ? | 10000 / 100 | — |
| `Cube_3D_Re3700` | cube | — | 200 / ? | 200 / 1000 | — |
| `Cylinder_2D_Re1000` | cylinder | ed3a | 200 / 350 | 200 / 225 | 1.533 |
| `Cylinder_2D_Re100_fine_FD` | fd_cylinder | — | ? / ? | ? / ? | — |
| `Cylinder_2D_Re100_medium_FD` | fd_cylinder | — | ? / ? | ? / ? | — |
| `Cylinder_2D_Re200` | cylinder | — | 200 / 200 | 200 / 87 | 1.341 |
| `Cylinder_2D_Re200_fine_FD` | fd_cylinder | — | ? / ? | ? / ? | — |
| `Cylinder_2D_Re200_medium_FD` | fd_cylinder | — | ? / ? | ? / ? | — |
| `Cylinder_2D_Re3900` | cylinder | ed2c | 200 / 600 | 200 / 500 | 1.68 |
| `Cylinder_3D_Re200` | cylinder | — | 200 / 400 | 200 / 575 | 1.318 |
| `Cylinder_3D_Re3900` | cylinder | — | 200 / 1200 | 200 / 1080 | 1.074 |
| `DRA2303Jet_3D_Re400000_Ma02` | dra_jet | — | 200 / 1000 | 200 / 1000 | — |
| `Kolmogorov_2D` | kolmogorov | fig3b（配置未对应） | ? / ? | 200 / 10 | — |
| `NACA0012Gust_2D_Re1000_AOA20` | naca_gust | — | ? / ? | 200 / 60 | — |
| `NACA0012Gust_2D_Re1000_AOA40` | naca_gust | — | ? / ? | 200 / 75 | — |
| `NACA0012Gust_2D_Re100_AOA20` | naca_gust | — | ? / ? | 200 / 75 | — |
| `NACA0012Gust_2D_Re100_AOA40` | naca_gust | — | ? / ? | 200 / 75 | — |
| `NACA0012Gust_3D_Re10000_AOA12` | naca_gust | — | ? / ? | 200 / 150 | — |
| `NACA0012Gust_3D_Re50000_AOA12` | naca_gust | — | ? / ? | 200 / 140 | — |
| `NACA0012_2D_Re1000_AOA20` | naca_steady | — | ? / ? | 200 / 170 | — |
| `NACA0012_2D_Re1000_AOA40` | naca_steady | — | ? / ? | 200 / 400 | — |
| `NACA0012_2D_Re100_AOA20` | naca_steady | — | ? / ? | 200 / 375 | — |
| `NACA0012_2D_Re100_AOA40` | naca_steady | — | ? / ? | 200 / 375 | — |
| `NACA0012_3D_Re10000_AOA12` | naca_steady | — | ? / ? | 200 / 350 | — |
| `NACA0012_3D_Re50000_AOA12` | naca_steady | — | ? / ? | 200 / 700 | — |
| `Nozzle2D_coarse` | nozzle | — | ? / ? | ? / 100 | — |
| `Nozzle2D_fine` | nozzle | — | ? / ? | ? / 200 | — |
| `Nozzle3D_coarse` | nozzle | — | ? / ? | ? / 200 | — |
| `Pinball_2D_Re100` | pinball | ed2a | 200 / 170 | 200 / 200 | 2.904 |
| `Pinball_2D_Re100_fine_FD` | fd_pinball | — | ? / ? | ? / ? | — |
| `Pinball_2D_Re150` | pinball | — | 200 / 100 | 200 / 150 | 2.922 |
| `Pinball_2D_Re150_fine_FD` | fd_pinball | — | ? / ? | ? / ? | — |
| `Pinball_2D_Re30` | pinball | — | 200 / 230 | 200 / 270 | 2.651 |
| `Pinball_2D_Re30_fine_FD` | fd_pinball | — | ? / ? | ? / ? | — |
| `Pinball_2D_Re75` | pinball | — | 200 / 190 | 200 / 200 | — |
| `Pinball_3D_Re100` | pinball | — | 200 / 225 | 200 / 195 | 2.795 |
| `Pinball_3D_Re150` | pinball | fig2a | 200 / 200 | 200 / 175 | 2.949 |
| `Pinball_3D_Re30` | pinball | — | 200 / 250 | 200 / 270 | 2.597 |
| `Pinball_3D_Re75` | pinball | — | 200 / 235 | 200 / 210 | — |
| `Sphere_3D_Re300` | sphere | — | 200 / 300 | 200 / 825 | 0.636 |
| `Sphere_3D_Re3700` | sphere | — | 200 / 750 | 200 / 2400 | 0.391 |
| `SquareCylinder_2D_Re1000` | square | — | 200 / 750 | 200 / 250 | 2.302 |
| `SquareCylinder_2D_Re200` | square | ed3c | 200 / 225 | 200 / 110 | 1.469 |
| `TCFmini_3D_Re180` | nek_channel | — | ? / ? | 2500 / ? | — |

完整逐行问题由 JSON 的 `issue_refs` 给出，叠加 `global_unresolved`；
表里有曲线也不表示该行已经通过验收。

## 6. 可从论文固定的训练方法部分

这是后续复现 recipe 的依据，不是完整配置；未知字段不以库默认值补齐。

| 方法 | 可以固定的论文信息 | 尚不能认定为原论文设置 |
| --- | --- | --- |
| PPO，SI 4A | GAE λ=.95；独立 actor/critic；全连接 ReLU；entropy regularization；学习率范围 5e-5–1e-3，batch 16–48 | 每工况具体学习率/宽度/层数、γ、entropy 权重、epoch/minibatch、normalization；clip .2 只写为 typical，不能当所有工况确定值 |
| DDPG，SI 4B | OU 探索、σ=.2；soft update τ=.005；replay 1e4–1e5；batch256；两隐层，每层256–512，batch normalization | 每工况宽度/学习率/预算；正文印作 μ_OU=.15，不能未经核实把它解释成某库的 OU 均值或 θ |
| TD3，SI 4C | 双 critic；target noise σ=.2、clip .5；policy delay2 | 每工况其他参数、探索噪声与 target smoothing 的区别 |
| GPPO，SI 4D | PPO 加可微 CFD 轨迹梯度；自适应 gradient/entropy 权重；Flax | 确切权重更新、反传 horizon、超参数、Fig.3b 配置与原代码未定位 |

旧作者 Pinball PPO YAML 另外给出：seed42、200 control steps、
num_episodes500、collector total_frames200000、frames_per_batch400、
256×256 Mish、GAE .97、clip .1、25 epochs、minibatch25、学习率3e-4。
这些数值本身可核实，但与最终论文描述有差异。
200000 transitions / 200 并不等于 500 episodes，不能靠字段名选择方便的预算。
代码 collector 才决定实际步数；需记录真实 controls/episodes/optimizer updates。
完整审计及旧例程问题见 [STRICT_REPRODUCTION.md](STRICT_REPRODUCTION.md)。

目前只有 ed3a/ed3c 的从头 PPO 对照线能明确作为单工况 PPO 的候选；
另三项多方法图尚不足以稳健选出“最优方法”。GPPO 是 Kolmogorov 中有直接证据
更强的候选，但实现和工况未确定。其余行不能仅凭通用算法名标注为“论文最佳”。

## 7. 未决事项与后续处理

| 优先级 | 未决事项 | 后续处理原则 |
| --- | --- | --- |
| A | 奖励公式与实现冲突：Pinball、Cavity、Kolmogorov、Channel、DRA 等 | 有明确论文公式的部分，以论文为复现目标；显式版本化 override，不能修改原始 YAML 后冒充来源一致 |
| A | 控制时域、传感器与驱动器映射 | 校验实际 solver 物理时间、动作与观测顺序；查清 subcycling 后再采用 SI 计数，不能直接改一个整数 |
| A | 固定测试初始化与扰动 | 写出精确 checkpoint 文件、相位/重启索引、gust/目标轨迹、warm-up、动作 smoothing、时长及终止语义；同一确定性重启换 seed 不是多个测试样本 |
| A | 原论文测试统计量与曲线数据 | 未找到精确数值的保留 null；有原始数据后才能固定 averaging window、误差容限和自动验收目标 |
| B | 41 个没有对应曲线的变体 | 保留现有 benchmark；若要完整覆盖，须讨论是否接受“我们新测的 specialist reference”，不得称为复现该工况论文分数 |
| B | Kolmogorov 的 Fig.3b 配置与 GPPO | 继续按公开描述重建时逐项标注选择；不可把 Eq.34 或 GI-PPO 改名当作原版 GPPO |
| B | Seeds、checkpoint selection、evaluation action convention | 论文给出的 seed 数量已记录，具体取值/策略未定位；后续若自己选，明确记为 organizer protocol |
| B | Nozzle3D、FD、NEK 等完整 evaluator binding | 实现与逐工况无控制/简单动作/冻结策略回放验证，不以数据下载完整代替执行认证 |

建议下一轮先逐项解决 A 类的可核实实现问题，同时讨论两项真正需要取舍的内容：
**无论文策略分数工况的新 reference 是否接受；无法恢复的测试窗口/初始化是否采用
明确标注的 organizer protocol。** 当前不替用户作这两项决定，也不缩减 47 个工况。

形式上，“冻结 benchmark 数据”“固定交付接口”“确定论文证据”“认证完整执行协议”
是四件事；前两项已经完成，第三项本轮可查证部分已落盘，第四项仍不能提前宣称完成。

## 一致性验证

本轮新增 5 个纯数据测试，检查 47 行与冻结 release/hash 一致、来源配置没有被
论文数值覆盖、图估计不会变成验收分数、不同工况/solver 不继承目标、无控制
数值与 RL 分数分离。在离线 R9 容器内只读挂载当前源码，全部 65 个 task tests
通过，无跳过；未重建镜像、未执行真实 CFD 或启动新的训练实验。
