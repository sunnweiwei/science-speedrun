# 固定数据发布：HydroGym Research v2

2026-09-08，用户确认后冻结。默认版本：`hydrogym-public-research-v2`。

**39 项：20 个 2D、19 个 3D；558 个源文件；453,756,315,982 字节 = 453.756 GB。**
500 GB 上限内剩余 46.244 GB。训练和测试共享同一份源资源、完全相同的 39 项成员。

## 固定的内容与尚未完成的内容

固定：任务成员、每个源文件的大小/哈希、原始资源版本、控制模型交付接口、
成本证据和三个双卡硬件例外。一个联合训练的共享 checkpoint 必须覆盖全部 39 项，
评测期间不能更新参数或学习到的归一化统计；成本档位不是可跳过的评分项。

未完成：所有任务的正式物理评测配置绑定、完整 39 项独立评测验收、
新增任务真实训练效果、论文分数复现、共享模型 baseline 和正式 Agent 实验。
`status=data-release-not-cfd-certified` 和数据回执的
`benchmark_validated=false` 保留。文件校验不替代这些验收。

当前 19 项继承 hour-v1 的短 PPO/完整回合重放证据；其余 20 项只有短成本测量。
Channel 沿用显式 actuator-clock 修复，Kolmogorov 成本配置为官方
Re200/64²/float64 示例，细喷管沿用原 restart 加载绑定。不能把这些称为
未经修改、全面对齐论文的设置。详细配置随清单保留，原始源文件不改。

## 每项任务、数据大小和成本

见 [完整 39 项选择表及 23 项排除理由](SELECTION-RESEARCH-V2.md) 和
[机器可读选择](selection-research-v2.json)。
参考每项 4,000 次控制：30 项单 H100 规划不超过 50 小时，5 项约 51–67 小时，
1 项约 101 小时；3 项明确使用两张 H100，约 54、56、137 墙钟小时，单卡成本未知。
整轮约 1,118 GPU 小时。成本包含显式估算开销和 25% 规划余量，不是实训完成时间、
收敛或 runtime 上界，不限制 Agent 后续继续训练。

数据指网格、几何、初始流场及配置，不是预生成的 RL 轨迹。
不扣重复数据或硬链接折扣；生成轨迹、模型、镜像、日志、缓存和 scratch 不计入源数据上限。

## 下载或复用本地数据

从任务仓库根目录运行：

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2 \
  --from-local /large-disk/existing-hydrogym
```

第二条只打印计划，不下载。没有本地资源时，第三条改为 `--download`；
现有 `hf` 接口只下载已选目录，固定 Hub revision
`cf5ef6748443ae72e9647f398487e98d421a7243`，不需要下载整个历史 1.09 TB 集合。
本地复用建立硬链接并逐文件校验完整 Git/LFS 哈希。源数据必须只读挂载。

本机新目录：
`/mnt/workspace/hydrogym/releases/hydrogym-public-research-v2`。

本机已于 2026-09-08 完成 materialization：558 个源文件全部通过完整 Git/LFS
哈希校验，目录的源文件数和字节数与冻结清单一致。所有源文件以硬链接复用既有
归档，没有额外复制一份 454 GB 数据；旧版本未删除或修改。数据回执已生成，
其中 `benchmark_validated=false`，不把文件验证解释为完整物理评测验收。

设置 `TASK_DATA_DIR` 指向这个目录，由既有 Compose/Runner 挂载到 `/data`。
新回执只在所有文件通过校验后发布；运行中的旧实验不改挂载。

规范化数据清单 SHA256：
`cabc8b718cdf9b22fcd661b9b3c5f8e452498e1c299b0fc9c7ac1afc22df47d2`。

## 历史版本和下一步

19 项 hour-v1 的清单与说明保存在 `release-hour-v1.json` 和
[RELEASE-HOUR-V1.md](RELEASE-HOUR-V1.md)，旧本地根目录不变。
更早的 47/28/40 项清单、数据和实测记录同样保留，没有删除、改写或重新标记旧数据。

**后续本地清理（2026-09-08）：** 用户随后批准清理旧源资源。旧版本清单和
复现记录仍保留，但旧本地数据视图已退役，不再是完整可用的历史发布。
仅两项尚在运行的圆柱 baseline 保留各自所需的原数据路径，其源文件与新版本
为硬链接。Research v2 的 39 项成员、源文件和清单未改变。清理回执及原始历史
清单保存在本机 `/mnt/workspace/hydrogym/maintenance/research-v2-cleanup-20260908`。
固定验收 workspace 为 `/mnt/workspace/hydrogym/workspaces/research-v2`。

下一步先完成 [完整评测与 baseline 计划](../../docs/ROADMAP.md)：
39 项物理/推理验收 → 人工 reference 与明确论文对比 → 单共享模型 baseline
→ Agent Runner 小实验 → 正式持续研究。当前没有启动新训练、Agent 或外部 push。
