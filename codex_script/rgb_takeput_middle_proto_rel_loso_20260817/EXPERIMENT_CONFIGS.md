# 实验配置与判定规则

## 1. Manifest 与开发协议

| manifest | 类别数 | 类别 |
|---|---:|---|
| take_put | 2 | take, put |
| middle | 11 | insert, cut, label, pull_out, wrap, move, measure, remove, open, tear, cap |
| full | 17 | 全部 tier1 类别 |

每个任务都生成 `dev_N/test_M/test_J/test_MR` 四套 train/test manifest 和任务专属、连续编号的 `label_map.json`。生成器检查 `sample_name`、`original_key` 唯一性、四人覆盖、类别覆盖、train/test 人员互斥和 `original_key` 零交集。

开发阶段允许每个分类 epoch 在 N 上计算 accuracy、balanced accuracy、macro-F1、per-class 指标和混淆矩阵，以观察过拟合节点。选择主指标为 balanced accuracy，macro-F1 为共同主指标，accuracy 只作辅助。冻结配置时同时冻结预训练 epoch（默认 200）和微调 epoch 规则，不能再依据 M/J/MR 的结果改配置。

## 2. Take/Put 网格

直接分类均为 100 epochs：

| backbone | 初始化 | 微调策略 |
|---|---|---|
| R3D-18 | random | full |
| R3D-18 | K400 | full / head-only |
| MViT-v2-S | random | full |
| MViT-v2-S | K400 | full / head-only |

没有 random + head-only：随机冻结的 backbone 不具有可解释价值。

SupLoss 路线为 200 epochs 预训练 + 50 epochs 下游训练。R3D-18/MViT-v2-S × random/K400 共 4 个预训练 checkpoint；每个 checkpoint 复用到 full 与 head-only 两种微调，避免重复预训练。

## 3. Middle 增强网格

本地旧训练器能表达的共同操作为：时序一致 RandomResizedCrop、水平/垂直翻转、ColorJitter、灰度和 Gaussian blur。论文/仓库还可能包含本 loader 不能等价表达的时序采样、channel drop 或 random erase；这些没有被静默伪装成别的增强。

| ID | crop scale | ratio | HFlip | jitter p / B,C,S,H | gray p | blur p / sigma | 性质 |
|---|---|---|---:|---|---:|---|---|
| a0_mild | .85–1.0 | .9–1.1 | .5 | .2 / .1,.1,.1,.02 | 0 | .1 / .1–1.0 | 当前温和基线 |
| a1_slowfast | .2–.766 | .75–1.3333 | .5 | .8 / .6,.6,.6,.15 | .2 | .5 / .1–2.0 | SlowFast/MoCo-v2 风格候选，适配值 |
| a2_cvrl_adapted | .3–1.0 | .5–2.0 | .5 | .8 / .8,.8,.8,.2 | .2 | .5 / .1–2.0 | CVRL/SimCLR 强空间增强适配 |
| a3_tclr_adapted | .36–1.0 | 1.0 | .5 | .3 / .25,.25,.25,.1 | .175 | 0 | TCLR 风格适配，不模拟 channel drop/erase |
| a4_geometry_only | .3–1.0 | .5–2.0 | .5 | 0 | 0 | 0 | 只增强几何 |
| a5_photo_only | .85–1.0 | .9–1.1 | .5 | .8 / .4,.4,.4,.1 | .2 | .5 / .1–2.0 | 只强化光度 |
| a6_no_flip | .85–1.0 | .9–1.1 | 0 | .2 / .1,.1,.1,.02 | 0 | .1 / .1–1.0 | 检验左右语义是否被翻转破坏 |

参数来源核对：

- [SlowFast 官方默认配置](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/config/defaults.py)明确给出 SSL color jitter B/C/S=.4、H=.1、blur sigma .1–2.0、水平翻转和相对 crop 参数接口；具体 recipe YAML 会覆盖默认值。因此 a1 是面向当前 loader 的候选，不标为逐字复刻。
- [CVRL 论文与代码入口](https://github.com/tensorflow/models/tree/master/official/projects/video_ssl)强调对整段 clip 使用时序一致的强空间增强与 temporal sampling。当前包严格实现前者；后者仍由现有双视图时序采样器承担，不能把空间参数误称为完整 CVRL recipe。
- [TCLR 官方仓库](https://github.com/DAVEISHAN/TCLR)包含其数据管线；a3 只映射当前 loader 有对应语义的 crop/jitter/grayscale/flip。无法等价的 channel drop 与 erase 明示为未实现。
- [SimCLR 官方增强实现](https://github.com/google-research/simclr/blob/master/data_util.py)给出 strength=1 时 B/C/S=.8、H=.2，以及随机裁剪、水平翻转和随机颜色扰动，为 a2 的光度参数提供可核查基准。

先比较 a0–a6 的 SupLoss 表征和下游结果，再把 `selected.middle_augmentation` 更新为胜出配置。不得在同一损失网格中继续改变增强。

## 4. Middle ProtoLoss/RelLoss 网格

主网格固定旧版损失实现、P=1、preview EMA=.5。`pnull_p1`/`joint_null_p1` 是严格状态路径 null：仍创建/更新 prototype 状态，但两个附加 loss 权重为 0，用于排除“只是启用状态管理”带来的差异。

第一阶段：

- SupLoss-only 基线。
- Proto λ：0、.1、.5、1.0。
- Rel λ：.25、.5、1.0，初筛 K=5。

第二阶段：固定初筛胜出的 λ，比较 K=3、5、10。K=10 等价于 all，不再重复单独的 all 行。

第三阶段只在前面结果支持时运行：

- Rel 起点 epoch 50 vs 125。
- P=1 Proto+Rel 联合 active vs matched null。
- P=2 null/active sentinel。若 P2 prototype 的类内余弦相似度仍接近 1 且下游 BA/F1 没有稳定改善，即停止 P>1；不做 P3。

每个预训练配置都复用同一 epoch-200 checkpoint 做 full/head-only 两种 50-epoch 微调。预训练中不会每 50 epoch 自动启动下游训练。

## 5. 诊断与故障判据

每 10 iteration 记录：总 loss、SupLoss、加权 Proto/Rel 贡献、batch 类分布与正样本信息、q 特征范数/方差、prototype batch 信息、梯度 top-32、指定参数更新量、NaN/Inf。每个 epoch 保存 prototype 诊断，包括 active 数、类内 prototype 相似/距离、类间关系、漂移和配置元数据；每 10 epochs 重聚类。

以下任一情况都应先诊断，不继续扩网格：

- 附加 loss 加权贡献长期远大于 SupLoss；
- Proto/Rel 梯度不可见、为零或突然爆炸；
- 参数 update/parameter norm 长期接近零；
- 特征有效秩快速塌缩，类内距离未下降而类间距离下降；
- prototype 大量空槽、频繁跳变或 P2 类内 prototype 余弦长期接近 1；
- 出现 NaN/Inf；
- N 上最佳 BA 很早出现且末 epoch 明显下降。

正式报告应同时展示：逐 epoch BA/F1 曲线、最佳与末 epoch 差值、per-class 指标/混淆矩阵、train-fit UMAP/PCA、线性探针、1-NN、silhouette、Davies–Bouldin、类内/类间距离比、有效秩、prototype 漂移/相似度以及梯度/参数更新轨迹。

