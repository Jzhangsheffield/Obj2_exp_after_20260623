# Take/Put 与 Middle：R3D-18、MViT-v2-S、K400 初始化与 SupLoss 实验分析

日期：2026-08-18；Middle 与历史实验附录更新：2026-08-19  
分析对象：`results/rgb_takeput_middle_proto_rel_loso_20260817`  
实验包：`codex_script/rgb_takeput_middle_proto_rel_loso_20260817`

> **2026-08-19 更新摘要。**新增的11类 Middle 实验再次确认 K400 初始化有效：R3D direct-full 的最佳 N BA 从77.98%提高到91.82%，MViT从65.37%提高到89.36%；SupLoss 后的 frozen-backbone probe 分别从62.23%提高到84.20%、从42.39%提高到80.41%。Middle 的最高结果为 MViT-K400 SupLoss + full FT（92.23%），与 R3D-K400 direct-full（91.82%）非常接近。新增历史附录表明，旧 R3D 对比学习表征不可分是可复现的，但最有证据支持的解释不是“R3D 天生不能学习时间/类别”，而是旧实验同时使用随机初始化、不同的自定义3D ResNet、16.7倍更高的预训练 LR、更强且可能破坏语义的增强、无效 queue entry 未屏蔽，以及更困难的类别集合；由于这些因素没有逐一消融，报告将其表述为按证据强弱排序的原因，而不是单一因果结论。

## 1. 执行摘要

本轮训练结果与后续无需重训练的时间扰动诊断给出了七个相当清楚的结论。

1. **K400 初始化对 MViT-v2-S 是决定性的，而对 R3D-18 是稳定但较温和的增益。**在 direct full 训练中，K400 相对 random 使 R3D-18 的最佳 N balanced accuracy（BA）提高 5.61 个百分点，却使 MViT-v2-S 提高 37.29 个百分点。MViT-random 无论 direct 还是 SupLoss 路线都停留在接近随机水平，且训练集本身也没有拟合成功，因此这不是单纯的跨被试泛化问题，而是优化/表示塌缩。

2. **SupLoss 确实学到了可用的 take/put frozen representation。**K400 原始 backbone 只训练 head 时，R3D/MViT 的 BA 分别为 59.84%/59.11%；K400 经 SupLoss 后，head-only BA 提高到 89.39%/87.84%。因此 SupLoss 并非“没有学到类别信息”。

3. **SupLoss 对 full fine-tuning 的贡献依 backbone 而异。**R3D-random 和 R3D-K400 的 SupLoss-full 分别比对应 direct-full 低 1.19 和 0.41 个百分点，基本没有正增益；MViT-K400 则从 87.84% 提高到 89.97%，增益 2.13 个百分点。换言之，SupLoss 对 R3D 的主要价值是让冻结特征提前可用，而不是抬高最终可达到的二分类上限。

4. **R3D-18 在 take/put 上能够形成非常清晰、但低秩的类别轴。**R3D-K400 SupLoss backbone 在 N 上的线性探针 BA 为 89.29%，但有效秩只有 1.89，前5个方向解释 98.57% 方差；MViT-K400 的 BA 为 87.24%，有效秩为 10.25，前5方向解释 73.09% 方差。R3D 的表示更像把二分类压到一条轴上，MViT 则保留更多类内结构。

5. **现有 checkpoint 的诊断明确证明：在 Take/Put 上，R3D-K400 并不是只使用顺序无关线索。**其 frozen-backbone probe 原序 BA 为 89.29%；global shuffle 后降至 54.29%（下降 35.00 pp），reverse 后降至 14.43%（下降 74.86 pp）。R3D-random 也分别下降 32.40 和 55.91 pp。此前“R3D 对帧序不敏感”的观察不能外推到本轮 Take/Put SupLoss 表征。

6. **两种成功模型主要依赖粗粒度方向和阶段顺序，而不是四帧块内部的精确次序。**R3D-K400/MViT-K400 的 within-block shuffle 仅使 backbone BA 下降 6.53/3.74 pp；打乱四个连续四帧块的顺序则下降 28.56/18.50 pp；完全逆序更使两者降到明显低于随机水平，说明时间方向被系统性反转，而非仅增加噪声。

7. **MViT-K400 的最终 backbone 几何变化更大，但分类层面的时间依赖并不强于 R3D-K400。**global shuffle 时二者 backbone cosine 分别为 0.772 和 0.397，reverse 时为 0.476 和 -0.022；然而 BA 下降分别为 35.00/32.12 pp 和 74.86/69.92 pp。MViT 用更大范围的特征重构表示时间变化，R3D 则在较压缩的判别轴上发生足以跨越分类边界的移动。

综合而言，本轮实验修正了两个过强表述：**R3D 在容易的二分类任务上不仅可以很好地区分 take/put，而且其当前 SupLoss 表征明确编码了时间方向；它的问题更可能是表征高度压缩、复杂多类动作的可分性不足，而不是完全没有时间信息。**此前多类任务中“打乱帧序不掉点”的现象仍需在相同诊断协议下复核，不能由本轮二分类结果直接解释。

## 2. 数据与实验协议

### 2.1 数据划分

开发协议为 `dev_N`：

| Split | 被试 | 样本数 | take | put |
|---|---|---:|---:|---:|
| Train | M、MR、J | 2,205 | 1,247 | 958 |
| Development held-out | N | 715 | 395 | 320 |

Manifest 审计报告 `overlap_original_key=0`，没有相同原始样本跨 train/N。但 N 在每个分类 epoch 都被评估并用于选择最佳 epoch，因此 N 是开发被试，不是无偏最终测试集。

### 2.2 模型与训练路线

- Backbone：torchvision R3D-18、MViT-v2-S。
- 初始化：random 或 Kinetics-400（K400）。
- Direct：直接监督训练；full 为100 epoch，K400 另有 head-only 对照。
- SupLoss：200 epoch supervised contrastive pretraining；随后 head-only 或 full fine-tuning 50 epoch。
- 统一 classifier 优化器：AdamW，backbone LR `6e-5`，head LR `2e-3`，weight decay `1e-4`。
- 单次运行：seed 1。

### 2.3 UMAP 与特征指标

本报告重新提取了12个 checkpoint 的确定性特征：

- 4个 epoch-200 SupLoss checkpoint；
- 4个 direct-full 最佳 N-BA checkpoint；
- 4个 SupLoss-full 最佳 N-BA checkpoint。

UMAP 使用 cosine metric，**只在 M/MR/J 的训练特征上拟合**，然后将 N 特征 transform 到同一空间，不把 N 与训练集联合拟合。图中圆点为训练被试，叉号为 N；蓝色为 take，橙色为 put。

定量指标包括：

- 训练/N cosine silhouette；
- 在训练特征上拟合、到 N 上测试的 balanced logistic linear probe；
- 1-NN BA；
- effective rank、top-5 explained variance；
- N 上 take/put recall 与混淆矩阵。

UMAP 仅用于解释几何，模型优劣以 held-out probe、BA/F1 和混淆矩阵为主。

## 3. 输出完整性检查

当前下载内容包括：

- 4个 SupLoss epoch-200 checkpoint；
- 14组 downstream `summary.json`；
- 每组 downstream 的 best-accuracy、best-BA、best-F1、last checkpoint；
- 4组200 epoch SupLoss debug 日志和逐 epoch prototype/null-path 诊断；
- 无 NaN/Inf 记录。

本地结果目录共发现60个 `.pth`。实验包目录最初只包含代码、配置和 manifest，实际权重/日志位于项目 `results/rgb_takeput_middle_proto_rel_loso_20260817` 下；本报告及新增 UMAP 资产按要求写入实验包目录。

当前没有下载 epoch 50/100/150 的 pretrain checkpoint，只有相应训练日志和 epoch-200 checkpoint，因此不能直接画出不同 pretrain epoch 的 held-out feature geometry 演化。2026-08-18 已基于4个 epoch-200 checkpoint 补充完成 reverse/shuffle/repeat 的 checkpoint-level temporal diagnostics；原序特征与上一轮缓存特征的平均余弦均为 1.000000，确认权重、样本顺序和预处理一致。

## 4. Downstream 分类结果

### 4.1 完整结果

| 路线 | Backbone | 初始化 | 策略 | Final train BA | Best N BA | Best N F1 | Best epoch | Final N BA | Best→final drop |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Direct | R3D-18 | Random | Full | 100.00% | 86.78% | 86.93% | 48 | 84.93% | 1.85 pp |
| Direct | R3D-18 | K400 | Head | 65.97% | 59.84% | 57.80% | 21 | 55.41% | 4.43 pp |
| Direct | R3D-18 | K400 | Full | 100.00% | **92.39%** | **92.37%** | 52 | 91.49% | 0.90 pp |
| Direct | MViT-v2-S | Random | Full | 50.00% | 50.55% | 46.78% | 26 | 50.00% | 0.55 pp |
| Direct | MViT-v2-S | K400 | Head | 71.17% | 59.11% | 58.90% | 60 | 58.27% | 0.84 pp |
| Direct | MViT-v2-S | K400 | Full | 100.00% | 87.84% | 87.84% | 78 | 87.12% | 0.72 pp |
| SupLoss | R3D-18 | Random | Head | 99.95% | 80.27% | 80.72% | 9 | 76.43% | 3.84 pp |
| SupLoss | R3D-18 | Random | Full | 100.00% | 85.59% | 85.59% | 11 | 84.70% | 0.89 pp |
| SupLoss | R3D-18 | K400 | Head | 99.96% | 89.39% | 89.91% | 5 | 89.08% | 0.31 pp |
| SupLoss | R3D-18 | K400 | Full | 100.00% | 91.98% | 91.95% | 4 | 90.72% | 1.26 pp |
| SupLoss | MViT-v2-S | Random | Head | 52.90% | 57.02% | 56.15% | 47 | 51.49% | 5.53 pp |
| SupLoss | MViT-v2-S | Random | Full | 54.05% | 53.03% | 51.81% | 47 | 51.60% | 1.43 pp |
| SupLoss | MViT-v2-S | K400 | Head | 100.00% | 87.84% | 88.24% | 23 | 87.19% | 0.65 pp |
| SupLoss | MViT-v2-S | K400 | Full | 99.96% | 89.97% | 89.96% | 8 | 86.96% | 3.01 pp |

![Best N balanced accuracy comparison](report_assets/best_val_ba_comparison.png)

### 4.2 K400 初始化效应

以 full 模型比较：

| 路线 | R3D K400 − Random | MViT K400 − Random |
|---|---:|---:|
| Direct full | +5.61 pp | **+37.29 pp** |
| SupLoss + full FT | +6.39 pp | **+36.94 pp** |

R3D-random 不仅能够拟合训练集，也能在 N 达到86%左右 BA；K400 提供额外但非决定性的改善。MViT-random 则连训练集都停在50–54% BA，说明它在当前数据规模、LR和训练配置下没有被成功优化。MViT 的巨大 K400 增益不能简单解释为“K400 提供更好的泛化起点”，它首先解决了模型能否训练的问题。

因此，**random R3D 与 random MViT 的结果不是公平的纯架构能力比较**。两个 backbone 使用相同 `6e-5` backbone LR，但随机初始化 Transformer 很可能需要不同 LR、warm-up、layer-wise decay 或更长/更稳定的优化。

### 4.3 SupLoss 相对 direct 的作用

比较 full 模型：

| Backbone / init | Direct full | SupLoss + full FT | SupLoss − Direct |
|---|---:|---:|---:|
| R3D random | 86.78% | 85.59% | −1.19 pp |
| R3D K400 | **92.39%** | 91.98% | −0.41 pp |
| MViT random | 50.55% | 53.03% | +2.48 pp，但仍接近随机 |
| MViT K400 | 87.84% | **89.97%** | +2.13 pp |

对 R3D，SupLoss 没有提高 full fine-tuning 上限；对 K400 MViT 有小幅正增益。这个结果与“SupLoss 没有学到东西”并不相同，因为 head-only 结果显示 frozen representation 改善非常大：

| Backbone | K400 direct head | K400 SupLoss head | 增益 |
|---|---:|---:|---:|
| R3D-18 | 59.84% | 89.39% | **+29.55 pp** |
| MViT-v2-S | 59.11% | 87.84% | **+28.73 pp** |

更准确的解释是：

- 原始 K400 feature 对本项目 take/put 只有弱线性可分性；
- SupLoss 将两类重新组织为强线性可分结构；
- full supervised training 自身也能完成这种适配，尤其是 R3D；
- 因而 R3D 的 SupLoss 主要减少了 downstream 需要学习的工作，而没有提高最终二分类上限；
- MViT-K400 则从 SupLoss 得到约2个百分点的额外泛化增益。

### 4.4 收敛速度与最佳 epoch

![Held-out N learning curves](report_assets/classifier_val_ba_curves.png)

SupLoss 后的模型非常快地达到最佳 N BA：R3D-K400 为第4 epoch，MViT-K400 为第8 epoch，R3D-random 为第11 epoch。Direct full 则分别需要约48–78 epoch。

这说明 SupLoss 的主要实际价值之一是提供了接近最终解的初始化。但这也意味着固定50 epoch fine-tuning 并非最优：MViT-K400 从89.97%回落到86.96%，下降3.01个百分点。未来正式确认应预注册 checkpoint 规则，例如：

- 仅用内部 held-out subject 选择 best BA；或
- 固定较短的10/15/25 epoch；或
- 在多折内确定统一 epoch，再锁定到外层测试。

不能继续用 N 逐 epoch 选择后又把 N 称为最终测试。

## 5. SupLoss 训练动力学

![SupLoss training dynamics](report_assets/pretrain_dynamics.png)

### 5.1 MViT-random 出现明确的训练/表示塌缩

MViT-random 的特征和优化轨迹与其余三组完全不同：

- SupLoss 在约50 epoch 后停在7.0485，之后几乎不再下降；
- projection 每维平均标准差从 epoch 1 的0.054降至 epoch 50 的0.0093，并进一步降到约0.005；
- 平均梯度范数从 epoch 1 的8.32降到 epoch 50 的0.0247、epoch 200 的0.0059；
- downstream direct/sup 模型的 final train BA 只有50–54%；
- UMAP 的多个局部簇内 take/put 完全混合，训练/N silhouette 都接近0。

这是一条相当完整的塌缩证据链。MViT-random 的失败不应被解释为“MViT 不适合 take/put”，而应解释为“当前 random-MViT 优化配置失败”。

### 5.2 R3D-random 曾接近低变化状态，但随后恢复

R3D-random 在早期也出现 projection dispersion 很低的阶段：epoch 10 每维标准差约0.0045，loss约7.0486；但随后梯度重新增强，epoch 50/100/200 的 dispersion 逐步恢复到0.0225/0.0290/0.0322，loss降到6.6897/6.4698/6.3991。

这说明 R3D-random 的优化动力学比 MViT-random 更具恢复能力。K400 则使 R3D 和 MViT 都快速进入有效优化区域，在约50 epoch 后 loss 已接近最终值。

### 5.3 200 epoch 对 K400 模型可能偏长

R3D-K400 和 MViT-K400 的 loss 在约50–100 epoch 后已基本平台化；其 downstream 最佳 epoch又非常早。因此后续时间实验如果主要做因果筛选，可优先保存并比较 pretrain epoch 25/50/100/200 的 frozen probe，而不应默认 epoch 200 一定最好。

## 6. UMAP 与 frozen feature geometry

### 6.1 SupLoss 后的 backbone 表征

![SupLoss backbone UMAP](report_assets/umap_pretrain_backbone_grid.png)

定量结果：

| Backbone | Init | Train silhouette | N silhouette | N linear BA | N 1-NN BA | N eff. rank | Top-5 variance | take recall | put recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3D-18 | Random | 0.905 | 0.444 | 80.67% | 78.87% | 3.05 | 96.24% | 92.91% | 68.44% |
| R3D-18 | K400 | 0.978 | **0.681** | **89.29%** | 88.82% | 1.89 | 98.57% | 96.71% | 81.88% |
| MViT-v2-S | Random | −0.001 | −0.001 | 56.37% | 51.82% | 4.23 | 94.50% | 38.99% | 73.75% |
| MViT-v2-S | K400 | 0.886 | 0.546 | 87.24% | **87.43%** | **10.25** | **73.09%** | 94.18% | 80.31% |

#### R3D-random

训练集两类已经明显分离，但 N 的 put 大量侵入 take 区域。线性探针中101/320个 put 被预测为 take，而 take 只有28/395个被预测为 put。这说明它不是“完全没有类别信息”，而是 N 上的类别边界不对称，put 的跨被试稳定性更弱。

#### R3D-K400

K400 主要扩大了跨被试 margin，并把 put recall 从68.44%提高到81.88%。N silhouette达到0.681，是四个预训练 backbone 中最高的。它的有效秩却更低，说明当前二分类可以通过一条非常强的类别方向完成。

#### MViT-random

UMAP 形成许多局部岛，但每个岛内蓝/橙高度混合。这些岛更可能编码 run、场景、物体位置或其他非类别因素。聚类丰富不等于类别可分；silhouette约0和线性/1-NN接近随机共同确认该表示不可用。

#### MViT-K400

两类总体清晰分离，N 中只有边界处出现交叉。它的有效秩10.25远高于R3D-K400的1.89，在保持相近二分类 BA 的同时保留了更丰富的类内变化。这种结构更有可能支持更复杂的15类动作或时间关系，但“更高有效秩”本身仍不能证明其中包含帧序。

### 6.2 Projection head 表征

![SupLoss projection UMAP](report_assets/umap_pretrain_projection_grid.png)

| Backbone | Init | N projection silhouette | N projection linear BA | N projection eff. rank |
|---|---|---:|---:|---:|
| R3D-18 | Random | 0.497 | 78.78% | 1.13 |
| R3D-18 | K400 | 0.723 | 89.49% | 1.07 |
| MViT-v2-S | Random | −0.000 | 53.45% | 4.61 |
| MViT-v2-S | K400 | 0.664 | 86.40% | 1.22 |

成功模型的128维 projection 几乎退化为一维，这是二分类 SupLoss 将同类压紧、异类推开的自然结果。Projection 对 R3D-K400 很有效，但没有稳定超过 backbone；MViT-K400 projection BA 反而比 backbone低0.84个百分点。因此 downstream 使用 backbone feature 是合理的，不能只看 projection UMAP 得出“backbone 已经学好”的结论。

### 6.3 Direct 与 SupLoss-full 的 downstream feature

![Downstream backbone UMAP](report_assets/umap_downstream_backbone_grid.png)

| Backbone/init | Direct N probe BA | SupLoss-full N probe BA | Direct N sil. | SupLoss-full N sil. | Direct rank | SupLoss-full rank |
|---|---:|---:|---:|---:|---:|---:|
| R3D random | 87.55% | 84.70% | 0.581 | 0.537 | 1.85 | 1.99 |
| R3D K400 | 92.39% | 91.98% | 0.758 | 0.754 | 1.41 | 1.42 |
| MViT random | 50.00% | 54.25% | −0.006 | −0.002 | 2.63 | 1.27 |
| MViT K400 | 86.77% | 89.97% | 0.508 | 0.644 | 9.91 | 4.43 |

UMAP 与分类结果一致：

- R3D direct 与 SupLoss-full 的最终几何几乎相同，direct 略优；
- MViT-random 两种训练路线都没有形成类别方向；
- MViT-K400 在 SupLoss + full FT 后 silhouette 和 probe BA 同时提高；
- MViT-K400 的 rank 从9.91压缩到4.43，但类别可分性提高，说明 SupLoss/微调将部分非类别变化压缩为更聚焦的类别表示。

## 7. 这些结果对“R3D 是否缺少时间特征”的启发

### 7.1 新结果否定了一个过强解释

不能再笼统地说“R3D-18 对比学习后不能区分不同类别”。在 take/put 上：

- R3D-random SupLoss backbone 的 N linear BA 已有80.67%；
- R3D-K400 达89.29%；
- K400 head-only 达89.39%；
- full 达91.98–92.39%。

因此 R3D 能够学习类别。此前15类中类别混合，更可能是随着类别数、跨被试变化和动作相似度提高，R3D 的低秩/静态捷径表示不再够用。

### 7.2 高 take/put BA 本身不代表利用了帧序，但配对扰动提供了直接证据

Take 与 put 通常包含相反的物体状态变化，但一个16帧 clip 的单帧外观、物体最终位置、手与物体的相对位置，也可能足以区分两类。R3D 在 N 上形成近一维类别轴，恰好符合“抓住一个强静态或状态特征”的可能性。

因此单凭原序 BA/UMAP 不能判断时间利用。第8.1节随后在同一 checkpoint、同一 clip、同一 frozen probe 下只改变帧序：R3D-K400 在 global shuffle/reverse 后分别下降35.00/74.86 pp，R3D-random 也下降32.40/55.91 pp。这把“时间扰动”从相关性推测变成了配对干预证据。

更新后的判断是：

> R3D 的问题并非完全没有时间特征；在 Take/Put 上它形成了高度压缩、但明确依赖时间方向和粗粒度阶段顺序的类别方向。该方向是否足以支持复杂多类动作，仍是独立问题。

### 7.3 MViT 的高秩表示不是单独证据，但扰动显示其最终特征重构更强

K400 MViT 在 take/put 上保留更高 effective rank。UMAP/秩本身不能将高秩分量具体归因为时间、被试、视角还是运行环境；但配对扰动进一步显示，MViT-K400 的最终 backbone 对 global shuffle/reverse 的 cosine 下降到0.397/-0.022，明显低于 R3D-K400 的0.772/0.476。与此同时两者 BA 下降相近，因此更准确的表述是：MViT 的整体表示空间对帧序发生更大重构，R3D 则在更低秩的判别方向上编码足够强的时间信息。

### 7.4 初始化与时间建模必须分开检验

MViT-random 的失败是优化塌缩，不能用它判断时间建模。后续时间实验应优先比较：

- R3D-K400 vs MViT-K400；
- 完全相同的帧集合；
- ordered vs reverse/shuffle；
- 同一个 frozen checkpoint 和同一个 linear probe。

Random 初始化可以作为附加因子，但不能作为主 backbone 对比的唯一基础。

## 8. 已完成的时间诊断与推荐下一步

### 8.1 已完成：现有权重的无需重训练时间扰动诊断

#### 8.1.1 诊断协议与每种 N clip 构造的含义

使用4个 epoch-200 SupLoss checkpoint：R3D-random、R3D-K400、MViT-random（失败负控制）和 MViT-K400（成功正控制）。每个原始视频仍先按原实验的确定性验证采样得到16帧，随后才在这16帧上施加下列操作：

1. **Original：**保持采样后的16帧为 `[f0,f1,...,f15]`，作为每个 clip 自身的参照。
2. **Reverse：**变为 `[f15,f14,...,f0]`。帧集合完全相同，只把时间方向反转；如果模型编码 Take/Put 的方向，反转可能把预测系统性推向相反类别。
3. **Sample-wise global shuffle：**对每个 N clip 单独生成一个非恒等的16帧随机排列，例如 `[f7,f1,f14,...]`。排列由 `seed + N manifest index` 确定，四个模型对同一 clip 使用完全相同的排列。它保留所有原始帧，但同时破坏全局与局部顺序。
4. **4-block shuffle：**先切成 `[f0–f3]、[f4–f7]、[f8–f11]、[f12–f15]` 四个连续块，再只打乱四个块的位置，每个块内部仍按原序。例如可变成 `[f8–f11,f0–f3,f12–f15,f4–f7]`。它检验模型是否依赖粗粒度动作阶段的先后关系。
5. **Within-block local shuffle：**四个块仍留在原位置，但分别随机打乱每个四帧块内部的次序。例如第一块可能由 `[f0,f1,f2,f3]` 变为 `[f2,f0,f3,f1]`。它破坏短时局部运动，保留粗粒度阶段顺序。
6. **Repeat-center：**取时间索引 `T//2=8`，即采样序列的第9帧 `f8`，复制16次得到 `[f8,f8,...,f8]`。它完全移除运动和帧间变化，但保留一个真实的中心时刻静态画面。
7. **Temporal-mean frame repeated 16 times：**对同一 clip 的16帧逐像素求时间均值 `m=(f0+...+f15)/16`，再构造 `[m,m,...,m]`。实际在标准化张量上求均值；由于标准化是逐通道仿射变换，这等价于先对原始像素求均值再用同一 mean/std 标准化。它保留整段的平均外观，移除时间方向、运动和帧间变化；画面可能因跨时刻叠加而出现运动模糊。

前五种中的 original/reverse/三种 shuffle 使用相同的16帧集合，因而更适合单独归因于顺序。Repeat-center 和 temporal-mean-repeat 会改变帧集合，只能回答“动态信息是否重要”，不能把下降全部解释为顺序效应。

对每个模型和 backbone/projection 分别执行以下协议：

- 只用 M/MR/J 的 **original** 特征训练一个 class-balanced logistic linear probe；
- probe 固定后，在 N 的 original 和六种扰动上测试，不在 N 上重拟合或调参；
- `BA/F1 drop = original metric - perturbed metric`，正值表示性能下降；
- prediction flip rate 比较同一 N clip 在 original 与扰动下的预测是否改变；
- cosine 始终是同一 clip 的 original feature 与 perturbed feature 之间的余弦；
- R3D 在 layer1–4 输出后做全局平均池化；MViT 取 block 0/8/15 的 token 平均。中间层数值适合观察同一模型内敏感性如何向深层累积，不应直接用绝对值比较 CNN 与 Transformer。

#### 8.1.2 原序基线与主要 K400 结果

原序 frozen probe 基线为：R3D-random backbone/projection 80.67%/78.78%，R3D-K400 89.29%/89.49%，MViT-random 56.37%/53.45%，MViT-K400 87.24%/86.40%。这与第6节完全一致。

下表给出两个 K400 主对照的全部结果。`ΔBA`、`ΔF1`、Take/Put `ΔRecall` 都以原序为参照；单位为百分点（pp）。

| 表征 | 模型 | 扰动 | Cos | BA | ΔBA | ΔF1 | Flip | Take ΔRecall | Put ΔRecall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Backbone | R3D-K400 | Reverse | 0.476 | 14.43% | 74.86 | 75.19 | 80.98% | 77.22 | 72.50 |
| Backbone | R3D-K400 | Global shuffle | 0.772 | 54.29% | 35.00 | 39.09 | 35.24% | 10.63 | 59.38 |
| Backbone | R3D-K400 | 4-block shuffle | 0.795 | 60.74% | 28.56 | 29.43 | 32.59% | 18.99 | 38.12 |
| Backbone | R3D-K400 | Within-block shuffle | 0.976 | 82.76% | 6.53 | 6.42 | 7.55% | 0.25 | 12.81 |
| Backbone | R3D-K400 | Repeat-center | 0.630 | 65.29% | 24.00 | 26.01 | 37.06% | 44.56 | 3.44 |
| Backbone | R3D-K400 | Temporal-mean repeat | 0.610 | 59.67% | 29.62 | 30.91 | 41.54% | 44.56 | 14.69 |
| Backbone | MViT-K400 | Reverse | -0.022 | 17.32% | 69.92 | 70.19 | 77.48% | 72.66 | 67.19 |
| Backbone | MViT-K400 | Global shuffle | 0.397 | 55.12% | 32.12 | 32.87 | 38.32% | 24.56 | 39.69 |
| Backbone | MViT-K400 | 4-block shuffle | 0.643 | 68.74% | 18.50 | 18.78 | 23.78% | 15.44 | 21.56 |
| Backbone | MViT-K400 | Within-block shuffle | 0.806 | 83.51% | 3.74 | 3.76 | 8.95% | 2.78 | 4.69 |
| Backbone | MViT-K400 | Repeat-center | 0.268 | 60.64% | 26.60 | 28.52 | 42.66% | 46.33 | 6.88 |
| Backbone | MViT-K400 | Temporal-mean repeat | 0.266 | 58.76% | 28.48 | 29.61 | 41.96% | 42.28 | 14.69 |
| Projection | R3D-K400 | Reverse | 0.663 | 14.98% | 74.51 | 74.93 | 78.88% | 76.20 | 72.81 |
| Projection | R3D-K400 | Global shuffle | 0.877 | 54.32% | 35.17 | 39.23 | 34.69% | 11.90 | 58.44 |
| Projection | R3D-K400 | 4-block shuffle | 0.883 | 60.46% | 29.02 | 30.09 | 31.89% | 18.99 | 39.06 |
| Projection | R3D-K400 | Within-block shuffle | 0.987 | 82.45% | 7.04 | 6.99 | 7.55% | 1.27 | 12.81 |
| Projection | R3D-K400 | Repeat-center | 0.902 | 64.83% | 24.66 | 25.15 | 32.59% | 25.57 | 23.75 |
| Projection | R3D-K400 | Temporal-mean repeat | 0.896 | 59.50% | 29.99 | 30.84 | 33.99% | 22.78 | 37.19 |
| Projection | MViT-K400 | Reverse | 0.704 | 17.96% | 68.44 | 68.80 | 75.94% | 70.63 | 66.25 |
| Projection | MViT-K400 | Global shuffle | 0.885 | 55.27% | 31.13 | 32.34 | 36.50% | 21.01 | 41.25 |
| Projection | MViT-K400 | 4-block shuffle | 0.930 | 68.19% | 18.21 | 18.60 | 23.64% | 13.92 | 22.50 |
| Projection | MViT-K400 | Within-block shuffle | 0.979 | 81.63% | 4.77 | 4.82 | 9.37% | 3.29 | 6.25 |
| Projection | MViT-K400 | Repeat-center | 0.888 | 58.66% | 27.75 | 28.59 | 40.70% | 39.24 | 16.25 |
| Projection | MViT-K400 | Temporal-mean repeat | 0.896 | 58.77% | 27.64 | 28.12 | 40.42% | 32.15 | 23.12 |

![原序与扰动特征余弦](report_assets/temporal_diagnostics/temporal_feature_cosine.png)

![Frozen linear BA下降](report_assets/temporal_diagnostics/temporal_ba_drop.png)

![预测翻转率](report_assets/temporal_diagnostics/temporal_prediction_flip.png)

#### 8.1.3 负控制、分类别效应与逐层敏感性

MViT-random 的 backbone cosine 在所有扰动下均为 `0.9997–1.0000`，BA 变化仅为 `-0.45` 到 `+1.26 pp`；projection cosine 四舍五入到六位小数仍接近1，BA 甚至出现微小随机改善。它不是“学到了顺序不变性”，而是原模型已经塌缩、没有形成可被扰动的有效表征。该负控制说明本诊断能够把“模型没学会”与“成功模型对时间扰动敏感”区分开。

R3D-random 则表现出真实时间敏感性：backbone 在 reverse/global/4-block/within-block 下分别下降 55.91/32.40/28.22/3.22 pp，projection 分别下降 52.48/30.69/27.77/3.91 pp。因此时间敏感性并非仅由 K400 初始化继承；random R3D 在 SupLoss 训练恢复后也学到了 Take/Put 的方向结构，而 K400 使该结构更强、更稳定。

分类别结果揭示了两种不同失败方式：

- **Reverse 对 Take 和 Put 都造成极大下降。**R3D-K400 backbone 的 Take/Put recall 分别下降 77.22/72.50 pp，MViT-K400 分别下降 72.66/67.19 pp；这产生低于随机的 BA，是“方向反转导致类别系统性互换”的强证据。
- **Global shuffle 对 Put 更不利。**R3D-K400 backbone 的 Take/Put 分别下降 10.63/59.38 pp；MViT-K400 为 24.56/39.69 pp。R3D 的全局打乱结果尤其倾向预测为 Take，说明 Put 的判别更依赖完整阶段顺序。
- **Repeat-center/temporal-mean 对 Take 更不利。**在 backbone 上，两个 K400 模型的 Take recall 下降约42–46 pp，而 Put 的 repeat-center 仅下降3–7 pp。选取单帧或平均帧会保留某些偏向 Put 的静态状态，说明这两种构造还包含静态帧选择偏置，不能单独视作纯顺序实验。

![Take与Put分开统计的Recall下降](report_assets/temporal_diagnostics/temporal_class_recall_drop.png)

逐层结果以 `1 - cosine` 表示，越大说明该层受到扰动的影响越强：

- R3D-random 的 reverse 敏感性从 layer1/2/3 的约 `0.000004/0.000018/0.000456` 到 layer4 的 `0.1178`；R3D-K400 从约 `0.000007/0.000064/0.000825` 跃升到 layer4 的 `0.5238`。顺序差异主要在最高语义层被放大，而不是来自低层逐帧外观变化。
- R3D-K400 layer4 对 global/4-block/within-block 的距离为 `0.2276/0.2046/0.0237`，再次显示粗粒度阶段顺序远比块内四帧次序重要。
- MViT-K400 的 block 0/8 仍接近不变，block 15 对 reverse/global/4-block/within-block 的距离升至 `0.0547/0.0311/0.0175/0.0121`；经过最终 norm 与全局池化后，backbone cosine 进一步变为 `-0.022/0.397/0.643/0.806`。因此 MViT 的时间差异也主要在后段形成。
- MViT-random 各层变化均极小，继续支持塌缩负控制解释。

![逐层时间敏感性](report_assets/temporal_diagnostics/temporal_layer_sensitivity.png)

#### 8.1.4 对原假设的裁决

预设判据是：若 R3D-K400 高 BA 但 shuffle 后不下降，则支持“使用顺序无关线索”；实际结果与此相反。Global shuffle 使其 backbone/projection BA 都下降约35 pp，reverse 下降约75 pp，且 R3D-random 得到同方向结果。**因此，在本轮 Take/Put、epoch-200 SupLoss、dev-N frozen-probe 条件下，可以否定“R3D 仅依赖顺序无关线索”这一解释。**

MViT-K400 同时出现显著 feature change、BA/F1下降和 prediction flip，成功构成正控制；但 R3D-K400 的分类下降不比它小。两者真正的差别是：MViT 最终 backbone 的角度变化更大，R3D 的特征整体角度变化较小，却沿低秩判别轴跨越分类边界。也就是说，本轮结果支持“表示几何不同”，不支持“R3D 没有时间特征、MViT 才有”。

这也说明 cosine 不能代替任务指标：R3D-random 在 global shuffle 后的 backbone cosine 仍高达0.934，但 BA 已下降32.40 pp；R3D-K400 projection 在相同扰动下 cosine 为0.877，BA 下降35.17 pp。对于近一维或低秩类别轴，看似不大的角度变化仍可能穿过 linear probe 的决策边界。因此后续时间诊断应始终同时报告 feature change、BA/F1与 prediction flip。

限制是本诊断仍为单 checkpoint、单 dev-N fold、每个 clip 每种 shuffle 一个固定随机排列。715个 clip 使用独立排列降低了偶然性，但正式论文结论仍应在多个 shuffle seeds、多个训练 seeds 和完整 LOSO 上复现；同时应把此协议应用到此前帧序打乱不掉点的多类 checkpoint，检查差异来自任务、训练阶段还是当时的扰动实现。

### 8.2 新训练实验优先使用 K400 主对照

最小因果网格：

| Backbone | 初始化 | SupLoss 输入顺序 |
|---|---|---|
| R3D-18 | K400 | chronological |
| R3D-18 | K400 | randomized order |
| MViT-v2-S | K400 | chronological |
| MViT-v2-S | K400 | randomized order |

随机顺序组应保持相同16帧和空间增强；两个正 view 在同一 iteration 使用相同随机排列，并在下一 epoch 重新采样，以移除真实时间顺序但不额外制造正样本错位。

### 8.3 修复 random MViT 后才能讨论架构公平性

若后续仍希望比较 random backbone，应单独为 MViT 做小规模优化筛选：

- backbone LR：`6e-5 / 1e-4 / 3e-4`；
- warm-up：10/20 epoch；
- layer-wise LR decay；
- gradient clipping；
- 检查 patch projection 与前几个 block 的 update/parameter norm；
- 以 epoch 10/25 的 projection dispersion 和 frozen probe 作为早停标准。

当前 random MViT 已在约50 epoch 形成低梯度平台，继续无条件训练到200 epoch没有意义。

### 8.4 改进 checkpoint 与统计协议

- pretrain 保存10/25/50/100/200，而不只下载epoch 200；
- 用 frozen probe、temporal sensitivity 和 effective rank共同选 pretrain epoch；
- 初筛至少3 seeds；
- 参数锁定后执行 M/J/MR/N LOSO，而不是继续只看dev-N；
- 报告 fold/seed 均值、标准差与配对差值；
- UMAP 每个模型使用固定 seed，并始终 train-fit/test-transform。

## 9. Take/Put 阶段性判断

本轮 take/put 实验最重要的贡献不是证明某个 backbone 绝对最好，而是分离了四类机制：

1. **优化机制：**random MViT 在当前配置下塌缩；K400 首先让它能够被训练。
2. **表示机制：**R3D 倾向于形成非常低秩、强类别轴；MViT-K400 保留更丰富的类内结构。
3. **时间机制：**R3D-K400 与 MViT-K400 都强烈依赖时间方向和粗粒度阶段顺序；R3D 并非只使用顺序无关线索。
4. **任务机制：**Take/Put 的时间方向非常强，reverse 会让两类系统性互换；这一结果不能自动外推到此前更复杂多类任务。

在当前单seed dev-N协议中，最高下游结果是 **R3D-K400 direct full：92.39% BA**。在匹配 frozen 分类能力后，R3D-K400 与 MViT-K400 的 global shuffle BA 分别下降35.00/32.12 pp，reverse 分别下降74.86/69.92 pp。MViT 的 feature cosine 变化更大，但行为下降并不更强。因此本轮数据不支持“R3D 时间建模失效、MViT 正常”这一二分解释。

下一步最优先的工作应转向两个方向：第一，把同一诊断协议应用到此前出现“shuffle 不掉点”的多类 checkpoint，确认现象是否可复现并排除扰动实现差异；第二，对本轮诊断增加多个 shuffle seeds、多个训练 seeds 和完整 LOSO。若多类 R3D 仍对 global/block shuffle 不敏感，而 Take/Put R3D 明显敏感，则问题应重新定义为“R3D 学到的时间方向是否过于任务特定、低秩，无法支持细粒度多类区分”，而不是“R3D 是否完全编码时间”。

## 10. Take/Put 可复核文件

- 分类汇总：`report_assets/takeput_classifier_summary.csv`
- 特征质量汇总：`report_assets/takeput_feature_summary.csv`
- UMAP/PCA、feature arrays 与单模型指标：`report_assets/umap/<run>/`
- 汇总图：`report_assets/*.png`
- 可视化/汇总生成脚本：`tools/build_takeput_report_assets.py`
- 时间扰动完整汇总：`report_assets/temporal_diagnostics/temporal_diagnostics_summary.csv`
- 每个 N clip 的 cosine、原序/扰动预测与 flip：`report_assets/temporal_diagnostics/temporal_diagnostics_per_clip.csv`
- 逐层敏感性：`report_assets/temporal_diagnostics/temporal_layer_sensitivity.csv`
- 四个模型的全部扰动特征数组：`report_assets/temporal_diagnostics/*_temporal_features.npz`
- 时间扰动图表：`report_assets/temporal_diagnostics/*.png`
- 可复现诊断脚本：`tools/temporal_perturbation_diagnostics.py`
- 原始分类结果：`results/rgb_takeput_middle_proto_rel_loso_20260817/classifier/take_put/dev_N/`
- 原始预训练日志：`results/rgb_takeput_middle_proto_rel_loso_20260817/pretrain/take_put/dev_N/`
- 训练诊断：`results/rgb_takeput_middle_proto_rel_loso_20260817/analysis/take_put/dev_N/`

所有相对路径均以项目根目录或本实验包目录为基准；报告中的图片路径以本报告所在目录为基准。

## 11. Middle backbone / initialization 实验（2026-08-19）

### 11.1 问题与数据

Middle 任务使用与 Take/Put 相同的 `dev_N` 主体划分和训练框架，但从二分类变为11类：`insert、cut、label、pull_out、wrap、move、measure、remove、open、tear、cap`。训练集为 M/MR/J 共1,073个 clip，N 为384个 clip；类别样本明显不均衡，训练集每类为48–235、N每类为15–88，因此本节继续以 balanced accuracy（BA）和 macro-F1 为主，11类随机 BA 为9.09%。

本轮输出完整：4个 epoch-200 SupLoss checkpoint、14个 downstream summary，以及对应 best/last 权重均存在。比较包含：

- Direct full：R3D/MViT × random/K400；
- 原始 K400 frozen backbone 的 direct head-only；
- SupLoss epoch-200 后的 full/head-only：R3D/MViT × random/K400；
- 对4个 pretrain checkpoint 重新提取 backbone 与 projection 特征，在 M/MR/J 上拟合 linear probe 和 UMAP，再将 N 直接 transform/evaluate。

由于当前仍是单 seed、单 dev-N，以下差值是本轮观察值，不给出显著性声明。

### 11.2 Downstream 完整结果

| 路线 | Backbone | 初始化 | 策略 | Best N BA | Best N F1 | Best epoch | Final N BA | Final train BA |
|---|---|---|---|---:|---:|---:|---:|---:|
| Direct | R3D-18 | Random | Full | 77.98% | 75.81% | 93 | 77.00% | 100.00% |
| Direct | R3D-18 | K400 | Head | 54.24% | 51.93% | 49 | 51.03% | 69.55% |
| Direct | R3D-18 | K400 | Full | **91.82%** | 90.34% | 70 | 90.56% | 100.00% |
| Direct | MViT-v2-S | Random | Full | 65.37% | 62.40% | 85 | 63.59% | 89.71% |
| Direct | MViT-v2-S | K400 | Head | 58.63% | 56.84% | 74 | 58.08% | 79.58% |
| Direct | MViT-v2-S | K400 | Full | 89.36% | 88.44% | 42 | 88.29% | 99.69% |
| SupLoss | R3D-18 | Random | Head | 64.52% | 62.01% | 20 | 63.33% | 100.00% |
| SupLoss | R3D-18 | Random | Full | 70.46% | 69.45% | 33 | 69.76% | 100.00% |
| SupLoss | R3D-18 | K400 | Head | 82.66% | 81.11% | 3 | 82.66% | 100.00% |
| SupLoss | R3D-18 | K400 | Full | 88.97% | 89.39% | 4 | 87.11% | 100.00% |
| SupLoss | MViT-v2-S | Random | Head | 45.23% | 41.84% | 22 | 41.59% | 57.42% |
| SupLoss | MViT-v2-S | Random | Full | 53.09% | 50.54% | 34 | 51.52% | 70.70% |
| SupLoss | MViT-v2-S | K400 | Head | 81.48% | 79.28% | 23 | 81.18% | 100.00% |
| SupLoss | MViT-v2-S | K400 | Full | **92.23%** | **91.41%** | 21 | 89.81% | 100.00% |

![Middle最佳N balanced accuracy](report_assets/middle_best_ba.png)

最重要的效应如下：

| 比较 | R3D-18 | MViT-v2-S |
|---|---:|---:|
| Direct：K400 − random | **+13.84 pp** | **+23.99 pp** |
| SupLoss + full FT：K400 − random | **+18.51 pp** | **+39.14 pp** |
| Random：SupLoss-full − direct-full | −7.53 pp | −12.28 pp |
| K400：SupLoss-full − direct-full | −2.85 pp | **+2.87 pp** |
| K400 head-only：SupLoss − 原始K400 | **+28.42 pp** | **+22.84 pp** |

这组结果支持四点判断：

1. **K400 对 Middle 的两个 backbone 都非常有用，而且对 MViT 的作用更大。**与 Take/Put 一样，不能把 random-MViT 的失败直接当作架构能力；但 Middle 的 MViT-random direct 仍达到65.37%，说明它并非像 Take/Put 那样完全停在随机水平，而是 SupLoss 路线在当前超参数下明显没有训练到好解。
2. **最高最终性能没有显示明显 backbone 胜者。**MViT-K400 SupLoss-full 为92.23%，R3D-K400 direct-full 为91.82%，只差0.41 pp；单 seed 下不应据此宣称 MViT 优于 R3D。
3. **SupLoss 的收益主要体现在 frozen representation。**两个 K400 backbone 的 head-only 均提高22–28 pp，表明预训练确实重组了类别空间；但 full FT 后，R3D 反而低于 direct 2.85 pp，MViT仅提高2.87 pp。
4. **random SupLoss 对两种 backbone 都有负迁移。**这不是 R3D 独有问题；当没有 K400 时，当前对比学习目标/优化协议比直接监督更难，且 epoch-200 不一定是最适合迁移的 checkpoint。

### 11.3 Epoch-200 frozen feature 与 UMAP

![Middle epoch-200 frozen probe](report_assets/middle_frozen_probe.png)

| Backbone | Init | 表征 | Train silhouette | N silhouette | N linear BA | N macro-F1 | N 1-NN BA | N effective rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R3D-18 | Random | Backbone | 0.780 | 0.138 | 62.23% | 59.41% | 62.22% | 7.79 |
| R3D-18 | Random | Projection | 0.962 | 0.139 | 59.79% | 57.80% | 59.77% | 8.85 |
| R3D-18 | K400 | Backbone | **0.945** | **0.522** | **84.20%** | **82.17%** | 84.42% | 13.50 |
| R3D-18 | K400 | Projection | 0.993 | **0.555** | **84.41%** | **82.85%** | 84.51% | 9.25 |
| MViT-v2-S | Random | Backbone | 0.209 | −0.027 | 42.39% | 38.36% | 39.09% | 4.40 |
| MViT-v2-S | Random | Projection | 0.225 | −0.086 | 41.69% | 37.30% | 37.18% | 3.34 |
| MViT-v2-S | K400 | Backbone | 0.847 | 0.360 | 80.41% | 77.05% | 80.82% | **28.46** |
| MViT-v2-S | K400 | Projection | 0.986 | 0.441 | 80.65% | 77.37% | **82.04%** | 9.09 |

![Middle epoch-200 UMAP](report_assets/middle_umap_grid.png)

图中圆点为 M/MR/J，叉号为 held-out N。UMAP 只在训练主体上拟合，因此叉号相对训练簇的位置可以用于观察跨被试漂移。主要现象是：

- R3D-random 在训练集已形成紧凑类别岛，但 N 上的 `insert、remove、tear、cap` 明显漂移，frozen-backbone recall 仅28.4%、40.0%、33.3%、38.9%；这更像过拟合训练主体的类别原型，而不是所有类别都没有被分开。
- R3D-K400 将 N silhouette 从0.138提高到0.522、linear BA 从62.23%提高到84.20%。困难仍集中在 `insert`（51.1% recall）、`cap`（50.0%）与 `tear`（72.2%），而 `cut、move、remove` 达到100%。
- MViT-random 的多个类别沿连续曲线混合，N silhouette 为负；其 SupLoss 表征虽然高于9.09%随机水平，但跨被试类别几何很弱。这与 downstream SupLoss-full 只有53.09%一致。
- MViT-K400 也形成明确类别岛，frozen BA 80.41%；其 effective rank 28.46，明显高于 R3D-K400 的13.50。MViT 保留更多变化维度，但没有自动转化为更高 frozen BA；R3D 的类别 margin 更清楚。
- Projection head 没有“修复”一个坏 backbone，也没有造成 K400 模型的信息断层：同一模型的 backbone/projection N BA 非常接近。因而本轮可分性结论不是只看错了表示层。

### 11.4 训练动力学

![Middle SupLoss训练动力学](report_assets/middle_pretrain_dynamics.png)

K400 R3D/MViT 的 loss 在约50–70 epoch 已降至约4.8并进入平台，projection dispersion 稳定在约0.064–0.067；它们的 downstream 最佳 epoch 又分别只有4和21。相比之下：

- R3D-random 的 loss 缓慢降至约4.9，表示离散度从极低值逐步恢复，和 Take/Put 的“先低变化、后恢复”轨迹一致；
- MViT-random 到epoch 200的 loss 仍约6.1，梯度活动持续升高，表示离散度虽恢复到约0.04但仍显著低于成功模型。它不是完全静止的数值塌缩，却显然没有收敛到好的跨被试类别几何。

这说明固定训练200 epoch并不能保证公平：K400 模型较早形成有效表示，random 模型则可能需要完全不同的 LR/warm-up/regularization，而不是单纯延长训练。后续应保存25/50/100/200并以 frozen N-independent validation probe 选择预训练阶段。

### 11.5 Middle 对原 R3D 问题的直接回答

Middle 已经否定“只要类别多，R3D 对比学习就无法区分特征”的强命题。11类情况下：

- R3D-random epoch-200 backbone probe 为62.23%，远高于9.09%随机水平；
- R3D-K400 达到84.20%，且 UMAP 中多数类别形成清晰训练/N对应簇；
- R3D-K400 SupLoss head-only 达82.66%，证明类别信息在冻结 backbone 中已经可用；
- full FT 后 R3D-K400 达88.97%，而 direct-full 达91.82%。

因此，R3D 当前真正暴露的问题不是“不能形成多类特征”，而是：random 初始化下跨被试 margin 较弱；若 full FT 可用，SupLoss 没有超过 direct；若关注 frozen representation，K400 是主要成功条件。Middle 尚未做帧序扰动，故不能由本节证明11类 R3D 使用了时间顺序；它只证明了类别可分性。

## 12. 附录：为何旧 R3D 对比学习后看起来无法区分类别

### 12.1 复核范围与结论边界

本附录复核了用户列出的7组 `cl_rgb_*` 预训练实验、7组对应 `ft_rgb_*_seed1` 微调实验，以及 `analysis/N_as_test/umap_rgb_pretrain`、`umap_rgb_last` 和 `umap_rgb_last_3d`。`analysis` 元数据中的旧 checkpoint 路径有一部分使用不带 `cl_` 的历史别名；对 baseline `suploss_only/checkpoint_0200.pth` 计算 SHA-256 后，带/不带 `cl_` 的文件完全一致，因此该别名不影响本文的 baseline 数值。

结论分为两层：

- **可以确认：**旧 R3D epoch-200 对比学习输出在 Take/Put 上几乎没有 frozen 线性可分性；在15类 except-take/put 上虽高于随机，但整体很弱，而且 proto/rel 组合往往进一步降低 frozen probe。
- **不能仅凭现有实验确认：**究竟是某一个因素单独导致失败。旧/新实验同时改变了 backbone 实现、初始化、LR、增强、queue 处理和类别集合，因此原因只能按证据强弱排序，并需要最小消融验证。

### 12.2 旧 epoch-200 frozen probe：不可分现象确实存在

旧分析使用训练主体特征拟合 linear probe、在 N 上测试。关键结果如下：

| 旧任务 | Epoch-200方法 | Backbone linear BA | Projection linear BA | 随机BA |
|---|---|---:|---:|---:|
| Take/Put | SupLoss only | 49.81% | 50.00% | 50.00% |
| Take/Put | SupLoss + Proto P3 | 50.00% | 50.00% | 50.00% |
| Take/Put | SupLoss + Proto/Rel | 50.00% | 50.00% | 50.00% |
| Take/Put | SupLoss + Rel | 50.00% | 50.00% | 50.00% |
| Except Take/Put（15类） | SupLoss only | **32.93%** | 26.66% | 6.67% |
| Except Take/Put（15类） | SupLoss + Proto P3 | 23.50% | 20.52% | 6.67% |
| Except Take/Put（15类） | SupLoss + Proto/Rel | 19.35% | 14.99% | 6.67% |
| Except Take/Put（15类） | SupLoss + Rel | 17.23% | 14.90% | 6.67% |

所以旧观察不能简单归咎于“UMAP画得不好”：Take/Put 的 linear probe 与 k-NN 都约等于随机；15类中 SupLoss-only 尚有信息，但随着 proto/rel 约束加入，probe 反而下降。特别是 projection 通常比512维 backbone 更差，说明优化目标可能在投影空间形成了不利于 held-out subject 的结构。

同时也要注意，旧 UMAP 脚本对 train、test、combined 分别调用 `fit_transform`，不同图的坐标系并不相同；它适合看单图内部混合，却不适合比较训练簇与 N 漂移。新报告统一采用 train-fit/test-transform，并用 probe/silhouette作为主证据。因此“旧特征弱”成立，但不能只凭旧二维图判断弱到什么程度。

### 12.3 旧下游微调：预训练通常没有超过 scratch

| 旧实验族 | 最佳 scratch full BA | 最佳 pretrained full BA | 差值 |
|---|---:|---:|---:|
| Except + stage5 rel-topk | 90.81% | 79.99% | −10.83 pp |
| Take/Put 22组 | 95.32% | 94.55% | −0.77 pp |
| Except Take/Put 22组 | 91.01% | 86.07% | −4.93 pp |
| Except + depth10 | 86.75% | 79.40% | −7.35 pp |
| Except + random queue size | 87.31% | 84.59% | −2.72 pp |
| Except + rel-topk | 89.03% | 83.46% | −5.57 pp |
| Except + sampler | 89.90% | 85.96% | −3.95 pp |

七个实验族中，即使选择该族最好的 pretrained full run，也没有一个超过各自 scratch full；这说明旧对比学习问题不只是“二维空间不好看”，而是整体没有带来正迁移。但 pretrained full 仍可达到约80%–94.5%，说明 backbone 不是永久损坏，监督微调能够重组特征。最准确的表述是：**旧对比学习 checkpoint 没有提供一个良好的、可冻结迁移的类别空间，而且通常还是比 scratch 更差的初始化；这不等于 R3D 本身无法学习分类。**

### 12.4 新旧协议的关键差异

| 因素 | 旧 R3D 系列 | 本轮 Middle | 可能影响 |
|---|---|---|---|
| 初始化 | Random only | Random + K400 | 本轮最强实证因素；R3D frozen BA +21.97 pp，full BA +13.84至18.51 pp |
| Backbone 实现 | 项目自定义3D ResNet-18；stem时间核7，后续均为3×3×3，3D max-pool | torchvision `r3d_18` | 两者都叫R3D-18但结构/预训练兼容性不同，不能视作同一模型 |
| 预训练 LR | `1e-3` | `6e-5` | 旧值高16.7倍；random 训练可能不稳，K400若直接套用更易破坏先验 |
| LR schedule | 50/100/150 step，`cos=false` | warm-up + cosine，`cos=true` | 新协议更新更平滑 |
| Batch | 64 | 32 | 改变每批正样本数、BN和queue更新节奏 |
| Queue | K=1088，`exclude_invalid_queue=false` | K=1088，`true` | 旧训练早期将未填充queue位置纳入候选，可能污染对比损失 |
| Crop | scale 0.6–1.0，ratio 0.75–1.33 | scale 0.85–1.0，ratio 0.9–1.1 | 旧裁剪更容易移除手-物关系与操作上下文 |
| 翻转 | H=0.5，**V=0.5** | H=0.5，V=0 | 垂直翻转不符合真实操作场景，可能迫使模型忽略位置/方向线索 |
| 光度增强 | jitter/gray/blur均0.5，jitter较强 | jitter 0.2、gray 0、blur 0.1，jitter较弱 | 旧增强可能在小数据上过强，正对被迫对不自然视图保持一致 |
| 随机性 | `seed=null` | seed 1 | 旧复现与run间归因更困难 |
| 任务 | Take/Put 2类或Except 15类 | Middle 11类 | 类定义、样本量和视觉/时间线索均不同 |

还存在一个重要的混杂：旧15类测试标签中 `close` 在 N 上为0样本，旧 probe 的15类 BA实际会受缺失类别处理方式影响；报告引用原分析实现的数值，但不把旧32.93%与新11类62.23%做严格同尺度的绝对比较。

### 12.5 原因排序

结合新旧证据，最合理的排序是：

1. **K400 缺失是第一优先原因，证据最强。**同一新协议、同一 Middle 数据中，K400 使 R3D epoch-200 backbone probe 从62.23%升到84.20%，并使 SupLoss-full从70.46%升到88.97%。Take/Put 中也有同方向改善。旧实验完全没有这一关键控制。
2. **旧优化与增强组合是第二优先原因。**旧 LR 高16.7倍，并同时使用宽裁剪、0.5垂直翻转、0.5灰度和0.5模糊。对需要手-物位置、状态和方向的动作，这些不变性可能直接压掉判别线索。Middle 的 random R3D 在温和协议下已能达到62.23% frozen BA，说明“random必然不可分”也不成立。
3. **旧/新 R3D 不是同一个实现。**旧自定义3D ResNet与 torchvision R3D-18 的 stem、block和下采样设计不同；目前没有在同一协议下只替换实现的消融，因此它是高可信混杂，但尚不能量化贡献。
4. **无效 queue entry 与 sampler/queue 细节可能加重早期训练问题。**新协议显式屏蔽无效 queue，旧协议不屏蔽；旧的 queue size、balanced sampler、top-k、proto/rel 大量搜索并未稳定解决问题，说明它们更像次级因素，而不是主因。
5. **任务难度和跨被试漂移解释了一部分，但不是全部。**15类当然比2类难；然而旧 Take/Put frozen BA也只有50%，所以不能只怪类别多。反过来，新11类 R3D-K400 frozen BA达到84.20%，说明类别数量本身也不是充分原因。
6. **“R3D没有时间特征”目前不是首选解释。**本轮 Take/Put 时间扰动已证明 R3D-K400/global shuffle下降35.00 pp、reverse下降74.86 pp，R3D-random也显著下降。旧实验更可能没有把类别/时间结构优化进可迁移空间，而不是架构原则上无法编码时间。

### 12.6 仍不能排除的机制

新 Middle UMAP 主要证明类别可分，没有对11类 checkpoint 做帧序扰动。因此仍可能存在一种更窄的机制：R3D 能借助静态物体状态、手的位置和场景上下文把 Middle 分开，但对区分某些时间对称或细粒度类别仍较少使用帧序。要检验这一点，应把已经实现的 `original/reverse/global shuffle/4-block/within-block/repeat-center/temporal-mean repeat` 诊断原样应用到 Middle 的四个 epoch-200 checkpoint，并按11类分别报告 recall drop。只有当 R3D-K400 在高原序BA下对 global/block shuffle仍不下降，而 MViT-K400下降，才能把差异归因到时间顺序利用。

### 12.7 最小验证实验

为了从“协议混杂”走向因果解释，建议按优先级做以下最小实验，而不是继续大范围搜索 proto/rel/top-k：

1. 固定 Middle、torchvision R3D-18、mild augmentation、LR `6e-5`，只比较 random/K400；本轮已完成。
2. 固定 R3D-K400，依次单独切换 mild→old augmentation、`exclude_invalid_queue=true→false`、LR `6e-5→1e-3`。每项至少3 seeds，并保存epoch 25/50/100/200 frozen probe。
3. 固定数据与全部优化参数，只比较旧 custom ResNet3D-18 与 torchvision R3D-18；这是隔离“同名不同backbone”的必要实验。
4. 对当前 Middle 四个权重执行无需重训练的时间扰动；若 R3D/MViT 都下降，旧问题主要是训练协议；若只有 MViT下降，再设计 chronological-vs-shuffled pretraining 因果网格。
5. 最终锁定配置后做 M/J/MR/N 完整 LOSO；当前 N 被用于逐epoch选模，只能称为开发被试。

## 13. 更新后的总体结论

把 Take/Put、Middle 和旧实验放在一起后，证据支持以下统一解释：

- R3D-18 可以学习清晰的二类和11类表征，也可以编码时间方向；“R3D天生不具备时间/类别建模能力”不成立。
- K400 是当前最稳定的成功条件。它对 MViT尤其关键，对 R3D也从“可训练但跨被试边界较弱”提升到强 frozen representation。
- SupLoss 对 frozen/head-only 的价值非常明确，但不保证提高 full fine-tuning 上限。R3D 在 Take/Put和Middle均由 direct-full略胜SupLoss-full；MViT-K400则由SupLoss获得约2–3 pp。
- 旧 R3D checkpoint 的不可分现象是真实的，且预训练通常没有超过scratch；但旧协议改变因素过多。现有最强解释是 random初始化 + 激进增强 + 高LR + queue早期污染 + 不同R3D实现的组合，而不是单一“时间特征缺失”。
- 当前最值得做的不是继续扩展proto/rel超参数，而是对 Middle 做同口径时间扰动，并做三个单变量消融：旧增强、旧LR、旧custom R3D。

## 14. Middle 与历史附录可复核文件

- Middle downstream汇总：`report_assets/middle_classifier_summary.csv`
- Middle frozen feature汇总：`report_assets/middle_feature_summary.csv`
- Middle逐epoch训练诊断：`report_assets/middle_pretrain_debug_by_epoch.csv`
- Middle UMAP、PCA、原始特征与单模型指标：`report_assets/middle_umap/<run>/`
- Middle图表：`report_assets/middle_best_ba.png`、`middle_frozen_probe.png`、`middle_umap_grid.png`、`middle_pretrain_dynamics.png`
- 旧 epoch-200 probe汇总：`report_assets/historical_r3d_pretrain_probe_summary.csv`
- 旧7组微调实验族汇总：`report_assets/historical_r3d_finetune_family_summary.csv`
- 本节可复现生成脚本：`tools/build_middle_analysis_assets.py`
- 旧UMAP/probe原始分析：`analysis/N_as_test/umap_rgb_pretrain/`
- 旧fine-tuned UMAP：`analysis/N_as_test/umap_rgb_last/` 与 `umap_rgb_last_3d/`
