# 全部实验配置与逐项说明

> 2026-08-14新增：Stage 8 四被试探索配置直接记录在本文件和 `config/experiment_plan.json`，不放入 `confirmation_runner`。详见 F 节。

> 2026-08-13后续实验更新：15类最小确认、17类take/put任务、增强策略、采样策略和无验证50轮最终协议统一记录在[`confirmation_runner/UNIFIED_EXPERIMENT_CONFIGS.md`](./confirmation_runner/UNIFIED_EXPERIMENT_CONFIGS.md)。本文件 A–E 描述原始Stage 1–7；Stage 8 是为复用原 MR 协议而新增的明确例外。

> 新增的跨对象/多seed锁定确认配置（包括修正后的H2严格Null和2×2消融）统一记录在 [`confirmation_runner/LOCKED_CONFIG_PARAMETER_REFERENCE.md`](./confirmation_runner/LOCKED_CONFIG_PARAMETER_REFERENCE.md)，机器配置位于`confirmation_runner/config/locked_config_registry.json`。

## A. 固定配置

### A1. 数据和任务

| 项目 | 固定值 | 说明 |
|---|---|---|
| 模态 | RGB | 单摄像头 `00143` |
| 类别层级 | tier1 | 15 类，沿用 except-take-put 标签体系 |
| 输入 | 16 帧，224×224 | 与已验证 MViT Stage 6 保持一致 |
| 人员 | M、J、MR、N | 支持单人留出或四人 LOSO |
| 环境字段 | `lighting` | `left / normal / right` 三种光照 |
| 筛选折 | MR | Stage 1–5 只用其 inner-val 做模型选择 |
| inner-val | 训练人员数据的 20% | 按人、类别、光照、run 分组，避免片段泄漏 |

### A2. Backbone

- `mvit_v2_s`，Kinetics-400 初始化。
- 对比训练与微调均允许 patch embedding/第一层更新；本包不默认冻结第一层，因为先前 Stage 6 已验证当前 MViT 载入和训练路径，而且本轮目标是损失函数的等条件比较。
- Direct FT、SupLoss、ProtoLoss、RelLoss 使用同一骨干、输入和微调策略。

### A3. 对比预训练公共参数

| 参数 | 值 |
|---|---:|
| epoch / batch size | 200 / 32 |
| optimizer | AdamW |
| 初始学习率 / weight decay | 6e-5 / 1e-4 |
| LR 策略 | 10 epoch warmup + cosine |
| projection dim / queue | 128 / 1088 |
| SupCon temperature | 0.07 |
| positives per query | 6 |
| recluster interval | 10 epoch |
| prototype temperature | 0.07 |
| prototype EMA | 0.99 |
| K-means | seed 42，n_init 10，max_iter 300 |
| 主权重保存 | 每 50 epoch |
| prototype 诊断 | 每 10 epoch，独立小文件 |
| Rel 边界权重 | 启动前后 10 epoch 范围额外记录 |

### A4. 数据增强

所有损失配置使用完全相同的增强，避免把增强收益误判为损失收益。

预训练：16 帧全局均匀采样；两个对比 view 共享时间索引，只改变空间/颜色，以避免将不同动作时刻误当作同一实例。空间增强为 RandomResizedCrop（scale 0.85–1.0、ratio 0.9–1.1）、水平翻转 0.5；颜色抖动概率 0.2，强度为亮度/对比度/饱和度 0.1、色相 0.02；高斯模糊概率 0.1，kernel 5、sigma 0.1–1.0。关闭垂直翻转、灰度化。

微调：全局均匀 16 帧；RandomResizedCrop 与水平翻转参数同上；关闭颜色抖动、模糊、灰度和垂直翻转。这样微调增强更保守，不人为破坏细粒度手部/工具颜色信息。

### A5. 微调公共参数

100 epoch，AdamW，batch size 32；backbone LR 6e-5，分类头 LR 2e-3，weight decay 1e-4，学习率里程碑为 50/75 epoch。执行**全参数微调**，不是只训练分类头。每 25 epoch 保存周期权重，并保留验证集最佳权重。

相对于此前 Stage 6C 的 MViT-v2-S（batch size 8），本包为 H100 将 batch size 提高到 32。由于 AdamW 下直接采用四倍线性学习率可能使旧版 ProtoLoss/RelLoss 在启动和重聚类边界出现过强梯度，这里采用较保守的平方根缩放：预训练 LR 从 3e-5 调到 6e-5，微调 backbone LR 从 3e-5 调到 6e-5，head LR 从 1e-3 调到 2e-3。queue 仍为 1088（可被 32 整除，共 34 个 batch），`num_positive=6`、warmup、epoch 和保存频率保持不变，以尽量不改变损失定义与负样本规模。

## B. 旧版损失语义

### B1. SupLoss

同类别样本作为正对，其他类别作为负对，是全部对比预训练配置的主损失。

### B2. ProtoLoss 三种 positive mode

- `single`：每个特征只使用其所属类别中距离最近的一个 prototype 作为硬正目标。优点是目标明确；缺点是边界样本会被硬分配，易造成早期错误自强化。
- `soft`：用特征到同类多个 prototype 的相似度形成 soft responsibility，再加权计算正目标。它允许一个样本部分属于多个子簇，最适合验证三种光照是否形成渐变子结构。
- `all`：同类的所有 prototype 都作为正目标。约束最强，但当同类 prototype 表示不同环境/动作细节时，可能把它们重新拉到一起，因而设置低权重和 collapse 对照。

`P1/P2/P3` 表示每个动作类别拥有 1/2/3 个 prototype。P2 与 P3 的主体实验严格成对；P1 只作为类中心基线及 EMG/IMU 配置迁移参照。

### B3. RelLoss

旧版 RelLoss 使用 prototype 之间的相似关系约束表示空间：

- `same_weight` 控制同类别 prototype 关系项；`diff_weight` 控制不同类别项。
- `diff-only` 即 `same_weight=0, diff_weight=1`，只推动最易混淆的异类 prototype 分开，避免同类子簇被过度约束。
- `Top-K3/Top-K10` 表示每个目标只处理最相近的 3/10 个异类关系；K3 更聚焦，K10 更全面但梯度更强。
- `rel_start=125` 是保守晚启动；`rel_start=50` 是较早启动；Stage 4 还测试 epoch 5 启动。
- `preview_ema_momentum` 是 RelLoss 用于稳定关系目标的 EMA；0.5 更快响应，0.3 更跟随当前模型。
- `constant/cosine` 表示 RelLoss 权重在启动后的调度方式。

Null-proto/Null-rel 保留相同代码路径与聚类/关系计算，但把对应 λ 设为 0，用于识别“额外路径、随机性或保存策略”造成的伪提升。

## C. 各阶段逐实验表

### Stage 1：强骨干基线（2）

| idx / ID | 内容 | 关键设置与目的 |
|---|---|---|
| 0 `d0_k400_direct` | K400 MViT 直接微调 | 无对比预训练；衡量 K400 初始化本身 |
| 1 `s0_sup` | SupLoss-only | 200 epoch SupLoss 后全参数微调；全部新增损失的主基线 |

### Stage 2A：ProtoLoss 主筛选（11）

| idx / ID | P | mode / λproto | 目的 |
|---|---:|---|---|
| 0 `pn2_soft_null` | 2 | soft / 0 | P2 Null-proto |
| 1 `pn3_soft_null` | 3 | soft / 0 | P3 Null-proto |
| 2 `ps2_l010` | 2 | soft / 0.10 | P2 柔性子簇 |
| 3 `ps3_l010` | 3 | soft / 0.10 | 三光照主候选 |
| 4 `ph2_l010` | 2 | single / 0.10 | P2 硬分配 |
| 5 `ph3_l010` | 3 | single / 0.10 | P3 硬分配 |
| 6 `pa2_l010` | 2 | all / 0.10 | P2 全正 prototype |
| 7 `pa3_l010` | 3 | all / 0.10 | P3 collapse 风险对照 |
| 8 `pt2_l100` | 2 | all / 1.00 | 传感器强权重迁移，P2 |
| 9 `pt3_l100` | 3 | all / 1.00 | 传感器强权重迁移，P3 |
| 10 `pt1_l100` | 1 | all / 1.00 | 少量 P1 类中心参照 |

全部在 epoch 50 启动 ProtoLoss。每一项都检查 assignment 数量、dead prototype、同类 prototype 余弦相似度、soft entropy、损失值和梯度/主损失相对强度；P2/P3 还比较 assignment 与三种光照的 NMI/ARI/purity。

### Stage 2B：ProtoLoss 权重加密（4，条件运行）

仅当 Stage 2A 的 soft 模式优于或接近 SupLoss，且 λ=0.10 附近仍无法判断趋势时运行。

| idx / ID | P | λproto |
|---|---:|---:|
| 0 `ps2_l005` | 2 | 0.05 |
| 1 `ps3_l005` | 3 | 0.05 |
| 2 `ps2_l020` | 2 | 0.20 |
| 3 `ps3_l020` | 3 | 0.20 |

### Stage 3A：RelLoss 主筛选（6）

| idx / ID | P | λrel / start / K | 目的 |
|---|---:|---|---|
| 0 `rn2_k3_s125` | 2 | 0 / 125 / 3 | P2 Null-rel |
| 1 `rn3_k3_s125` | 3 | 0 / 125 / 3 | P3 Null-rel |
| 2 `rl2_k3_s125` | 2 | 0.5 / 125 / 3 | R9-like，晚启动、聚焦异类 |
| 3 `rl3_k3_s125` | 3 | 0.5 / 125 / 3 | 上项的 P3 成对实验 |
| 4 `re2_k10_s50` | 2 | 0.5 / 50 / 10 | R7-like，早启动、更广异类关系 |
| 5 `re3_k10_s50` | 3 | 0.5 / 50 / 10 | 上项的 P3 成对实验 |

Stage 3A 全部为 diff-only、constant schedule、EMA 0.5。它回答“晚而聚焦”还是“早而广泛”更适合已经能形成类别结构的 MViT。

### Stage 3B：RelLoss 局部加密（8，条件运行）

| idx | ID | P | λrel | start | K | 比较点 |
|---:|---|---:|---:|---:|---:|---|
| 0–1 | `r025_2/3_k3_s125` | 2/3 | 0.25 | 125 | 3 | 更弱关系梯度 |
| 2–3 | `r100_2/3_k3_s125` | 2/3 | 1.00 | 125 | 3 | 更强关系梯度 |
| 4–5 | `rk10l_2/3_s125` | 2/3 | 0.50 | 125 | 10 | 固定晚启动，单独扩大 K |
| 6–7 | `rk3e_2/3_s50` | 2/3 | 0.50 | 50 | 3 | 固定 K3，单独提前启动 |

因此 Stage 3B 不是无结构网格：前四项确定 λrel；后四项把“启动时机”和“关系覆盖面”从 Stage 3A 的耦合比较中拆开。

### Stage 4：EMG/IMU 迁移及 RGB 改写（11）

| idx / ID | P | 损失 | 关键设置 |
|---|---:|---|---|
| 0 `h1_imu_rel_p1` | 1 | Rel | λrel=1，start50，K3，same+diff，cosine |
| 1 `h2_emg_both_p1_k10` | 1 | Proto+Rel | λp=λr=1，start50，K10，same+diff，cosine |
| 2 `hn1_null_p1` | 1 | Null | 与传感器路径匹配但两 λ=0 |
| 3 `sr2_sensor_rel` | 2 | Rel | H1 的 P2 RGB 改写 |
| 4 `sr3_sensor_rel` | 3 | Rel | H1 的 P3 RGB 改写 |
| 5 `se2_emg_both_k10` | 2 | Proto+Rel | H2 的 P2 RGB 改写 |
| 6 `se3_emg_both_k10` | 3 | Proto+Rel | H2 的 P3 RGB 改写 |
| 7 `si2_late_k3` | 2 | Proto+Rel | start50，K3，EMA0.3，cosine |
| 8 `si3_late_k3` | 3 | Proto+Rel | 上项的 P3 配对 |
| 9 `si2_early_k3` | 2 | Proto+Rel | warmup/start5，K3，EMA0.3 |
| 10 `si3_early_k3` | 3 | Proto+Rel | 上项的 P3 配对 |

Stage 4 的核心不是复制传感器最优值，而是用 P1 原样迁移作为参照，再用 P2/P3 成对实验判断 RGB 的三环境结构是否需要更多 prototype。

### Stage 5：入选项组合（6，动态生成）

先在 `selection.json` 中各选出一个 P2/P3 Proto 和 P2/P3 Rel。程序动态生成：

| 动态配置 | 内容 |
|---|---|
| C2 / C3 RGB-balanced | 入选 Proto + 入选 Rel，保持 RGB 筛选出的启动/权重 |
| C2 / C3 sensor-schedule | 入选 Proto 形式配合 Stage 4 的 same+diff/cosine 关系策略 |
| Null-C2 / Null-C3 | 与组合路径匹配，但 λproto=λrel=0 |

这 6 项用于检验两个损失是否互补、相互冲突，及组合收益是否只是额外训练路径导致。

### Stage 6：四人 LOSO 确认（6 配置 × 4 人 = 24）

对以下六个候选分别运行 M/J/MR/N 留一人测试：Direct FT、SupLoss、最佳 P2、最佳 P3、总体最佳、最佳 P1。若某个候选 ID 重复，仍保留其角色，但汇总时应注明重复，避免把相同配置误当独立证据。

Stage 6 才允许使用 outer-test；选参仍依据 Stage 1–5 的 inner-val，不能依据某个留出人的 test 结果回头调整配置。

### Stage 7：多 seed 最终确认（4 配置 × 4 人 = 16，可选）

固定 SupLoss 和 Stage 6 总体胜者，分别补 seed 2、3，并在四个人上运行。加上 Stage 6 的 seed 1 后，每种方法得到 3 seeds × 4 folds。此阶段只确认稳定性，不再调参。

## D. 入选规则

Stage 2/3 入选时按以下顺序：

1. inner-val macro-F1；并列时看 accuracy；
2. 相对 SupLoss 的增益，而不只是绝对最高 epoch；
3. Null 对照差值；
4. prototype 健康度：无持续 dead prototype、同类 prototype 不完全重合、assignment 不极端失衡；
5. P3 的环境相关性：若 NMI/purity 为零，说明三光照假设没有得到支持；若接近完全一一对应且动作分类变差，说明模型可能过拟合环境；
6. 优先选择损失曲线和梯度比例稳定、而非偶然尖峰的配置。

Stage 6 最佳设置必须同时报告四折 mean±std、每折结果和相对 SupLoss 的配对差值。只有 Stage 7 仍保持优势，才适合写成“ProtoLoss/RelLoss 改善 MViT”；否则应表述为“在单次筛选中有候选，但没有稳定超过基线”。

## E. 实验数量总览

| 方案 | 训练次数 |
|---|---:|
| 核心筛选（不跑 2B/3B） | 36 |
| 完整筛选（Stage 1–5 全部） | 48 |
| 核心筛选 + Stage 6 | 60 |
| 完整 Stage 1–6 | 72 |
| 完整 Stage 1–7 | 88 |
| Stage 8 四被试探索（单独计数） | 32 |

权威机器可读配置以 `config/experiment_plan.json` 为准；动态入选结果以 `config/selection.json` 为准。

## F. Stage 8 四被试探索配置（2026-08-14）

### F1. 研究问题与统计定位

本阶段用于回答两个问题：

1. P1 的 Proto-only、Rel-only、Proto+Rel 在 M/J/MR/N 上是否呈现一致方向；
2. P3 `rl3_k3_s125 - rn3_k3_s125` 在 MR 为正、N 为负的现象是否是普遍的被试异质性。

它不是新的超参数网格，也不是独立最终测试。四折 outer-test 都会被查看，因此任何基于 Stage 8 结果产生的选择都必须在后续冻结配置、多 seed 协议中重新确认。

### F2. 精确配置表

所有配置均为 seed 1，沿用 A 节的 200 epoch 预训练、100 epoch 全参数微调、每折 20% 分组 inner-val，以及 `best_val_balanced.pth` 外测规则。

| idx / ID | 来源/角色 | mode | P | λproto | λrel | proto/rel start | same/diff | Top-K | schedule / EMA |
|---:|---|---|---:|---:|---:|---|---|---:|---|
| 0 `x8_d0_direct` | `d0_k400_direct` | Direct FT | — | — | — | — | — | — | — |
| 1 `x8_s0_sup` | `s0_sup` | contrastive_only | 1 | 0 | 0 | — | — | — | — |
| 2 `x8_h00_p1_k10` | H2 strict null | contrastive_proto_rel | 1 | 0 | 0 | 50 / 50 | 1 / 1 | 10 | cosine / 0.5 |
| 3 `x8_h10_p1_k10` | H2 Proto-only | contrastive_proto_rel | 1 | 1 | 0 | 50 / 50 | 1 / 1 | 10 | cosine / 0.5 |
| 4 `x8_h01_p1_k10` | H2 Rel-only | contrastive_proto_rel | 1 | 0 | 1 | 50 / 50 | 1 / 1 | 10 | cosine / 0.5 |
| 5 `x8_h11_p1_k10` | H2 Proto+Rel | contrastive_proto_rel | 1 | 1 | 1 | 50 / 50 | 1 / 1 | 10 | cosine / 0.5 |
| 6 `x8_rn3_k3_s125` | `rn3_k3_s125` bridge null | contrastive_rel | 3 | 0 | 0 | — / 125 | 0 / 1 | 3 | constant / 0.5 |
| 7 `x8_rl3_k3_s125` | `rl3_k3_s125` bridge active | contrastive_rel | 3 | 0 | 0.5 | — / 125 | 0 / 1 | 3 | constant / 0.5 |

`x8_h00/h10/h01/h11` 除 λproto 和 λrel 外完全相同；这修正了旧 `hn1_null_p1` 使用 Top-K3、而 H2 active 使用 Top-K10 所造成的不严格对照。`x8_rn3/x8_rl3` 也只改变 λrel，专门用于四折配对比较。

### F3. 为什么不全部设为 P=1

P=1 适合做当前主线，因为它避免 P2/P3 同类 prototype 高度重合，并形成清楚的类中心约束；但 P=1 不再表示类内多 prototype。若把所有配置都设为 P=1，会删除触发本次实验的 P3 Rel 机制，无法解释 MR/N 的方向反转。

因此本阶段：

- 以 P1 严格 2×2 消融作为主体；
- 暂停 P2 网格；
- 只保留一组 P3 null/active 桥接对照。

这不是断言“P2 永远无效”，而是基于当前证据降低其优先级。只有当 P3 配对在多数折稳定同向时，才值得另开匹配 schedule、start、Top-K 的 P1/P2/P3 专项比较。

### F4. 运行规模与预设比较

规模为 8 配置 × 4 留出人 = 32 个 fold-config pipeline。预设主要比较为：

- `x8_h10_p1_k10 - x8_h00_p1_k10`：Proto 主效应；
- `x8_h01_p1_k10 - x8_h00_p1_k10`：Rel 主效应；
- `x8_h11_p1_k10 - x8_h00_p1_k10`：联合路径净效应；
- `x8_h11_p1_k10 - x8_s0_sup`：相对实际主基线的增益；
- `x8_rl3_k3_s125 - x8_rn3_k3_s125`：P3 Rel 净效应；
- `x8_rl3_k3_s125 - x8_s0_sup`：P3 active 相对主基线。

每项应报告四折逐折 BA、macro-F1、accuracy、四折 mean±sample std 和方向一致性。Stage 8 不使用 `config/selection.json`，权威配置就是本文件对应的 `config/experiment_plan.json` 中 `stages.stage8`。
