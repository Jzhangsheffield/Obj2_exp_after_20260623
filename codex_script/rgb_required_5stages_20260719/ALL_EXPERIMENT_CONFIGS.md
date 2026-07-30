# RGB 必做实验包：完整实验配置与设计说明

本文档是 `rgb_required_5stages_20260719` 实验包的总说明书。它不仅记录参数，还解释每个实验改变了什么、为什么要做、应该与谁比较，以及结果应如何解释。

> 配置数值的最终机器可执行来源是：
>
> - 主实验：`common/experiment_plan.json`
> - Stage 4B：`stage4b_small_confirmation/config/confirmation_plan.json`
> - Stage 7 最终候选：`stage7_finetune/config/selected_sources.json`
>
> 本文档用于理解和审计。若本文档与 JSON 不一致，以 JSON 为准，并应同步修正文档。

---

## 1. 整体研究问题与阶段关系

本实验包围绕三个核心问题展开：

1. SupLoss 预训练相对 scratch 是否有稳定收益，历史基线差异是否主要来自随机性或训练协议；
2. `ProtoLoss` 和 `RelLoss` 是否真的通过各自梯度改善表征，而不是 prototype refresh、EMA 或随机波动造成的假象；
3. 当前 ResNet3D-18 是否因时间维压缩过强而损失动作顺序信息。

推荐的逻辑顺序为：

```text
Stage 0：建立可靠 scratch / SupLoss 基线
    ├── Stage T3：检查时间维过度压缩
    ├── Stage 1：筛选 ProtoLoss 形式
    └── Stage 4：筛选 RelLoss 机制
             ↓
Stage 4B：用 3 seeds + Null-loss 做小规模确认
             ↓
Stage 5：只组合已经确认有效的 ProtoLoss 与 RelLoss
             ↓
Stage 7：统一比较微调协议，冻结模型后才运行测试集
```

Stage 编号沿用已有实验包，因此没有 Stage 2、Stage 6；T3 在目录中对应 `stage3_temporal_stride`。

---

## 2. 全阶段固定数据条件

| 类别 | 固定配置 | 含义 |
|---|---|---|
| 项目根目录 | `/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623` | 集群上的代码与结果根目录 |
| 数据集根目录 | `/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset` | 所有阶段必须使用同一数据版本 |
| train | `N_as_test/train_manifest_except_take_put.jsonl` | 训练集；排除 take/put |
| validation | `N_as_test/val_manifest_except_take_put.jsonl` | 模型选择和实验决策使用 |
| test | `N_as_test/test_manifest_except_take_put.jsonl` | 仅在最终配置冻结后使用 |
| label map | `label_map_except_take_put.json` | `tier1` 的 15 类映射 |
| 模态 | RGB | 本实验包不混入 EMG/IMU |
| 摄像头 | `00143` | 所有可比实验保持相同视角 |
| 输入 | 16 帧，RGB，`224×224` | 不在单个实验中私自改变 |
| RGB mean | `[0.3752, 0.3864, 0.3960]` | 训练、微调和测试一致 |
| RGB std | `[0.2934, 0.2724, 0.2644]` | 训练、微调和测试一致 |
| 主干网络 | 3D ResNet-18 | T3 只修改时间 stride，不改变参数形状 |
| 主选择指标 | validation Balanced Accuracy | 对类别不平衡比普通 accuracy 更合适 |
| 测试权重 | `best_val_balanced.pth` | 禁止根据测试结果选择 epoch |

### 2.1 为什么必须固定这些条件

摄像头 ID 相同不等于实验协议相同。数据清单、类别映射、随机种子、增强、预训练 checkpoint、微调方式、学习率和最终权重选择方式中任一项不同，都可能形成明显差距。因此跨阶段比较时，必须优先确认 provenance、配置 JSON 和实际命令，而不能只看目录名。

---

## 3. 关键术语、损失和 epoch 约定

### 3.1 总损失

训练器中的目标可概括为：

```text
Ltotal = Lsup + λproto × Lproto + λrel × Lrel
```

- `Lsup`：监督式对比损失 SupLoss，是所有预训练实验的主损失；
- `Lproto`：样本到 prototype 的对比损失；
- `Lrel`：约束 prototype 更新前后相对几何关系的 directional loss；
- `λproto`、`λrel`：只缩放相应辅助损失对总梯度的贡献，不是学习率。

### 3.2 四种 ablation mode

| `ablation_mode` | prototype state/refresh | ProtoLoss | RelLoss | 用途 |
|---|---:|---:|---:|---|
| `contrastive_only` | 关闭 | 关闭 | 关闭 | 纯 SupLoss 基线 |
| `contrastive_proto` | 开启 | 允许开启 | 关闭 | 单独研究 ProtoLoss；`λproto=0` 时可构造 Null-proto |
| `contrastive_rel` | 开启 | 关闭 | 允许开启 | 单独研究 RelLoss；`λrel=0` 时可构造 Null-rel |
| `contrastive_proto_rel` | 开启 | 开启 | 开启 | Stage 5 组合实验 |

`contrastive_proto/rel + λ=0` 不等价于 `contrastive_only`：前者仍会建立 prototype、周期性聚类、执行 EMA，并经过相同计算路径，只是该损失对总梯度的加权贡献为零。这正是 Null-loss 对照的意义。

### 3.3 Prototype 的建立与更新

- 每个类别独立进行 KMeans，得到该类的 1、2 或 3 个 prototype；
- `sample_to_proto` 记录训练样本在本类别内部的硬 assignment；
- prototype refresh 间隔为 10 epochs；
- refresh 之间，每个 batch 后使用真实 EMA 更新 bank，`proto_ema_momentum=0.99`；
- `proto_temperature=0.07`；
- padded 或无效 prototype 槽位不会进入 loss 分母。

### 3.4 `proto_start` / `rel_start` 的准确含义

训练代码内部使用从 0 开始的 epoch index，而 checkpoint 文件使用人类习惯的 1-based epoch 数：

| 配置 | 实际含义 |
|---|---|
| `proto_start=50` | 第 1–50 轮为纯 SupLoss；从第 51 轮开始建立/使用 prototype 分支 |
| `rel_start=125` | 第 1–125 轮 RelLoss 不参与；从第 126 轮开始参与更新 |
| `checkpoint_0125.pth` | 完成第 125 轮后的状态，即 RelLoss 第一次参与更新之前 |
| `checkpoint_0135.pth` | 完成第 135 轮后的状态，即 RelLoss 已参与 10 轮更新 |

后文为了和 ID、JSON 一致，仍写 `start50/start125`，但应按上表理解。

---

## 4. 对比预训练公共配置

| 参数 | 数值 | 说明 |
|---|---:|---|
| epochs | 200 | 所有主预训练一致 |
| batch size | 64 | 单个任务固定 |
| optimizer | AdamW | 解耦 weight decay |
| initial LR | `1e-3` | 所有模型参数 |
| weight decay | `1e-4` | 固定 |
| LR milestones | `50/100/150` | 每到一个节点乘 0.1，即 `1e-3 → 1e-4 → 1e-5 → 1e-6` |
| projection dim | 128 | 对比投影空间维度 |
| queue size | 1088 | SupLoss queue |
| SupLoss temperature | 0.07 | 固定 |
| queue positives | 6 | `num_positive=6` |
| invalid queue | 排除 | 使用 `exclude_invalid_queue` |
| sampler | none | 不启用额外重采样 |
| DDP / SyncBN | 关闭 | 当前包以单 GPU 运行 |
| seed | 各实验表指定 | 不要私自替换 |

### 4.1 预训练增强 A2：动作保持增强

两个 view 使用完全相同的帧索引，即 `temporal_mode=shared`、最小时间重叠率 1.0。这样可以降低两个 view 因动作阶段不同而形成的假负面干扰。

| 增强 | 参数 |
|---|---|
| Random Resized Crop scale | `[0.85, 1.0]` |
| Random Resized Crop ratio | `[0.9, 1.1]` |
| horizontal flip | `p=0.5` |
| vertical flip | `p=0` |
| color jitter probability | `0.2` |
| brightness/contrast/saturation/hue | `[0.1, 0.1, 0.1, 0.02]` |
| grayscale | `p=0` |
| Gaussian blur | `p=0.1` |
| blur kernel / sigma | `5` / `[0.1, 1.0]` |

### 4.2 权重和诊断保存

- 完整 checkpoint：默认保存 `50/100/150/200`；
- 最后一轮始终保存，即使总 epochs 不是 50 的倍数；
- 每 10 轮保存 `prototype_diagnostics/proto_diag_epoch_XXXX.json`；
- prototype state 可用时，同时保存轻量 `proto_state_epoch_XXXX.pt`；
- RelLoss 实验额外保存 `checkpoint_{rel_start}.pth` 和 `checkpoint_{rel_start+10}.pth`；
- 同一节点由多个规则触发时只写一次，原因记录在 `save_reasons`；
- 自动续训从目录内编号最大的完整 `checkpoint_*.pth` 开始。

轻量 prototype 诊断用于检查：

- 每个 prototype 的 assignment 数量；
- strict-dead 和 near-dead prototype；
- assignment 的类别内变异系数与归一化 entropy；
- 同类 prototype 两两 cosine；
- 每个 prototype 与最近异类 prototype 的 cosine；
- 有效/无效 assignment 数量；
- 当前 ProtoLoss、RelLoss 是否处于启用阶段。

---

## 5. 微调与测试公共配置

### 5.1 默认 full fine-tune

| 参数 | 数值 |
|---|---:|
| epochs | 100 |
| optimizer | AdamW |
| batch size | 64 |
| backbone LR | `3e-4` |
| classification head LR | `1e-3` |
| weight decay | `1e-4` |
| milestones | `50/75`，每次乘 0.1 |
| loss | 普通、未加权 Cross-Entropy |
| best checkpoint | validation Balanced Accuracy 最高的 `best_val_balanced.pth` |
| periodic save | 每 10 epochs |

这里的“full”是全局微调：backbone 和分类头都更新，只是使用不同学习率；不是只训练分类头。`head_only` 只在 Stage 7 中专门测试。

### 5.2 微调增强

- RRC scale `[0.85,1.0]`、ratio `[0.9,1.1]`；
- horizontal flip `p=0.5`；
- vertical flip、color jitter、grayscale、blur 均关闭；
- 不启用 weighted sampler、weighted CE 或 focal loss。

### 5.3 测试规则

- 测试时关闭所有随机增强；
- batch size 64；
- 只测试由 validation Balanced Accuracy 选出的权重；
- 测试脚本受 `ALLOW_LOCKED_TEST=YES` 保护；
- Stage 0/T3/1/4/4B/5 的筛选阶段不得根据测试集返回调参。

---

## 6. Stage 0：复现性与基线

### 6.1 目的

Stage 0 先建立可信的 scratch 和 SupLoss-only 基准，并回答“同为 00143 摄像头的结果为何差很多”。它同时测量：

1. 完全相同 seed 1 重复运行的确定性误差；
2. seed 1/2/3 的正常随机波动；
3. 每个 seed 下 SupLoss 预训练相对 scratch 的配对增益。

### 6.2 预训练实验

| index | ID | seed | 配置 | 作用 |
|---:|---|---:|---|---|
| 0 | `sup_s1_a` | 1 | SupLoss-only | seed 1 第一次 |
| 1 | `sup_s1_b` | 1 | 与 `sup_s1_a` 完全相同 | 检查同 seed 是否可复现 |
| 2 | `sup_s2` | 2 | SupLoss-only | seed 2 |
| 3 | `sup_s3` | 3 | SupLoss-only | seed 3 |

所有任务均为 `contrastive_only`，因此 `num_prototypes` 和 positive mode 不产生实际作用。

### 6.3 微调实验与正确配对

| Scratch | SupLoss 预训练 | 用途 |
|---|---|---|
| `scratch_s1_a` | `sup_s1_a_ft` | seed 1 主配对 |
| `scratch_s1_b` | `sup_s1_b_ft` | 同 seed 重复性 |
| `scratch_s2` | `sup_s2_ft` | seed 2 配对 |
| `scratch_s3` | `sup_s3_ft` | seed 3 配对 |

三 seed 均值使用 `s1_a/s2/s3`；`s1_b` 是确定性副本，不应当作为第四个独立 seed 放入均值。

### 6.4 判断重点

- `sup_s1_a` 与 `sup_s1_b` 差距大：优先检查数据清单、源码 SHA256、GPU、resume 起点和非确定性算子；
- 三 seed 方差大：后续单 seed 的小幅提升不能作为有效结论；
- SupLoss−scratch 在 2/3 seed 为正且均值提升超过随机波动：才支持 SupLoss 预训练有效；
- 只有 seed 1 提升：不能直接把收益归因于预训练。

输出：

```text
results/cl_rgb_req_s0_20260719/
results/ft_rgb_req_s0_20260719/
```

---

## 7. Stage 1：Prototype loss 形式

### 7.1 这一阶段在比较什么

固定以下内容：

- seed 1；
- `λproto=0.1`；
- `proto_start=50`；
- `λrel=0`，RelLoss 完全关闭；
- prototype refresh、真实 EMA 和所有主训练参数相同。

只改变两个因素：

1. 每类 prototype 数量：1、2、3；
2. 同类 prototype 在 ProtoLoss 中如何作为正样本：`single`、`all`、`soft`。

### 7.2 `single`、`all`、`soft` 的准确含义

设样本特征为 \(q_i\)，同类有效 prototype 集合为 \(P_i\)，全部类别的有效 prototype 集合为 \(A\)，温度为 \(\tau\)。

#### `single`：只拉近硬分配到的 prototype

- KMeans/assignment 给每个样本一个所属 prototype；
- 正样本只有这个 assigned prototype；
- 所有有效异类 prototype 是负样本；
- 同类但未被分配到的其他 prototype 被移出分母，既不是正样本，也不会被当成负样本。

直观效果：允许一个类别保留多个子模式，每个样本只负责靠近自己的子模式。风险是硬 assignment 错误时，样本会被强制拉向错误 prototype。

#### `all`：等权拉近同类全部 prototype

- 同类所有有效 prototype 都作为正样本；
- 对每个同类 prototype 分别计算 log-softmax 项，再等权平均；
- 分母包含所有类别的全部有效 prototype。

直观效果：监督信号最强，但当每类有多个真正不同的动作子模式时，会要求一个样本同时靠近全部同类 prototype，可能压缩或破坏多模态结构。prototype 数越多，这种“同时靠近全部中心”的约束越强。

#### `soft`：按当前相似度软加权同类 prototype

- 同类全部 prototype 仍是正样本候选；
- 根据样本与同类 prototype 的相似度计算 softmax responsibility；
- 用 responsibility 对各正 prototype 的 log-probability 加权；
- responsibility 使用 `detach()`，即权重本身不反向传播；
- 分母仍包含全部有效 prototype。

直观效果：比 `all` 更偏向当前最匹配的同类 prototype，但不像 `single` 那样只有一个硬正样本。它旨在兼顾多模态保持与 assignment 稳定性。

> 当每类只有 1 个 prototype 时，`single/all/soft` 的正样本定义事实上退化为同一种情况。因此 P1 主要用于确认“增加一个 ProtoLoss 分支”本身的作用，而 P2–P7 才真正比较多 prototype 形式。

### 7.3 完整实验矩阵

| index | ID | mode | prototype/类 | `λproto` | 唯一问题 |
|---:|---|---|---:|---:|---|
| 0 | `p0_sup` | 无 ProtoLoss | 1（不生效） | 0 | SupLoss matched control |
| 1 | `p1_single_p1` | single | 1 | 0.1 | 单 prototype ProtoLoss 是否有益 |
| 2 | `p2_single_p2` | single | 2 | 0.1 | 两个子模式 + 硬 assignment |
| 3 | `p3_all_p2` | all | 2 | 0.1 | 两个 prototype 全部等权拉近 |
| 4 | `p4_soft_p2` | soft | 2 | 0.1 | 两个 prototype 软 responsibility |
| 5 | `p5_single_p3` | single | 3 | 0.1 | 三个子模式 + 硬 assignment |
| 6 | `p6_all_p3` | all | 3 | 0.1 | 三个 prototype 全部等权拉近 |
| 7 | `p7_soft_p3` | soft | 3 | 0.1 | 三个 prototype 软 responsibility |

每个 checkpoint 都使用默认 full fine-tune：backbone LR `3e-4`、head LR `1e-3`。

### 7.4 必须做的对照

- P1−P0：增加单 prototype 辅助目标是否有基本收益；
- P2/P3/P4：固定 2 prototypes，只比较 positive 定义；
- P5/P6/P7：固定 3 prototypes，只比较 positive 定义；
- P2−P1、P5−P1：`single` 下增加 prototype 数是否有益；
- P3−P6：`all` 是否随 prototype 数增加而恶化；
- P4−P7：`soft` 使用 2 还是 3 prototypes 更稳定。

### 7.5 结果解释

- `all` 随 prototype 数增加而变差，而 `single/soft` 不变差：支持 all-positive 导致模式坍缩；
- `single` 最好但 seed 波动大：可能受硬 assignment 不稳定影响；
- `soft` 最好且 dead prototype 少：支持软 responsibility；
- 所有 ProtoLoss 都不如 P0：检查 `λproto×Lproto` 与 SupLoss 的量级、assignment 覆盖率、dead prototype、类内 prototype cosine；
- P4/P6 优于 P0 但不优于对应 Null-proto：收益可能来自 prototype refresh/EMA 路径，而非 ProtoLoss 梯度，必须由 Stage 4B 确认。

输出：

```text
results/cl_rgb_req_s1_20260719/
results/ft_rgb_req_s1_20260719/
```

---

## 8. Stage T3：时间压缩消融

### 8.1 目的

验证当前 ResNet3D-18 将 16 帧压缩为深层 1 个时间位置，是否限制了动作顺序和局部运动信息。T3 只改变时间 stride，不改变数据、空间增强、SupLoss、训练轮数、优化器或参数形状。

### 8.2 时间维变化

| 位置 | current | T3/LFB-style |
|---|---:|---:|
| input | 16 | 16 |
| conv1 | 16 | 16 |
| maxpool | 8 | 16 |
| layer1 | 8 | 16 |
| layer2 | 4 | 8 |
| layer3 | 2 | 8 |
| layer4 | 1 | 8 |
| final adaptive pooling | 1 | 1 |

T3 修改：

- maxpool temporal kernel/stride 改为 1；
- layer2 保留唯一一次 temporal stride 2；
- layer3/layer4 只做空间下采样；
- 最终仍通过 AdaptiveAvgPool 聚合为固定向量。

这是对当前 BasicBlock ResNet3D-18 的 LFB-style 适配，不是 LFB 论文 ResNet-50 I3D-NL 的逐层复制。

### 8.3 实验矩阵

预训练：

| ID | seed | 配置 |
|---|---:|---|
| `t3_sup_s1` | 1 | T3 backbone + SupLoss-only |
| `t3_sup_s2` | 2 | T3 backbone + SupLoss-only |
| `t3_sup_s3` | 3 | T3 backbone + SupLoss-only |

微调：

- `scratch_t3_s1/s2/s3`：T3 backbone 随机初始化；
- `t3_sup_s1_ft/s2_ft/s3_ft`：加载同 seed T3 SupLoss checkpoint。

### 8.4 公平比较

| current（Stage 0） | T3 | 测量 |
|---|---|---|
| `scratch_s1_a/s2/s3` | `scratch_t3_s1/s2/s3` | 架构本身对监督训练的影响 |
| `sup_s1_a_ft/s2_ft/s3_ft` | `t3_sup_s1_ft/s2_ft/s3_ft` | 时间维对预训练表征的影响 |

不要把 `sup_s1_b` 当作独立 seed。

### 8.5 结果解释

```text
Δscratch = mean(T3 scratch) − mean(current scratch)
Δsup     = mean(T3 SupLoss) − mean(current SupLoss)
```

- 两者均稳定为正：时间过度压缩很可能是主要限制；
- 只有 `Δsup>0`：保留时间位置主要帮助对比表征；
- 只有 `Δscratch>0`：当前预训练目标或增强没有利用新增时间分辨率；
- 两者均不提升：时间压缩不是主要瓶颈，或 T3 的计算/优化代价抵消收益。

T3 必须在预训练、微调和测试中始终使用 `backbone_temporal_mode=t3_lfb`，不能加载到 `current` 模式。

输出：

```text
results/cl_rgb_req_t3_20260721/
results/ft_rgb_req_t3_20260721/
```

---

## 9. Stage 4：Relation loss 机制

### 9.1 RelLoss 实际约束什么

RelLoss 不直接进行分类。每个 batch 中，它先用当前特征构造一个可微的 prototype EMA preview，然后比较更新前后 prototype 的 cosine distance：

```text
D = 1 − cosine_similarity
```

当前实现包含：

- `same` 项：同类 prototype 更新后变得更远，且超过 `same_margin=0.01` 时惩罚；
- `diff` 项：异类 prototype 更新后变得更近，且超过 `diff_margin=0.01` 时惩罚；
- far-reward 权重为 0，因此本实验只惩罚错误方向，不主动奖励无限拉远。

总目标中的实际贡献是：

```text
λrel × (same_weight × Lsame + diff_weight × Ldiff)
```

Stage 4 完全关闭 ProtoLoss：`λproto=0`。除 R0 外，所有实验均建立 3 prototypes/类，真实 bank EMA 固定 0.99。

### 9.2 关键参数

| 参数 | 含义 |
|---|---|
| `same_weight` | 是否惩罚同类 prototype 变远；0 表示关闭 same 项 |
| `diff_weight` | 是否惩罚异类 prototype 变近；本阶段固定为 1 |
| `topk_diff_classes` | 每个更新 prototype 只关注更新前距离最近的 K 个异类“类别”；0 表示全部异类类别 |
| `preview_ema_momentum` | 构造可微 preview bank 时旧 prototype 的保留比例；越大，单 batch preview 位移越小 |
| `rel_start` | 前多少轮保持 RelLoss 关闭 |
| `λrel` | RelLoss 对总梯度的外部缩放系数 |

Top-k 是按类别选择，不是按单个 prototype 选择：先用该异类中最近 prototype 的旧距离代表类别距离，选最近 K 类，再让所选类别下全部有效 prototype 参与 diff 项。

### 9.3 完整 17 项矩阵

| R | ID | same/diff | top-k | start | preview EMA | `λrel` | 主要问题 |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | `r0_sup` | 无 RelLoss | — | — | — | 0 | SupLoss-only matched control |
| 1 | `r1_current_p3` | same+diff | all(0) | 50 | 0.5 | 0.5 | 复现原始 relation 设置 |
| 2 | `r2_topk3` | same+diff | 3 | 50 | 0.5 | 0.5 | 只关注最近 3 类 |
| 3 | `r3_topk5` | same+diff | 5 | 50 | 0.5 | 0.5 | 最近 5 类 |
| 4 | `r4_topk10` | same+diff | 10 | 50 | 0.5 | 0.5 | 最近 10 类 |
| 5 | `r5_diff_k3` | diff-only | 3 | 50 | 0.5 | 0.5 | 去掉 same 项 |
| 6 | `r6_diff_k5` | diff-only | 5 | 50 | 0.5 | 0.5 | diff-only 最近 5 类 |
| 7 | `r7_diff_k10` | diff-only | 10 | 50 | 0.5 | 0.5 | diff-only 最近 10 类 |
| 8 | `r8_diff_k3_s100` | diff-only | 3 | 100 | 0.5 | 0.5 | 延后到 start100 |
| 9 | `r9_diff_k3_s125` | diff-only | 3 | 125 | 0.5 | 0.5 | 延后到 start125 |
| 10 | `r10_diff_k3_s150` | diff-only | 3 | 150 | 0.5 | 0.5 | 延后到 start150 |
| 11 | `r11_diff_k3_s125_pm08` | diff-only | 3 | 125 | 0.8 | 0.5 | 较平滑 preview |
| 12 | `r12_diff_k3_s125_pm09` | diff-only | 3 | 125 | 0.9 | 0.5 | 更平滑 preview |
| 13 | `r13_diff_k3_s125_pm099` | diff-only | 3 | 125 | 0.99 | 0.5 | preview 位移非常小 |
| 14 | `r14_diff_k3_s125_pm09_l1` | diff-only | 3 | 125 | 0.9 | 1.0 | 将 R12 的 RelLoss 权重放大 2 倍 |
| 15 | `r15_diff_k3_s125_pm09_l2` | diff-only | 3 | 125 | 0.9 | 2.0 | 将 R12 的 RelLoss 权重放大 4 倍 |
| 16 | `r16_diff_k3_s125_pm09_l5` | diff-only | 3 | 125 | 0.9 | 5.0 | 将 R12 的 RelLoss 权重放大 10 倍 |

### 9.4 分组阅读顺序

1. R0 vs R1：原始 relation 分支是否超过 SupLoss；
2. R1 vs R2/R3/R4：全部异类类别是否过于分散，hard-negative top-k 是否更有效；
3. R2/R3/R4 vs R5/R6/R7：same 项是否损害类内多模态结构；
4. R5 vs R8/R9/R10：RelLoss 是否因为过早启动而作用于未稳定 prototype；
5. R9 vs R11/R12/R13：preview 更新幅度的灵敏度；
6. R12 vs R14/R15/R16：固定最佳机制后，只改变 `λrel`。

### 9.5 R14–R16 的特别说明

R14、R15、R16 并不是三种新 RelLoss。它们与 R12 的以下参数完全相同：

```text
diff-only
top-k = 3
rel_start = 125
preview_ema_momentum = 0.9
same_weight = 0
diff_weight = 1
3 prototypes/class
proto_ema_momentum = 0.99
```

唯一变化是 `λrel`：

| 对照 | `λrel` | 相对 R12 的加权 RelLoss 强度 |
|---|---:|---:|
| R12 | 0.5 | 1×，本组基准 |
| R14 | 1.0 | 2× |
| R15 | 2.0 | 4× |
| R16 | 5.0 | 10× |

因此应同时报告原始 `Lrel` 和 `λrel×Lrel`。若 R15 比 R12 好，不能解释为 RelLoss 形式更好，只能说明在当前原始量级下更强的梯度权重更合适。若 R16 的训练 loss、梯度或验证性能明显不稳定，说明 5.0 可能已使辅助目标主导主损失。

### 9.6 结果判断

- Rel-only 至少应稳定高于 R0，才值得进入组合阶段；
- 单个 seed 的最好结果只能生成候选，不能确认机制；
- Rel 候选还必须在 Stage 4B 高于 Null-rel，才能归因于 RelLoss 梯度；
- 若只有增大 `λrel` 才有效，应检查原始 RelLoss 的非零率和梯度量级，而不是无限继续放大；
- 若 R13 近似无效，可能是 preview EMA 0.99 造成位移太小、方向损失接近零。

输出：

```text
results/cl_rgb_req_s4_20260719/
results/ft_rgb_req_s4_20260719/
```

---

## 10. Stage 4B：三 seed 与 Null-loss 确认

### 10.1 为什么增加这一阶段

Stage 1 和 Stage 4 的大部分筛选只有 seed 1。Stage 4B 不再大范围搜索，而是对候选和匹配对照补齐 seed，并使用 Null-loss 区分：

```text
辅助损失梯度贡献
vs
prototype state / refresh / EMA / 额外计算路径贡献
vs
随机波动
```

### 10.2 Family 定义

| Family | seeds | 配置 | 主要比较 |
|---|---|---|---|
| `rel_r0` | 原 seed1 + 新 seed2/3 | SupLoss-only | relation matched baseline |
| `rel_r9` | 原 seed1 + 新 seed2/3 | diff-only, K3, start125, preview EMA0.5, `λrel=0.5` | R9−R0、R9−Null-rel |
| `rel_r12` | 原 seed1 + 新 seed2/3 | 与 R9 相同，但 preview EMA0.9 | R12−R0、R12−Null-rel |
| `rel_null` | 新 seed1/2/3 | 保留 R9 全路径，但 `λrel=0` | 识别非梯度路径收益 |
| `proto_p0` | 原 seed1 + 新 seed2/3 | SupLoss-only | prototype matched baseline |
| `proto_null_p2` | 新 seed1/2/3 | soft, P2, start50, `λproto=0` | P4 的零梯度对照 |
| `proto_p4` | 原 seed1 + 新 seed2/3 | soft, P2, start50, `λproto=0.1` | P4−P0、P4−Null-P2 |
| `proto_null_p3` | 新 seed1/2/3 | all, P3, start50, `λproto=0` | P6 的零梯度对照 |
| `proto_p6` | 原 seed1 + 新 seed2/3 | all, P3, start50, `λproto=0.1` | P6−P0、P6−Null-P3 |
| `rel_r7_optional` | 原 seed1 + 可选 seed2/3 | diff-only, K10, start50, EMA0.5, `λrel=0.5` | 预算允许时确认 R7 |

必做数组 index 为 `0-14,17-22`，共 21 个预训练和 21 个 full fine-tune；index `15-16` 是可选 R7。

### 10.3 Null 对照的正确解释

- `candidate > matched Null > SupLoss`：部分收益来自 prototype 路径，额外部分才来自损失梯度；
- `candidate ≈ matched Null > SupLoss`：没有证据证明新损失本身有效；
- `candidate > SupLoss ≈ matched Null`：更支持收益来自新损失梯度；
- `candidate < matched Null`：辅助损失梯度可能正在破坏 prototype 路径带来的收益。

### 10.4 进入 Stage 5 的最低判据

1. family 完成三个 seeds；
2. 三 seed 平均 best BA 和 last-10 BA 高于匹配对照；
3. 至少 2/3 seed 的严格配对差为正；
4. R9/R12 高于 Null-rel；
5. P4/P6 高于各自 Null-proto；
6. 提升不是由极少数小 support 类别单独驱动；
7. 提升大于同配置重复运行的噪声量级。

输出：

```text
results/cl_rgb_req_s4b_confirm_20260724/
results/ft_rgb_req_s4b_confirm_20260724/
results/ft_rgb_req_s4b_confirm_20260724/analysis/confirmation_summary.md
```

Stage 4B 完成前不要运行测试集，也不要默认继续采用旧的 R15 推荐。

---

## 11. Stage 5：ProtoLoss + RelLoss 组合

### 11.1 目的与前提

Stage 5 检查两种已经分别隔离的辅助损失是否互补。它不是用来弥补 Stage 1/4B 中“单项无效”的阶段：如果某个 loss 没有稳定超过 matched control 和 Null-loss，应先修改机制，再决定是否组合。

默认组合基准：

```text
ProtoLoss：soft，2 prototypes/class，λproto=0.1
RelLoss：diff-only，top-k=3，preview EMA=0.9，λrel=2
真实 prototype EMA：0.99
```

### 11.2 完整实验矩阵

| index | ID | ProtoLoss | RelLoss | start | 作用 |
|---:|---|---|---|---|---|
| 0 | `c0_sup` | 关闭 | 关闭 | — | SupLoss-only control |
| 1 | `c1_proto_only` | soft-P2, `λp=0.1` | 关闭 | P50 | ProtoLoss 单项 |
| 2 | `c2_rel_only` | 关闭 | K3 diff, `λr=2` | R125 | RelLoss 单项 |
| 3 | `c3_both_s50` | soft-P2, `λp=0.1` | K3 diff, `λr=2` | P50/R50 | 两项同时较早启动 |
| 4 | `c4_p50_r125` | soft-P2, `λp=0.1` | K3 diff, `λr=2` | P50/R125 | 先稳定 prototype，再启用 RelLoss |
| 5 | `c5_p50_r100` | 同 C4 | 同 C4 | P50/R100 | RelLoss 较早 |
| 6 | `c6_p50_r150` | 同 C4 | 同 C4 | P50/R150 | RelLoss 较晚 |
| 7 | `c7_p25_r125` | 同 C4 | 同 C4 | P25/R125 | ProtoLoss 提前 |
| 8 | `c8_lp005` | soft-P2, `λp=0.05` | K3 diff, `λr=2` | P50/R125 | ProtoLoss 权重减半 |
| 9 | `c9_lp020` | soft-P2, `λp=0.2` | K3 diff, `λr=2` | P50/R125 | ProtoLoss 权重加倍 |
| 10 | `c10_lr050` | soft-P2, `λp=0.1` | K3 diff, `λr=0.5` | P50/R125 | RelLoss 权重降低 |
| 11 | `c11_lr500` | soft-P2, `λp=0.1` | K3 diff, `λr=5` | P50/R125 | RelLoss 权重提高 |

### 11.3 对照关系

- C1−C0：ProtoLoss 单项收益；
- C2−C0：RelLoss 单项收益；
- C4−max(C1,C2)：组合是否超过最强单项，这是“互补”的核心证据；
- C3/C4/C5/C6：RelLoss 启动时间；
- C4 vs C7：ProtoLoss 是否需要更早启动；
- C8/C4/C9：`λproto=0.05/0.1/0.2`；
- C10/C4/C11：`λrel=0.5/2/5`。

### 11.4 结果解释

- C4 超过 C1、C2 且在多 seed 稳定：支持互补；
- C1、C2 各自有效但组合下降：重点检查两项梯度方向冲突；
- C3 差、C4 好：RelLoss 需要等待 prototype 稳定；
- C6 好于 C4：RelLoss 可能仍启动过早；
- 高 `λrel` 提升训练目标但降低验证 BA：辅助几何约束正在压过分类相关结构；
- 不要仅凭总 loss 判断，需同时检查 weighted contribution、assignment、dead prototype 和 per-class recall。

输出：

```text
results/cl_rgb_req_s5_20260719/
results/ft_rgb_req_s5_20260719/
```

---

## 12. Stage 7：微调协议与最终测试

### 12.1 目的

预训练表征可能因微调学习率不合适而被低估或快速破坏。Stage 7 将候选 checkpoint 冻结后，统一比较 head-only 与不同 backbone LR 的 full fine-tune。

提交前必须更新并检查：

```text
stage7_finetune/config/selected_sources.json
```

候选类别：

| candidate | 来源 |
|---|---|
| `scratch` | 随机初始化 current backbone |
| `sup` | Stage 0 最终 SupLoss-only |
| `temporal_t3` | T3 SupLoss-only |
| `proto` | Stage 4B 确认后的最佳 ProtoLoss |
| `rel` | Stage 4B 确认后的最佳 RelLoss |
| `proto_rel` | Stage 5 最佳组合 |

JSON 中现有路径只是初始占位/候选，不代表最终结论。必须依据 validation 结果替换并冻结 SHA256。

### 12.2 22 个微调任务如何组成

Scratch：

| mode | 含义 |
|---|---|
| `head` | 对随机 backbone 做 head-only 主要作为协议完整性检查 |
| `full` | 从随机初始化端到端训练 |

其余 5 个预训练候选，每个运行 4 种协议：

| mode | backbone | head |
|---|---:|---:|
| `head` | 冻结 | LR `1e-3` |
| `full_b1e4` | LR `1e-4` | LR `1e-3` |
| `full_b3e4` | LR `3e-4` | LR `1e-3` |
| `full_b1e3` | LR `1e-3` | LR `1e-3` |

总数：

```text
2 个 scratch 协议 + 5 个预训练候选 × 4 个协议 = 22
```

### 12.3 如何解释 head-only 和 full

- head-only 好：冻结表征已经线性可分；
- head-only 一般、低 LR full 好：表征有用，但需要温和任务适配；
- 高 LR full 最好：预训练主要提供初始化，而非稳定线性表征；
- 高 LR full 明显下降：可能发生 catastrophic forgetting；
- 不同预训练方法必须比较各自最合适的协议，同时也报告统一 `3e-4` 协议以保证公平。

T3 候选必须使用 `t3_lfb`；其他候选使用 `current`。结构模式与 checkpoint 不一致时，即使参数 shape 能加载，也不能视为合法比较。

### 12.4 最终测试

只有 validation 决策全部冻结后才执行：

```bash
sbatch --export=ALL,ALLOW_LOCKED_TEST=YES \
  codex_script/rgb_required_5stages_20260719/stage7_finetune/slurm/03_test_array.slurm
```

然后汇总：

```bash
sbatch codex_script/rgb_required_5stages_20260719/stage7_finetune/slurm/04_summarize.slurm
```

正式测试排名：

```text
results/ft_rgb_req_s7_20260719/test/rgb_test_results_ranked.csv
```

禁止看到测试结果后返回 Stage 1/4/5 调参，否则测试集将不再是独立评估。

---

## 13. 各阶段任务规模与输出速查

| 阶段 | 预训练 | 微调 | 主要输出 |
|---|---:|---:|---|
| Stage 0 | 4 | 8 | `cl_rgb_req_s0_20260719` / `ft_rgb_req_s0_20260719` |
| Stage 1 | 8 | 8 | `cl_rgb_req_s1_20260719` / `ft_rgb_req_s1_20260719` |
| Stage T3 | 3 | 6 | `cl_rgb_req_t3_20260721` / `ft_rgb_req_t3_20260721` |
| Stage 4 | 17 | 自动生成 17 | `cl_rgb_req_s4_20260719` / `ft_rgb_req_s4_20260719` |
| Stage 4B 必做 | 21 | 21 | `cl_rgb_req_s4b_confirm_20260724` / `ft_rgb_req_s4b_confirm_20260724` |
| Stage 4B 可选 R7 | 2 | 2 | 同上 |
| Stage 5 | 12 | 自动生成 12 | `cl_rgb_req_s5_20260719` / `ft_rgb_req_s5_20260719` |
| Stage 7 | 0 | 22 | `ft_rgb_req_s7_20260719` |

Stage 4B 的任务数只包含本阶段新增运行；其汇总还会合并 Stage 1/4 已有 seed 1。

---

## 14. Slurm 与运行环境

GPU 作业公共资源：

| 项目 | 配置 |
|---|---|
| partitions | `gpu,gpu-h100,gpu-h100-nvl` |
| qos | `gpu` |
| GPU | 1 |
| CPU | 12 |
| memory | 60 GB |
| 默认预训练时限 | 18 h |
| 默认微调时限 | 14 h |
| 默认测试时限 | 6 h |
| T3 预训练/微调 | 24 h / 18 h |

环境：

```bash
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
```

脚本使用 `${CONDA_PREFIX}/bin/python`，并设置：

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

每次运行会写 provenance，记录配置、完整命令、Git commit、入口脚本 SHA256、Slurm 作业号和 GPU 可见性。分析异常差距时应先检查 provenance，而不是只看最终准确率。

---

## 15. 最终报告至少应包含什么

每个主要候选至少报告：

1. 每个 seed 的 best validation Balanced Accuracy；
2. 三 seed 均值、样本标准差和逐 seed 配对差；
3. final 与 last-10 validation 稳定性；
4. 相对 scratch、SupLoss-only 和 matched Null-loss 的提升；
5. per-class recall 与类别 support；
6. SupLoss、原始 ProtoLoss/RelLoss、加权贡献及非零率；
7. prototype assignment、dead/near-dead、entropy、类内与异类 cosine；
8. 训练过程中是否出现 NaN/Inf、异常梯度或 resume 差异；
9. head-only 与最佳 full fine-tune；
10. 最终冻结后的一次性测试结果。

只有“多 seed、严格配对、超过匹配对照和 Null-loss、且不是单一小类别驱动”的提升，才能作为 ProtoLoss 或 RelLoss 有效的主要证据。
