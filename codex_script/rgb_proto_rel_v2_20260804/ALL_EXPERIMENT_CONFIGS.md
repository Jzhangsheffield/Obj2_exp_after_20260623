# ProtoLoss v2 / RelLoss v2 全部实验配置

配置唯一事实源：`common/experiment_plan.json`。本文解释参数含义；若文档与 JSON 不一致，以 JSON 为准。

## 1. 固定的数据、模型和训练条件

| 项目 | 设置 |
|---|---|
| 集群项目根目录 | `/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623` |
| 数据根目录 | `/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset` |
| train manifest | `N_as_test/train_manifest_except_take_put.jsonl` |
| validation manifest | `N_as_test/val_manifest_except_take_put.jsonl` |
| locked test manifest | `N_as_test/test_manifest_except_take_put.jsonl` |
| label map | `label_map_except_take_put.json` |
| 类别数 | 15，tier1，排除 take/put |
| 摄像头 | RGB `00143` |
| 输入 | 16 帧，224×224 |
| backbone | ResNet3D-18（标准 temporal stride；不使用 T3） |
| projection dimension | 128 |
| RGB normalization | mean=`[0.3752,0.3864,0.3960]`; std=`[0.2934,0.2724,0.2644]` |
| 主损失 | supervised contrastive loss (`suploss`) |
| queue / temperature / positives | 1088 / 0.07 / 6 |
| 预训练 | 200 epochs；batch 64；AdamW；LR 1e-3；weight decay 1e-4；milestones 50/100/150 |
| sampler | `none`，不改变旧 matched baseline 的采样分布 |
| checkpoint | 每 50 epochs；Rel 启动前后额外 checkpoint 窗口为 10 epochs |
| 微调 | full fine-tuning，100 epochs，batch 64，AdamW，backbone LR 3e-4，head LR 1e-3，weight decay 1e-4，milestones 50/75；周期 checkpoint 每 25 epochs |
| 模型选择 | 仅按 validation balanced accuracy 选择 `best_val_balanced.pth` |

## 2. 数据增强

两个对比视图使用完全相同的 16 帧时间索引（`temporal_mode=shared`, overlap=1.0），因此辅助损失学习的是动作外观/状态子模态，而不是由时间错位制造的伪差异。

| 增强 | 参数 |
|---|---|
| Random resized crop | scale `[0.85,1.0]`, ratio `[0.9,1.1]`, output 224×224 |
| Horizontal flip | p=0.5 |
| Vertical flip | p=0 |
| Color jitter | p=0.2；brightness/contrast/saturation/hue=`0.1/0.1/0.1/0.02` |
| Grayscale | p=0 |
| Gaussian blur | p=0.1；kernel=5；sigma `[0.1,1.0]` |

微调保留 crop 与 horizontal flip；jitter、gray、blur 均关闭。验证与测试不使用随机增强。

## 3. ProtoLoss v2 定义

每类默认有 `M=2` 个 prototype。启动 epoch=50 时仍利用原流程完成一次 KMeans 初始化，但 `recluster_interval=10000`，因此 200 epoch 内不会再次硬重聚类。随后 bank 只用 teacher 的软责任更新。

### 3.1 teacher soft responsibility

teacher view 特征 `k` 与本类 prototypes 计算 cosine logits：

`r_ij = softmax(cos(k_i,p_cj) / tau_assign)`，其中 `tau_assign=0.05`。

`same_view_soft` 用在线特征 `q` 自己产生 target，容易形成自强化；`teacher_soft` 用 momentum teacher 产生更稳定的 stop-gradient target；`teacher_balanced` 再对当前 batch 内每个类别执行 3 次 Sinkhorn 归一化，避免所有样本落到同一个 prototype。

### 3.2 assignment prediction

在线特征 `q` 预测全部类别的全部有效 prototypes，以 teacher responsibility 作为软 target；prediction temperature=`0.07`。这既保留正确类约束，也允许同类动作存在多个连续子模态。

### 3.3 balance 与 diversity

- `balance_weight=0.2`：每类平均 responsibility 与均匀分布之间的 KL。它主要在非完全平衡、batch 太小或 Sinkhorn fallback 时提供约束。
- `diversity_weight=0.1`：若同类两个 preview prototypes 的 cosine similarity 高于 `0.85`，施加 hinge penalty，防止两个 prototype 退化成同一个方向。
- preview prototype 使用在线 q 与当前 responsibility 做可微更新，momentum=`0.5`；真实 bank 使用 teacher 做无梯度 EMA，momentum=`0.99`。

ProtoLoss v2 内部为：

`L_proto_v2 = L_assign + 0.2 L_balance + 0.1 L_diversity`

加入总损失时 `lambda_proto=0.1`，所以有效系数分别是 `0.1 / 0.02 / 0.01`。诊断必须同时报告未加权分量，避免总值掩盖某一项失效。

## 4. RelLoss v2 定义

对每个样本：正类分数是该样本对本类 prototypes 的 responsibility-weighted similarity；每个负类只保留最相似 prototype，再选择相似度最高的 top-K=3 个负类。

### 4.1 rank

`L_rank = softplus((s_neg - s_pos + margin) / tau) * tau`

其中 margin=`0.05`，tau=`0.05`。它只关注最易混淆的负类，避免大量简单负类稀释梯度。样本损失先在类别内平均，再在出现的类别之间平均，降低类别频率影响。

### 4.2 rank_direction

用当前 batch 对 prototype 做一次可微 preview update，并比较更新前后正负 gap。只有旧 gap 小于 margin 的 pair 被 gate 激活；要求新 gap 至少比旧 gap 增加 `direction_delta=0.005`。direction 内部权重=`0.25`。

`L_rel_v2 = L_rank + 0.25 L_direction`

加入总损失时 `lambda_rel=0.5`，有效系数为 `0.5` 和 `0.125`。Rel 在 epoch 75 启动，epoch 75–100 用 cosine schedule 从 0 ramp 到完整权重，此后保持完整权重直到 epoch 200；这是为避免早期不可靠 prototype 反向污染 backbone。

## 5. Stage 0：零权重与源码接入审计（3 runs）

| index | ID | epochs | 分支 | 权重 | 目的 |
|---:|---|---:|---|---|---|
| 0 | i0_sup_s1 | 60 | SupLoss only | proto=0, rel=0 | 参考轨迹 |
| 1 | i1_null_proto_s1 | 60 | 执行 Proto-v2 | proto=0 | 验证 null-proto 不改变模型更新 |
| 2 | i2_null_rel_s1 | 60 | 执行 Rel-v2 | rel=0 | 验证 null-rel 不改变模型更新 |

三组 seed=1。此阶段不微调。`audit_null_paths.py` 比较最终 state dict 的 L2 与最大绝对差异。若不为零，不能将后续差异解释为损失本身作用。

## 6. Stage 1：Proto 机制筛选（5 runs，seed 1）

| index | ID | assignment | balance | diversity | lambda_proto | 回答的问题 |
|---:|---|---|---:|---:|---:|---|
| 0 | p0_sup_s1 | 无 | 0 | 0 | 0 | matched SupLoss control |
| 1 | p1_teacher_soft_s1 | teacher soft | 0 | 0 | 0.1 | teacher target 是否有效 |
| 2 | p2_teacher_bal_s1 | teacher + Sinkhorn | 0 | 0 | 0.1 | assignment 平衡本身是否有效 |
| 3 | p3_bal_div_s1 | teacher + Sinkhorn | 0 | 0.1 | 0.1 | prototype 分离约束是否有效 |
| 4 | p4_proto_v2_full_s1 | teacher + Sinkhorn | 0.2 | 0.1 | 0.1 | 完整 ProtoLoss v2 |

这是筛选而非统计确认。优先选择 validation BA 不下降且 assignment/几何诊断健康的机制；不能只按单 seed 的最高点决定最终参数。

## 7. Stage 2：Proto 三 seed 确认（9 runs）

每组 seeds=1/2/3：

- P0：SupLoss-only。
- PN：Null-proto，执行相同 prototype 初始化、assignment、bank 更新，但 `lambda_proto=0`。
- PV2：完整 teacher-balanced ProtoLoss v2，M=2，start=50，`lambda_proto=0.1`。

核心效应是 `PV2 - PN`；`PN - P0` 检验代码路径/RNG 影响。若 PV2 平均 BA 没有提高，或只由单一 seed 驱动，就不应把它带入最终组合。

## 8. Stage 3：Rel 三 seed 确认（12 runs）

每组 seeds=1/2/3：

| family | 分支 | lambda_rel | 作用 |
|---|---|---:|---|
| R0 | SupLoss only | 0 | matched control |
| RN | Rel 路径完整运行 | 0 | Null-rel control |
| RRANK | hard-negative rank | 0.5 | 验证 top-K margin 排序 |
| RV2 | rank + gated direction | 0.5 | 验证 direction 是否额外有益 |

共同设置：M=2，prototype init=50，Rel 从 epoch 75 开始、到 epoch 100 完成 ramp，随后持续到训练结束；top-K=3，margin=0.05。主要效应分别是 `RRANK-RN` 和 `RV2-RRANK`。

## 9. Stage 4：2×2 三 seed 因子实验（12 runs）

| family | Proto v2 | Rel v2 | 目的 |
|---|---:|---:|---|
| F0 | 0 | 0 | SupLoss baseline |
| F1 | 1 | 0 | Proto 主效应 |
| F2 | 0 | 1 | Rel 主效应 |
| F3 | 1 | 1 | 联合效果与交互 |

每个 family seeds=1/2/3。交互可用 `(F3-F1)-(F2-F0)` 直观衡量：大于 0 表示协同，小于 0 表示相互干扰。只有 Stage 2/3 已有正信号才值得运行。

## 10. 诊断字段与判断重点

`v2_diagnostics.jsonl` 至少包括：

| 字段 | 含义 | 风险信号 |
|---|---|---|
| assignment_entropy | soft responsibility entropy | 长期接近 0：过硬/单 prototype；接近 log(M) 且几何不分离：无法形成子模态 |
| proto_assign | sample-to-all-prototypes 软交叉熵 | 启动后突增或不下降 |
| proto_balance | 类内平均 assignment 的均匀 KL | 持续高：使用极不均衡 |
| proto_diversity | 超过 cosine margin 的 hinge | 持续高且不下降：prototype collapse |
| rel_rank | hard-negative 排序项 | active window 内不下降 |
| rel_direction | preview gap 改善项 | 总为 0：gate/preview 无效；过大：更新方向冲突 |
| hard_negative_similarity | 最危险负类相似度 | 训练后不降或上升 |
| margin_violation | top-K pair 违反 margin 比例 | active window 内无下降 |
| dead_prototypes_in_batch | 当前诊断 batch 中硬分配为 0 的有效 prototype 数 | 长期非零且集中在同一 prototype |
| same_class_proto_cos_mean/max | 同类 prototype 两两 cosine | 长期接近 1 表示 collapse |
| assignment_soft_mass_min/max | 当前 batch 中最小/最大 soft assignment 质量 | 比值长期极端表示使用失衡 |
| bank_update_mean/max | 上一次 teacher EMA 更新位移 | 长期为 0 表示 bank 没有学习 |

原训练程序每 10 epochs 还会保存 prototype assignment/geometry 诊断。应联合检查每个 prototype assignment 数量、dead prototype、同类 prototype 两两 cosine，以及 epoch 50/75/100 前后的变化。

## 11. 继续/停止标准

1. Stage 0：null path 必须解释清楚；理想为权重逐元素完全一致。
2. Stage 1：至少一个 Proto-v2 配置相对 P0 不下降，并改善 assignment/geometry，才进入 Stage 2。
3. Stage 2：PV2 相对 PN 在多数 seeds 同方向，均值提升大于 seed 波动的合理比例，才认为 Proto-v2 有候选价值。
4. Stage 3：Rel 的 BA 与 macro-F1不能明显下降，同时 margin violation/hard-negative similarity 应出现机制一致的改善。
5. Stage 4：只在前两项通过后评估联合效果。所有设置冻结以后才能打开测试锁。

若机制诊断明显改善但 validation 不升，下一步优先调整启动窗口和损失权重；若机制诊断本身不改善，优先改 assignment/geometry 定义，而不是扩大 seed 数或盲目扫描 lambda。
