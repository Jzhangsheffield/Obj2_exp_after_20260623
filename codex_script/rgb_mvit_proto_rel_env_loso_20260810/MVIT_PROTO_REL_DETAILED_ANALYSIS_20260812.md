# K400 MViT-v2-S 上旧版 ProtoLoss / RelLoss 详细分析报告

生成日期：2026-08-12  
结果目录：`results/rgb_mvit_pr_env_loso_20260810`  
分析范围：Stage 1、Stage 2A、Stage 3A、Stage 4；M/J/N 开发集上的 inner-val 筛选，以及完全留出的 MR outer-test（385 个样本）

## 1. 结论摘要

### 1.1 可以确认的结论

1. **MViT-v2-S 上的 SupLoss 对比预训练是有用的。**K400 直接全参数微调的最佳验证 balanced accuracy（BA）为 91.64%，SupLoss-only 为 94.45%，提高 **2.81 个百分点**；accuracy 从 93.95% 提高到 95.56%，macro-F1 从 92.47% 提高到 94.15%。这说明更换为 K400 MViT-v2-S 后，对比学习已经不再像早期 ResNet3D-18 那样无法形成可用动作表示。

2. **旧版 ProtoLoss 与 RelLoss 都确实参与了计算，并能改变表示几何；训练过程没有数值崩溃。**29 个预训练实验的日志均未发现 NaN/Inf，所有 P2/P3 配置在最终 epoch 均无 strict dead prototype；RelLoss 配置明显降低了最近异类 prototype 的余弦相似度。

3. **但是，目前不能得出“ProtoLoss 或 RelLoss 已稳定提高分类性能”的最终结论。**当前只有一个随机种子和一个 held-out subject（MR），并同时比较了28个新增损失配置；inner-val 只有248个样本，MR test 也只有385个样本。最高值仍受到小类样本数与多重比较影响。

4. **三种光照对应三个 prototype 的假设没有得到支持。**P3 assignment 与 `left/normal/right` 的最终平均 NMI 约为 0.05–0.10、ARI 多数低于 0.05，Null 配置也达到相近数值；P3 没有一致优于 P2。因此后续不应因为存在三种光照就固定采用 P=3。

5. **prototype 的主要问题不是严格死亡，而是同类 prototype 方向高度重合。**P2/P3 最终同类 prototype 平均余弦相似度通常为 0.992–0.999。K-means 能把样本分给不同槽位，但 prototype 在归一化表示空间中几乎同方向，说明当前损失没有形成明确、稳定、具有语义的类内子模式。

6. **MR 外层测试进一步确认 SupLoss 有效。**Direct FT 的 test BA/macro-F1/accuracy 为 76.71%/74.80%/81.30%，SupLoss 为 83.39%/82.01%/82.34%，分别提高 **6.68/7.21/1.04 pp**。增益主要来自 `press/tear/adjust/close` 等少数类，因此 BA/F1 的改善显著大于 overall accuracy。

7. **外层测试否定了“hard P2 ProtoLoss 是当前最佳 Proto 设置”的初步判断。**`ph2_l010` 虽在 inner-val 相对 P2 Null 高 1.61 pp，但在 MR test 反而低 **3.38 pp BA、2.63 pp F1、2.34 pp accuracy**。本轮没有任何 Proto-only active 配置相对对应 Null 同时呈现可信、较大的三指标增益。

8. **RelLoss 的新首选信号变为 `rl3_k3_s125`，而不是验证集最高的 `re2_k10_s50`。**`rl3` 在 MR test 达到全组最高的 **91.06% BA、88.78% macro-F1**；相对完全匹配的 P3 Null `rn3`，BA +5.11 pp、F1 +2.42 pp，但 accuracy 相同。这说明收益主要来自类别间召回更加平衡，而不是增加总正确数。

9. **验证排名对留人测试的预测能力较弱。**30 个配置的 validation–test BA Pearson 相关系数为 0.50，Spearman 排名相关仅 0.318；`re2` 从验证第1降至测试第15，`rl3` 从验证第11升至测试第1。不能继续把单个 inner-val 最高值当作最终候选。

### 1.2 更新后的候选判断

- **Proto-only：当前没有胜者。**若后续必须保留一个机制对照，P2 Null `pn2_soft_null` 本身 test BA 87.93%，反而高于所有 Proto-only active；这说明 prototype 状态路径或训练随机性可能有用，但现有 ProtoLoss 梯度没有提供额外收益。
- **Rel-only 主候选：`rl3_k3_s125`。**P=3、diff-only、Top-K3、λrel=0.5、epoch 125 启动、constant、EMA=0.5；它同时有 exact P3 Null，对 BA/F1 的外层增益最清晰。
- **Rel-only 辅助候选：`rl2_k3_s125`。**test accuracy 88.31%（全组并列最高）、BA 88.88%、F1 88.55%；相对 P2 Null 的 BA −0.14 pp、F1 +1.81 pp、accuracy +5.97 pp，说明它偏向总正确率而不是类别均衡。
- **Proto+Rel 候选：`h2_emg_both_p1_k10`。**相对 P1 Null 在 test 上 BA +5.23 pp、F1 +4.24 pp、accuracy +4.94 pp，是三项同时改善最明显的组合；但仍只有一个 seed，而且 P1 ProtoLoss/RelLoss 的记录量级都很小。
- **不再优先：`ph2_l010` 与 `re2_k10_s50`。**两者验证表现很好，但 outer-test 没有支持其相对 Null 的优势。

这些仍是一个受试者、一个 seed 下的候选，不是项目最终最佳参数。由于本次同时查看了 30 个 MR 测试结果，MR 从现在起已不能继续作为无偏调参集。

## 2. 数据与结果完整性

### 2.1 已完成内容

| 阶段 | 计划数 | 完成微调数 | 状态 |
|---|---:|---:|---|
| Stage 1 | 2 | 2 | 完成 |
| Stage 2A | 11 | 11 | 完成 |
| Stage 3A | 6 | 6 | 完成 |
| Stage 4 | 11 | 11 | 完成 |
| 合计 | 30 | 30 | 核心筛选完成 |

随后已对上述30个最佳验证权重全部完成 fold_MR outer-test，测试结果完整度为30/30。

除 Direct FT 外的 29 个配置均有 200 epoch 对比预训练日志和最终权重。当前尚未运行 Stage 2B、3B、5、6、7，因此没有：

- λproto/λrel 局部加密结果；
- Proto+Rel 入选组合的正式 Stage 5 结果；
- M/J/MR/N 完整四人 LOSO 外层测试（当前仅有 MR 一折）；
- seed 2/3 稳定性确认。

### 2.2 文件完整性与目录异常

- 30 个 `summary.json` 均存在；29 个预训练 `args.json` 和 `debug_train_log.jsonl` 均存在。
- 共发现 580 个 prototype JSON 诊断、430 个 prototype state 文件。
- 共 139 个完整预训练 checkpoint，约 72.49 GB；整个结果目录约 119.27 GB。多出的 checkpoint 主要来自 RelLoss 启动边界保存。
- 下载/解压后，`prototype_diagnostics` 和一部分 datamap 被嵌套到 `pretrain/fold_MR/stage2a/rgb_mvit_pr_env_loso_20260810/...` 下。诊断内容本身完整，本次分析用递归定位处理；但未来汇总前建议修正下载结构，避免标准工具找不到文件。
- 下载后的原 `summary` 仍显示0个测试，是旧版汇总没有读取实际 `test_results.csv`。本次已用增强后的汇总程序重新生成30项测试排名；详细分析表位于 `results/rgb_mvit_pr_env_loso_20260810/analysis_20260812`。

## 3. 下游微调结果

所有实验均采用相同的 100 epoch 全参数微调、batch size 32、backbone LR 6e-5、head LR 2e-3，并按最佳 inner-val balanced accuracy 选择权重。

### 3.1 总体排名

| 排名 | 配置 | 类型 | BA | Macro-F1 | Accuracy | 相对 SupLoss BA |
|---:|---|---|---:|---:|---:|---:|
| 1 | `re2_k10_s50` | Rel P2 early K10 | **97.46%** | **96.83%** | **97.18%** | **+3.01 pp** |
| 2 | `h2_emg_both_p1_k10` | Proto+Rel P1 | 96.52% | 95.48% | 95.97% | +2.07 pp |
| 3 | `hn1_null_p1` | P1 Null | 96.22% | 96.15% | 96.77% | +1.77 pp |
| 4 | `si2_early_k3` | Proto+Rel P2 early | 95.96% | 95.35% | 95.97% | +1.51 pp |
| 5 | `sr3_sensor_rel` | Rel P3 sensor schedule | 95.82% | 95.68% | 96.37% | +1.37 pp |
| 6 | `rn3_k3_s125` | P3 Null-rel | 95.74% | 95.38% | 96.37% | +1.29 pp |
| 7 | `ph2_l010` | Proto P2 hard | 95.63% | 95.65% | 96.77% | +1.18 pp |
| 8 | `rn2_k3_s125` | P2 Null-rel | 95.61% | 94.93% | 95.97% | +1.16 pp |
| 9 | `re3_k10_s50` | Rel P3 early K10 | 95.37% | 95.19% | 95.56% | +0.92 pp |
| 10 | `si3_early_k3` | Proto+Rel P3 early | 95.28% | 95.22% | 95.97% | +0.83 pp |
| — | `s0_sup` | SupLoss-only | **94.45%** | **94.15%** | **95.56%** | 0 |
| — | `d0_k400_direct` | K400 Direct FT | 91.64% | 92.47% | 93.95% | −2.81 pp |

完整 30 项排名见 `analysis_20260812/classification_ranking.csv`。

### 3.2 SupLoss 是否有效

| 指标 | Direct FT | SupLoss | 差值 |
|---|---:|---:|---:|
| Balanced accuracy | 91.64% | 94.45% | **+2.81 pp** |
| Macro-F1 | 92.47% | 94.15% | **+1.68 pp** |
| Accuracy | 93.95% | 95.56% | **+1.61 pp** |

这是目前最清晰的正向结论。验证集共 248 个样本，Direct FT 约分对 233 个，SupLoss 约分对 237 个，即多分对 4 个样本。balanced accuracy 增益更大，说明提升偏向少数类，而不只是大类样本。

### 3.3 ProtoLoss 与 Null-proto 的配对比较

| Active | Null 对照 | ΔBA | ΔMacro-F1 | 判断 |
|---|---|---:|---:|---|
| `ps2_l010` | `pn2_soft_null` | 0.00 pp | +0.31 pp | 无 BA 改善 |
| `ps3_l010` | `pn3_soft_null` | +0.43 pp | +0.03 pp | 很弱 |
| `ph2_l010` | `pn2_soft_null` | **+1.61 pp** | **+1.67 pp** | Proto 中最值得确认 |
| `ph3_l010` | `pn3_soft_null` | +0.79 pp | +0.73 pp | 次选 |
| `pa2_l010` | `pn2_soft_null` | −0.03 pp | −0.38 pp | 无效 |
| `pa3_l010` | `pn3_soft_null` | +0.64 pp | +0.14 pp | 很弱 |
| `pt2_l100` | `pn2_soft_null` | +0.12 pp | +0.46 pp | 强权重无收益 |
| `pt3_l100` | `pn3_soft_null` | +0.17 pp | −0.24 pp | 强权重无收益 |

ProtoLoss 的证据不是“完全无用”，而是只有 hard/single P2 出现了值得复现的信号。soft/all 和 λ=1 的传感器强度设置没有稳定改善。

### 3.4 RelLoss 与 Null-rel 的配对比较

| Active | 对照 | ΔBA | ΔMacro-F1 | 说明 |
|---|---|---:|---:|---|
| `rl2_k3_s125` | `rn2_k3_s125` | −0.73 pp | −0.52 pp | 完全匹配的晚启动 K3，负向 |
| `rl3_k3_s125` | `rn3_k3_s125` | −0.56 pp | −0.69 pp | 完全匹配的晚启动 K3，负向 |
| `re2_k10_s50` | `rn2_k3_s125` | +1.85 pp | +1.90 pp | 对照并非完全匹配 |
| `re3_k10_s50` | `rn3_k3_s125` | −0.37 pp | −0.19 pp | P3 不支持 early K10 |

能够严格解释的结果是：**late-start、Top-K3、λrel=0.5 的 RelLoss 没有作用，反而略差于 matched Null。**`re2_k10_s50` 很强，但它同时改变了启动时间和 K，且没有 exact Null，所以还不能把增益归因于 RelLoss 梯度。

### 3.5 EMG/IMU 迁移配置

- `h2_emg_both_p1_k10` 比 SupLoss 高 2.07 pp BA，但其 P1 Null `hn1_null_p1` 已高出 SupLoss 1.77 pp；Active 相对 Null 仅 +0.30 pp，并且 macro-F1 反而 −0.68 pp。大部分表面收益不是两个损失能够单独解释的。
- `h1_imu_rel_p1` 相对 P1 Null 的 BA 为 −1.37 pp、macro-F1 为 −1.66 pp，说明 IMU 的 Rel-only 配置不能直接迁移到 RGB。
- `si2_early_k3` 达到 95.96% BA，但缺少完全一致的 P2 Proto+Rel Null；它应作为候选，不是证据。
- Stage 4 的 P2/P3 结果交替领先，没有一致的 prototype 数量规律。

### 3.6 改善来自哪些类别

`re2_k10_s50` 相对 SupLoss 的最佳 checkpoint 主要改变：

| 类别 | SupLoss recall | `re2` recall | 差值 | inner-val support |
|---|---:|---:|---:|---:|
| adjust | 60.0% | 90.0% | +30.0 pp | 10 |
| cap | 77.8% | 88.9% | +11.1 pp | 9 |
| remove | 90.9% | 100.0% | +9.1 pp | 11 |
| pull_out | 100.0% | 95.0% | −5.0 pp | 20 |
| 其余 11 类 | 基本不变 | 基本不变 | 约 0 | — |

因此 BA +3.01 pp 主要由少数几个小类样本决定；accuracy 实际只提高 1.61 pp，即约 4 个样本。该结果可能是真实的少数类改善，也可能是单次训练/最佳 epoch 选择波动，必须补 seed。

### 3.7 MR 外层测试结果（新增）

#### 3.7.1 测试协议与完整性

- 训练：M/J/N 开发人员中的 inner-train，共 990 个样本；
- 验证：M/J/N 中按 run 分组划出的 inner-val，共 248 个样本；
- 测试：完全未参与训练和选 checkpoint 的 MR，共 385 个样本、15 类；
- 测试权重：每个配置各自的 `best_val_balanced.pth`；
- 完整性：30/30 个配置均有 `test_results.csv`，每项 `num_loaded_keys=397`、missing/unexpected=0。

测试 CSV 中保留 overall metrics 和逐类 recall，但下载内容没有包含其引用的 per-sample CSV、详细 metrics JSON/混淆矩阵。因此本报告可以比较 accuracy、BA、F1 和逐类 recall，但不能进行配对 McNemar 检验、置信区间 bootstrap 或逐样本错误迁移分析。

#### 3.7.2 外层测试排名

| 排名 | 配置 | Test BA | Macro-F1 | Accuracy | ΔBA vs Sup |
|---:|---|---:|---:|---:|---:|
| 1 | `rl3_k3_s125` | **91.06%** | **88.78%** | 86.75% | **+7.67 pp** |
| 2 | `rn2_k3_s125` | 89.02% | 86.74% | 82.34% | +5.63 pp |
| 3 | `rl2_k3_s125` | 88.88% | 88.55% | **88.31%** | +5.49 pp |
| 4 | `h2_emg_both_p1_k10` | 88.59% | 86.11% | 83.64% | +5.20 pp |
| 5 | `si2_early_k3` | 88.39% | 86.83% | 85.71% | +5.01 pp |
| 6 | `sr3_sensor_rel` | 88.08% | 87.15% | 83.90% | +4.69 pp |
| 7 | `pn2_soft_null` | 87.93% | 86.76% | 87.27% | +4.54 pp |
| 8 | `pt2_l100` | 87.55% | 86.76% | 88.05% | +4.17 pp |
| 9 | `re3_k10_s50` | 87.35% | 86.08% | 84.16% | +3.96 pp |
| 10 | `si3_early_k3` | 87.14% | 84.26% | 81.82% | +3.75 pp |
| 15 | `re2_k10_s50` | 86.69% | 85.03% | 85.71% | +3.31 pp |
| 23 | `ph2_l010` | 84.55% | 84.13% | 84.94% | +1.16 pp |
| 26 | `s0_sup` | 83.39% | 82.01% | 82.34% | 0 |
| 30 | `d0_k400_direct` | 76.71% | 74.80% | 81.30% | −6.68 pp |

如果以 balanced accuracy/macro-F1 为主指标，`rl3` 最优；如果只看 overall accuracy，`rl2` 与 `ph3` 并列 88.31%。数据类别不均衡且小类是项目难点，因此继续以 BA/F1 为主更合理，但必须同时报告 accuracy，避免只靠小类波动形成“最高结果”。

#### 3.7.3 SupLoss 的外层泛化

| 指标 | Direct FT | SupLoss | 差值 |
|---|---:|---:|---:|
| BA | 76.71% | 83.39% | **+6.68 pp** |
| Macro-F1 | 74.80% | 82.01% | **+7.21 pp** |
| Accuracy | 81.30%（313/385） | 82.34%（317/385） | +1.04 pp（+4 样本） |

SupLoss 对少数类非常有帮助：`press` +41.7 pp、`tear` +35.3 pp、`adjust` +30 pp、`close` +25 pp；但 `open` −31.2 pp、`insert` −19.2 pp。它不是全面提升，而是把表示从大类偏置转向更均衡的类别召回。这也解释了 accuracy 只改善 4 个样本，而 BA/F1 改善很大。

#### 3.7.4 ProtoLoss 的外层 Active–Null 比较

| Active | 对应 Null | ΔBA | ΔF1 | ΔAccuracy |
|---|---|---:|---:|---:|
| `ps2_l010` | `pn2_soft_null` | −1.56 pp | −2.28 pp | −4.16 pp |
| `ps3_l010` | `pn3_soft_null` | +0.09 pp | +0.65 pp | +0.26 pp |
| `ph2_l010` | `pn2_soft_null` | **−3.38 pp** | −2.63 pp | −2.34 pp |
| `ph3_l010` | `pn3_soft_null` | −0.99 pp | +0.20 pp | +4.68 pp |
| `pa2_l010` | `pn2_soft_null` | −0.80 pp | −3.33 pp | −7.53 pp |
| `pa3_l010` | `pn3_soft_null` | +0.03 pp | +0.23 pp | +1.56 pp |
| `pt2_l100` | `pn2_soft_null` | −0.37 pp | 0.00 pp | +0.78 pp |
| `pt3_l100` | `pn3_soft_null` | −1.18 pp | −2.87 pp | −3.12 pp |

外层测试对旧版 ProtoLoss 的结论比 inner-val 更明确：**没有一个 active 配置相对 Null 同时在 BA、F1、accuracy 上产生有意义的改善。**P2 Null 自身达到 87.93% BA，说明“启用 prototype 状态/重聚类路径但不给梯度”与普通 SupLoss 产生了很不同的随机训练轨迹；active 必须超过这个强 Null 才能归因给 ProtoLoss，目前没有做到。

#### 3.7.5 RelLoss 的外层 Active–Null 比较

| Active | 对照 | ΔBA | ΔF1 | ΔAccuracy | 解释 |
|---|---|---:|---:|---:|---|
| `rl2_k3_s125` | exact `rn2_k3_s125` | −0.14 pp | +1.81 pp | **+5.97 pp** | 更偏 overall accuracy |
| `rl3_k3_s125` | exact `rn3_k3_s125` | **+5.11 pp** | **+2.42 pp** | 0.00 pp | 更偏类别均衡 |
| `re2_k10_s50` | 非匹配 `rn2` | −2.32 pp | −1.71 pp | +3.38 pp | 不能作因果对照 |
| `re3_k10_s50` | 非匹配 `rn3` | +1.40 pp | −0.29 pp | −2.60 pp | 指标混合 |

`rl3` 与 `rn3` 的总正确数相同（均约 334/385），但 `rl3` 把正确预测重新分配到了小类：`close` +50 pp、`press` +16.7 pp、`tear` +11.8 pp、`adjust` +10 pp；代价是 `cap/insert/wrap/cut` 降低。因而 BA +5.11 pp、F1 +2.42 pp，却没有 accuracy 增加。

这是一条有意义的结果：RelLoss 的设计目标正是改变类间关系，它可能改善了小类决策边界。但它仍需要其它受试者和 seed 确认，不能把“相同总正确数但更均衡”包装成绝对性能全面提高。

#### 3.7.6 Proto+Rel 组合

`h2_emg_both_p1_k10` 相对 P1 Null：

- BA：88.59% vs 83.37%，+5.23 pp；
- F1：86.11% vs 81.86%，+4.24 pp；
- accuracy：83.64% vs 78.70%，+4.94 pp，即约多分对 19 个样本。

三项均提高，是当前最完整的组合信号。逐类上主要改善 `close` +25 pp、`cap` +16.7 pp、`tear` +11.8 pp、`insert` +10.3 pp，同时 `open` −12.5 pp。由于 P1 不存在类内多 prototype，ProtoLoss 实际退化为类中心约束；这个结果更可能来自“类中心 Proto + Rel”的联合正则，而不是多 prototype 建模成功。

`si2_early_k3` test BA 88.39%，明显优于 `si2_late_k3` 的 81.73%，P3 也呈现 early 87.14% > late 82.66%。这支持组合损失应较早介入，但这些配置缺少 exact Null，不能独立拆分 Proto 与 Rel 的贡献。

#### 3.7.7 验证–测试排名稳定性

- Pearson（validation BA vs test BA）：0.500；
- Spearman 排名相关：0.318；
- 平均 validation–test BA gap：8.88 pp；中位数 8.27 pp；
- `re2`：验证第1 → 测试第15；
- `ph2`：验证第7 → 测试第23；
- `rl3`：验证第11 → 测试第1；
- `pn2`：验证第25 → 测试第7。

inner-val 来源仍是训练人员 M/J/N，而 outer-test 是新人员 MR；低排名相关说明这里最大的分布变化是**跨人泛化**。仅在开发人员内部筛选不能可靠预测新人员表现。后续方法选择应使用多折 LOSO，而不是继续针对 MR test 调参。

#### 3.7.8 多重比较与统计限制

本次是30个配置第一次接触MR，单个预先指定模型的MR结果原本是无偏测试；但现在同时查看30个结果并选择最高者，会产生 winner's curse。特别是 `close` 只有4个样本，一个样本就改变25 pp recall。`adjust/cap/press/tear` 也只有10–18个样本，BA很容易因少数预测翻转而变化。

因此最严谨的表述是：`rl3` 和 `h2` 在 held-out MR 上出现强候选信号；并非已经证明它们优于 SupLoss。由于逐样本预测文件未下载，目前也无法进行 paired bootstrap/McNemar；即使补做统计，单一受试者仍不能替代多折 LOSO。

## 4. 对比预训练损失分析

### 4.1 训练稳定性

- 29 个预训练日志中没有发现 NaN/Inf。
- 最终 queue 的 1088 个槽位均有效；每个 anchor 的 queue 同类正样本均值通常约 100，但随类别频率变化很大。
- 最终 SupLoss 均约为 4.56–4.59，各配置相近；新增损失没有引起主损失爆炸。
- batch size 32、queue 1088、LR 6e-5 在这些实验中数值稳定。

### 4.2 ProtoLoss 的实际量级

| 配置类型 | P | λproto | 最终 ProtoLoss | 加权后占 SupLoss |
|---|---:|---:|---:|---:|
| hard `ph2` | 2 | 0.1 | 0.0092 | **0.02%** |
| hard `ph3` | 3 | 0.1 | 0.0169 | **0.04%** |
| soft/all P2 | 2 | 0.1 | 约 0.71 | 约 1.55% |
| soft/all P3 | 3 | 0.1 | 约 1.11 | 约 2.43% |
| all P2 strong | 2 | 1.0 | 约 0.70 | 约 15.3% |
| all P3 strong | 3 | 1.0 | 约 1.10 | 约 24.2% |

关键现象：

1. 增大 λproto 到 1 后，加权 ProtoLoss 已达到 SupLoss 的 15%–24%，但下游没有提高，说明问题不是简单的“ProtoLoss 权重不够”。
2. hard/single 的最终损失几乎降到零，说明该目标很快被满足/饱和。它可能在早期提供了少量有用约束，但后期基本不再贡献。
3. soft 与 all 的损失量级、prototype 几何和性能都很接近；在同类 prototype 高度相似的条件下，两种目标实际提供的差异有限。
4. 当前日志记录了总梯度范数，但没有分别记录 `||∇Lsup||`、`||∇(λpLp)||` 和二者的梯度夹角。因此“损失值占比”不能替代“梯度占比”；后续必须补这项诊断。

### 4.3 RelLoss 的实际量级

- `re2/re3/rl2/rl3` 的训练期平均加权 RelLoss 约 0.0009–0.0012，只相当于 SupLoss 的 **0.02%–0.03%**。
- Stage 4 的大多数 Rel 配置平均约为 SupLoss 的 0.02%–0.09%；即使训练末期，最强者通常也低于约 0.2%。
- constant 配置在启动后的第一个 epoch 约有 0.002–0.004 的加权贡献，随后快速减小。
- cosine 配置在启动 epoch 和下一 epoch 基本为 0，启动后 10 epoch 仍非常小，意味着 RelLoss 在很长一段时间中几乎没有梯度影响。

这说明当前 RelLoss 的主要不足是**尺度未校准**。尽管 raw λrel 设为 0.5 或 1.0，看起来不小，但实际加权项与 SupLoss 相比非常弱。

## 5. Prototype 健康度与环境分析

### 5.1 Assignment 与 dead prototype

- P2/P3 的最终 strict dead prototype 均为 0，说明 K-means 没有完全空槽。
- P2 最终通常有 0–2 个 near-dead；P3 通常有 2–5 个 near-dead，P3 对小样本类更脆弱。
- 最终归一化 assignment entropy 大多在 0.80–0.94，整体没有严重单槽占据。
- 例如 `ps3_l010` 最终 45 个 active prototypes、0 strict dead、3 near-dead、平均 assignment entropy 0.901。

所以“prototype 完全死亡”不是本轮失败的主要原因。

### 5.2 Prototype 高度重合

更严重的问题是同类 prototype 的余弦相似度：

- Proto-only P2/P3：多数为 0.992–0.999；
- Rel-only：多数仍为 0.992–0.998；
- Stage 4 Proto+Rel：多数为 0.996–0.999。

以 `ps3_l010` 为例，同类平均余弦相似度从 epoch 60 的 0.9879 上升到 epoch 200 的 0.9949。`ph2_l010` 从 0.9918 上升到 0.9967。即使 assignment 看起来均衡，prototype 方向仍越来越接近。

这解释了为何 P=2/P=3 没有稳定差异：当前模型更多是在几乎相同的类方向附近做细小 K-means 切分，而不是学习清晰的类内动作/环境子模式。

### 5.3 RelLoss 是否改变了几何

是的。P2 Null `rn2` 的最终最近异类 prototype 平均余弦相似度为 0.5281：

- `re2_k10_s50` 降至 0.4772；
- `rl2_k3_s125` 降至 0.4702。

P3 Null `rn3` 为 0.5292：

- `re3` 降至 0.5019；
- `rl3` 降至 0.4810。

因此 RelLoss 不是“没有执行”；它确实把异类 prototype 推得更开。但是 `rl2/rl3` 几何分离更强却分类更差，说明“把 prototype 拉开”本身不保证更好的动作判别，当前关系目标可能和下游决策边界不一致。

### 5.4 P3 是否学习了三种光照

| 代表配置 | P | 最终 Lighting NMI | ARI | Purity |
|---|---:|---:|---:|---:|
| `pn3_soft_null` | 3 | 0.083 | 0.034 | 0.468 |
| `ps3_l010` | 3 | 0.101 | 0.045 | 0.475 |
| `ph3_l010` | 3 | 0.088 | 0.039 | 0.472 |
| `pt3_l100` | 3 | 0.081 | 0.027 | 0.459 |
| `re3_k10_s50` | 3 | 0.067 | 0.008 | 0.448 |
| `rn3_k3_s125` | 3 | 0.099 | 0.043 | 0.478 |

这些 NMI/ARI 都很低，而且 active 与 Null 相近。Purity 随 prototype 数增加会自然变大，不能单独作为光照对应证据。`si3_early_k3` 曾出现 max NMI 0.23，但最终降到 0.083，是短暂且不稳定的结构。

结论：**RGB 的类内差异没有自然地按三种光照分成三个稳定 prototype。**可能是光照增强已削弱环境信息，也可能真正的类内因素是受试者、动作阶段、物体状态或视角，而不是光照。

### 5.5 当前缺少的诊断

现有文件可以分析 assignment 数量、dead/near-dead、assignment entropy、prototype 相似度、loss 数值、总梯度和光照关联，但仍缺少：

- soft responsibility entropy（当前 entropy 是 assignment 数量熵，不是 soft responsibility 熵）；
- SupLoss、ProtoLoss、RelLoss 各自的梯度范数；
- 新损失与 SupLoss 的梯度余弦相似度/冲突率；
- 每个 prototype 对受试者、动作阶段、run、物体状态的关联；
- 固定样本跨重聚类 epoch 的 assignment 稳定性。

## 6. 微调阶段的过拟合与评估风险

- 30 个实验的最佳 BA epoch 中位数为 28，平均为 31；24/30 在前 50 epoch 达到最佳，29/30 在前 75 epoch 达到最佳。
- 多个配置从最佳到 epoch 100 明显下降：`ps3` −5.65 pp、`pa2` −5.06 pp、`re2` −4.55 pp、`re3` −4.19 pp。
- 最终训练 BA 几乎全部接近 100%，说明后半程主要是训练集拟合，而不是持续改善泛化。

当前保存最佳 checkpoint 的做法是必要的，但在 30 个配置 × 100 个 epoch 中挑最高值，会放大验证集偶然波动。后续建议：

1. 保留最佳 checkpoint，同时报告固定 epoch（如 25/50）或最后 5 次验证均值；
2. 加 early stopping（patience 15–20），最大 75 epoch；
3. 最终结论以四折 LOSO mean±std 和多 seed 为准，不再根据外层测试回调超参数。

## 7. ProtoLoss 是否发挥作用

更新后的综合判断：**ProtoLoss 正常执行，但 MR 外层测试不支持旧版 ProtoLoss 带来额外泛化收益。**

支持“机制在工作”的证据：

- active 配置有非零损失，且不同 mode/λ 产生不同损失尺度；
- 没有数值崩溃或 strict dead prototype；
- prototype assignment 和几何随配置变化。

反对“改善分类”的关键证据：

- inner-val 最好的 hard P2 在 test 上相对 P2 Null 变为 BA −3.38 pp、F1 −2.63 pp；
- 8 组 Active–Null Proto 比较中，没有一组在 test BA/F1/accuracy 三项同时得到有意义改善；
- λ=1、加权项达到主损失 15%–24% 后仍没有改善，排除“只因权重太小”的简单解释；
- 同类 prototype 几乎重合，没有形成有意义的多中心结构；
- P3 与光照关系很弱，且 P3 没有一致优于 P2；
- P2 Null 自身显著超过 SupLoss，说明训练路径/随机轨迹影响很大，active 必须超过强 Null 才能归因给损失。

因此，就当前旧版定义而言，应把 ProtoLoss 结论写成“未获得支持”，而不是“弱有效”。这不否定 prototype 主思路，但说明需要先解决同类 prototype 重合、assignment 语义和梯度目标问题。

## 8. RelLoss 是否发挥作用

更新后的综合判断：**RelLoss 明显改变了 prototype 几何，并在 held-out MR 上出现了比 ProtoLoss 更强的泛化信号，但尚未完成跨人/多 seed 证明。**

支持 RelLoss 的证据：

- active Rel 显著降低最近异类 prototype 的余弦相似度；
- exact matched P3 late-K3 active `rl3` 相对 `rn3`：test BA +5.11 pp、F1 +2.42 pp；
- `rl3` 达到全组最高 test BA/F1，而且它并不是 inner-val 第一名，降低了“只复现验证尖峰”的可能；
- exact P2 `rl2` 相对 `rn2` 虽 BA −0.14 pp，但 F1 +1.81 pp、accuracy +5.97 pp；P2/P3 都显示 Rel 会改变错误在类别之间的分配；
- P1 Rel-only `h1` 在 test 上相对 P1 Null 的 BA +2.43 pp，与 inner-val 的负向结果不同。

仍需谨慎的证据：

- `rl3` 与 `rn3` accuracy 完全相同，BA 改善来自小类 recall 重分配，并非更多总正确样本；
- `re2` 验证第1却测试第15，early K10 P2 并不稳定；
- RelLoss 的加权量长期只有 SupLoss 的约 0.02%–0.09%，其效果可能高度依赖训练轨迹；
- 几何分离程度与分类性能不单调；
- 只有一个 outer subject 和一个 seed，同时比较30个模型存在多重比较偏差。

因此当前最合理的表述是：**P3 late-K3 RelLoss 是最值得进一步确认的旧版损失候选；它在 MR 上改善类别均衡，但还不能称为稳定提高整体识别性能。**

## 9. 未来最优先的确认实验（本轮暂不执行）

外层测试改变了候选顺序。若以后继续，建议只补一个小而严格的因果确认组，所有配置用 seed 2、3，并重新完成预训练和微调：

| 配置 | 目的 |
|---|---|
| `S0` SupLoss-only | 同 seed 主基线 |
| `rn3_k3_s125` | P3 late-K3 exact Null |
| `rl3_k3_s125` | 复现 test BA/F1 的主要 Rel 信号 |
| `hn1_null_p1` | P1 Proto+Rel 路径 Null |
| `h2_emg_both_p1_k10` | 复现三指标共同改善的组合信号 |
| `pn2_soft_null` | 判断强 P2 Null 是否可复现，量化纯训练路径波动 |
| `rn2/rl2`（预算允许） | 确认 accuracy 导向的 P2 Rel 信号 |

前六项为 6 配置 × 2 新 seeds = 12 次；加已有 seed 1 后可形成三 seed 判断。旧版 Proto-only active 暂不列为必跑，因为 outer-test 没有任何 Active–Null 支持。只有 active 在多数 seed 中同时超过 SupLoss 和 exact Null，才值得进入多折 LOSO。

## 10. 最可能成功的损失改进方向

### 10.1 ProtoLoss：先解决多 prototype 重合

最优先改动不是继续扫 λ，而是改变 prototype 的结构：

1. **类中心 + 残差子 prototype**：`p(c,k)=normalize(class_center(c)+residual(c,k))`。分类方向由类中心保证，残差只描述类内差异。
2. **同类 prototype 多样性约束**：对 residual 而不是完整 prototype 施加正交/最小角度约束，避免 0.997 的方向重合，同时防止破坏类别中心。
3. **平衡 assignment**：使用 Sinkhorn/容量约束或最小使用率，减少 P3 near-dead；但不要强迫每类三槽完全等量。
4. **按类别自适应 P**：`close/adjust/press` 等小类优先 P1，大样本且确有多峰结构的类别用 P2；不要用“有三种光照”直接推导 P3。
5. **不再把 hard P2 当作旧版胜者**：它在 inner-val 的信号没有迁移到 MR。若保留 hard assignment，应改为带 margin/困难样本版本，并首先证明能超过同路径 Null。
6. **环境目标改为可选诊断而非硬绑定**：如果目标是光照不变性，更合理的是对 lighting 做 adversarial invariance/环境一致性，而不是要求三个 prototype 分别等于三个光照。

### 10.2 RelLoss：校准梯度而不是继续增大 raw λ

1. **记录并控制梯度比例**：每 10 epoch 计算 `g_sup`、`g_rel` 和 cosine；将 `||λrel g_rel|| / ||g_sup||` 控制在约 1%–5% 的起始范围，再根据验证调整。
2. **EMA 自适应缩放**：`λeff = clip(r × EMA(|Lsup|)/EMA(|Lrel|))`，避免 raw loss 只有 0.001 时 λ=1 仍几乎不起作用。
3. **短 warmup 后恒定**：启动后用 5–10 epoch 线性 ramp 到目标比例，然后保持，而不是让 cosine 在较长时间接近 0。
4. **以 P3 late-K3 作为确认起点**：outer-test 支持 `rl3` 而不支持验证集最高的 early-K10 P2；优先复现 P3 late-K3，再用 exact matched design 单独拆分 P、K 和启动时间。
5. **关系只约束真正混淆的类别**：根据验证混淆矩阵或 queue 中最近异类动态选择 pair；`adjust/cap/remove` 是当前最需要改善的类别。
6. **监测梯度冲突**：若 `cos(g_sup,g_rel)<0`，可对 Rel 梯度做投影或仅在不冲突时更新，防止几何分离损害分类结构。

### 10.3 SupLoss 主干也需要处理类别不平衡

当前使用 `sampler_type=none`，queue 中大类和小类的正样本数差异明显。建议新增一个不改 Proto/Rel 的控制实验：

- batch size 32；
- balanced batch 采用 8 类 × 4 样本，或使用 class-aware weighted sampler；
- queue 仍为 1088；
- SupLoss 使用所有同类正样本，但对类别/anchor 做等权。

如果 balanced SupLoss 已经超过当前新增损失，说明主要瓶颈是类别不平衡，而不是 prototype 数量。

## 11. 推荐决策

### 当前不建议

- 不建议直接宣布 P3 是三光照的最佳设置；
- 不建议把 `re2` 的 inner-val 97.46% 当作最终最佳性能；它在 MR test 仅排第15；
- 不建议直接运行全部 Stage 2B/3B 网格；
- 不建议把 λrel 从 1 粗暴增加到 5/10，而不看梯度比例；
- 不建议直接把 `rl3` 的 91.06% 写成最终性能；这是从30个模型中选出的单人单seed最高值；
- 不建议继续使用 MR test 调参，否则会失去当前唯一的外层测试证据。

### 当前建议

1. 本轮先保留结果，不根据 MR 继续搜索；
2. 若未来继续，按第 9 节复现 `rl3/rn3` 与 `h2/hn1`，同时补 SupLoss 同 seed；
3. 同时补独立梯度诊断、soft responsibility entropy 和逐样本预测保存；
4. Proto-only 旧版不再进入组合，除非改进后的版本先超过 exact Null；
5. Rel 改进以 P3 late-K3 为起点，但用梯度比例控制并监测与 SupLoss 的冲突；
6. 最终性能只能来自冻结配置后的其它受试者折与多 seed，不能再次选择 MR 最优。

## 12. 最终回答

- **SupLoss：有用。**inner-val 相对 Direct BA +2.81 pp，held-out MR test BA +6.68 pp、F1 +7.21 pp；但 accuracy 只 +1.04 pp，主要改善少数类均衡。
- **ProtoLoss：旧版 active 没有得到外层测试支持。**hard P2 的验证优势在 test 反转；所有 Proto-only Active–Null 比较均未出现可信的三指标共同改善。多 prototype 高度重合仍是核心不足。
- **RelLoss：有比 ProtoLoss 更强的候选证据。**`rl3_k3_s125` 相对 exact P3 Null 的 test BA +5.11 pp、F1 +2.42 pp，但 accuracy 不变；它改善的是类别均衡，仍需跨人/多 seed 复现。
- **Proto+Rel：`h2_emg_both_p1_k10` 是当前三指标最一致的组合信号。**相对 P1 Null 的 test BA/F1/accuracy 分别 +5.23/+4.24/+4.94 pp，但 P1 本质上是类中心而非多 prototype，且只有一个 seed。
- **更新后的暂定候选**：Rel=`rl3_k3_s125`；组合=`h2_emg_both_p1_k10`；Proto-only=无。它们都是后续确认对象，不是可以直接写入论文的最终最佳配置。

## 13. 可复核分析文件

本报告由 `tools/analyze_completed_screen.py` 从原始结果重新抽取。生成表包括：

- `analysis_20260812/classification_ranking.csv`
- `analysis_20260812/selected_per_class.csv`
- `analysis_20260812/pretrain_dynamics.csv`
- `analysis_20260812/prototype_diagnostics.csv`
- `analysis_20260812/prototype_lighting.csv`
- `analysis_20260812/outer_test_analysis.csv`
- `analysis_20260812/outer_test_per_class.csv`
- `analysis_20260812/outer_test_meta.json`
- `analysis_20260812/analysis_tables.json`

这些表保留了本报告中的验证/测试排名、损失量级、启动边界、prototype 健康度、光照关联和逐类测试证据。
