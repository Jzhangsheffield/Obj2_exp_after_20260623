# K400 MViT-v2-S 上旧版 ProtoLoss / RelLoss 详细分析报告

生成日期：2026-08-12；最近更新：2026-08-14
结果目录：`results/rgb_mvit_pr_env_loso_20260810`、`results/rgb_mvit_pr_unified_followup_20260813`
分析范围：Stage 1、Stage 2A、Stage 3A、Stage 4；M/J/N 开发集上的 inner-val 筛选、完全留出的 MR outer-test（385 个样本），以及统一协议 U1a 的 N final-refit 15 类最小确认（416 个样本）

## 1. 结论摘要

### 1.1 可以确认的结论

1. **MViT-v2-S 上的 SupLoss 对比预训练是有用的。**K400 直接全参数微调的最佳验证 balanced accuracy（BA）为 91.64%，SupLoss-only 为 94.45%，提高 **2.81 个百分点**；accuracy 从 93.95% 提高到 95.56%，macro-F1 从 92.47% 提高到 94.15%。这说明更换为 K400 MViT-v2-S 后，对比学习已经不再像早期 ResNet3D-18 那样无法形成可用动作表示。

2. **旧版 ProtoLoss 与 RelLoss 都确实参与了计算，并能改变表示几何；训练过程没有数值崩溃。**29 个预训练实验的日志均未发现 NaN/Inf，所有 P2/P3 配置在最终 epoch 均无 strict dead prototype；RelLoss 配置明显降低了最近异类 prototype 的余弦相似度。

3. **但是，目前不能得出“ProtoLoss 或 RelLoss 已稳定提高分类性能”的最终结论。**历史筛选同时比较了28个新增损失配置，只有 seed 1 和 MR 一折；新增 U1a 又只增加了 seed 1 的 N 一折，而且采用不同的 final-refit 固定 epoch 协议。两个受试者上的关键 RelLoss 效应方向相反，最高值仍受到小类样本数、多重比较与训练协议差异影响。

4. **三种光照对应三个 prototype 的假设没有得到支持。**P3 assignment 与 `left/normal/right` 的最终平均 NMI 约为 0.05–0.10、ARI 多数低于 0.05，Null 配置也达到相近数值；P3 没有一致优于 P2。因此后续不应因为存在三种光照就固定采用 P=3。

5. **prototype 的主要问题不是严格死亡，而是同类 prototype 方向高度重合。**P2/P3 最终同类 prototype 平均余弦相似度通常为 0.992–0.999。K-means 能把样本分给不同槽位，但 prototype 在归一化表示空间中几乎同方向，说明当前损失没有形成明确、稳定、具有语义的类内子模式。

6. **MR 外层测试进一步确认 SupLoss 有效。**Direct FT 的 test BA/macro-F1/accuracy 为 76.71%/74.80%/81.30%，SupLoss 为 83.39%/82.01%/82.34%，分别提高 **6.68/7.21/1.04 pp**。增益主要来自 `press/tear/adjust/close` 等少数类，因此 BA/F1 的改善显著大于 overall accuracy。

7. **外层测试否定了“hard P2 ProtoLoss 是当前最佳 Proto 设置”的初步判断。**`ph2_l010` 虽在 inner-val 相对 P2 Null 高 1.61 pp，但在 MR test 反而低 **3.38 pp BA、2.63 pp F1、2.34 pp accuracy**。本轮没有任何 Proto-only active 配置相对对应 Null 同时呈现可信、较大的三指标增益。

8. **在历史 MR 筛选中，RelLoss 的首选信号曾变为 `rl3_k3_s125`，而不是验证集最高的 `re2_k10_s50`。**`rl3` 在 MR test 达到全组最高的 **91.06% BA、88.78% macro-F1**；相对完全匹配的 P3 Null `rn3`，BA +5.11 pp、F1 +2.42 pp，但 accuracy 相同。该历史单折信号随后在 U1a N 折反转，不能再作为当前首选结论。

9. **验证排名对留人测试的预测能力较弱。**30 个配置的 validation–test BA Pearson 相关系数为 0.50，Spearman 排名相关仅 0.318；`re2` 从验证第1降至测试第15，`rl3` 从验证第11升至测试第1。不能继续把单个 inner-val 最高值当作最终候选。

10. **U1a 15 类最小确认已按锁定协议完整完成 5/5 次。**5 个配置均使用 M/J/MR 的全部 1207 个 t15 样本训练，以 N 的 416 个样本测试；无 validation，微调固定 50 epoch，并只报告 epoch 50。清单审计为 `expected_missing=[]`，训练/测试 split 哈希在5个配置间一致。

11. **`h11_p1_k10` 对严格联合 Null `h00_p1_k10` 的 N 折增益为正，但相对普通 SupLoss 仅近似持平。**`h11-h00` 的 BA/F1/accuracy 为 **+2.74/+2.73/+2.16 pp**（+9 个正确样本）；但 `h11-s0` 只有 **+0.13/+0.99/+0.72 pp**（+3 个正确样本）。这支持“联合损失改变了训练结果”，但不支持“已经带来有实际幅度的稳定增益”。

12. **MR 上的 `rl3` 强信号没有在 N 上复现，且方向反转。**N 折中 `rl3-rn3` 的 BA/F1/accuracy 为 **−2.53/−2.82/−0.72 pp**（少3个正确样本）；相对 `s0` 为 **−4.04/−4.16/−2.16 pp**。因此不能再把 P3 late-K3 RelLoss 写成当前首选性能候选，只能保留为会显著改变 prototype 几何、但跨人效应不稳定的机制候选。

13. **N 名义上是 t15，但测试集中没有 `close`。**所有 overall accuracy 使用416个样本，BA和macro-F1只在实际出现的14类上计算；自动汇总中的“±0.00”来自每配置仅1次运行，不代表零方差或稳定性。

14. **逐样本配对检验没有为正向增益提供显著证据。**`h11-h00` 的 discordant correct 为21比12，exact McNemar `p=0.163`；`h11-s0` 为18比15，`p=0.728`。`rl3-rn3` 为17比20，`p=0.743`。10,000次按真实类别分层的 paired bootstrap 中，前三组比较的 BA/F1/accuracy 95%区间均跨0；`rl3-s0` 的 BA/F1 区间则完全低于0。该检验只反映固定 N 测试样本上的预测差异，不包含训练随机性或受试者方差。

### 1.2 更新后的候选判断

- **Proto-only：当前没有胜者。**若后续必须保留一个机制对照，P2 Null `pn2_soft_null` 本身 MR test BA 87.93%，反而高于所有 Proto-only active；这说明 prototype 状态路径或训练随机性可能有用，但现有 ProtoLoss 梯度没有提供额外收益。
- **Rel-only：`rl3_k3_s125` 从“主性能候选”降级为“跨人异质性/机制候选”。**它在 MR 上相对 exact `rn3` 为 BA/F1 +5.11/+2.42 pp，但在 N 上变为 −2.53/−2.82 pp，且两个受试者都只有 seed 1。`rl2` 尚未进入本次最小确认，其 MR accuracy 信号仍未复现。
- **Proto+Rel：`h11_p1_k10`（原 `h2_emg_both_p1_k10`）是目前唯一在 N 上同时超过 exact Null 与 `s0` 的 active 配置，但对 `s0` 的幅度很小。**N 上相对严格、完全匹配的 Top-K10 Null `h00_p1_k10` 三项均提高；相对 `s0` 仅 BA +0.13 pp、F1 +0.99 pp、accuracy +0.72 pp。它是弱确认信号，不是已确认胜者。
- **不再优先：`ph2_l010` 与 `re2_k10_s50`。**两者验证表现很好，但 outer-test 没有支持其相对 Null 的优势。

当前证据来自 MR 与 N 两个不同受试者，但每个比较仍只有 seed 1，而且两折的训练协议不同（历史 MR 使用 inner-val 最佳权重，U1a N 使用 final-refit 固定 epoch 50），不能直接求两折均值。MR 已在30配置筛选中被使用，N 也已在 U1a 中被查看；二者都不能再被描述为后续调参的全新独立测试集。

## 2. 数据与结果完整性

### 2.1 已完成内容

| 阶段 | 计划数 | 完成微调数 | 状态 |
|---|---:|---:|---|
| Stage 1 | 2 | 2 | 完成 |
| Stage 2A | 11 | 11 | 完成 |
| Stage 3A | 6 | 6 | 完成 |
| Stage 4 | 11 | 11 | 完成 |
| 历史筛选合计 | 30 | 30 | 核心筛选完成 |
| U1a t15 final-refit N/s1 | 5 | 5 | 最小确认完成 |

随后已对上述30个最佳验证权重全部完成 fold_MR outer-test，测试结果完整度为30/30。

除 Direct FT 外的历史29个配置均有200 epoch对比预训练日志和最终权重；U1a 的5个配置也均有200 epoch预训练日志、epoch 200预训练权重、50 epoch无验证微调结果和固定 epoch 50测试指标。当前尚未运行 Stage 2B、3B、5、6、7，也尚未完成 U1b，因此没有：

- λproto/λrel 局部加密结果；
- Proto+Rel 入选组合的正式 Stage 5 结果；
- 使用同一 final-refit 固定 epoch 协议的 M/J/MR/N 完整四人 LOSO；
- U1a 候选的跨 seed 稳定性确认。

### 2.2 文件完整性与目录异常

- 30 个 `summary.json` 均存在；29 个预训练 `args.json` 和 `debug_train_log.jsonl` 均存在。
- 共发现 580 个 prototype JSON 诊断、430 个 prototype state 文件。
- 共 139 个完整预训练 checkpoint，约 72.49 GB；整个结果目录约 119.27 GB。多出的 checkpoint 主要来自 RelLoss 启动边界保存。
- 下载/解压后，`prototype_diagnostics` 和一部分 datamap 被嵌套到 `pretrain/fold_MR/stage2a/rgb_mvit_pr_env_loso_20260810/...` 下。诊断内容本身完整，本次分析用递归定位处理；但未来汇总前建议修正下载结构，避免标准工具找不到文件。
- 下载后的原 `summary` 仍显示0个测试，是旧版汇总没有读取实际 `test_results.csv`。本次已用增强后的汇总程序重新生成30项测试排名；详细分析表位于 `results/rgb_mvit_pr_env_loso_20260810/analysis_20260812`。
- U1a 的 `analysis_unified/analysis_audit.json` 报告5次运行、无预期缺失；`manifests/u1a_confirm15_min.meta.json` 的清单哈希为 `06d6f457...0279f0`。5个 `last_test_metrics.json` 均包含 overall、逐类指标和混淆矩阵。
- U1a 每个配置均有416行 `predictions.csv` 和 `last_per_sample_test.csv`；5份预测按 `original_key` 一一对齐，真实标签完全一致。因此可以执行 paired bootstrap 与 exact McNemar。当前自动 `UNIFIED_STATISTICAL_REPORT.md` 尚未展示这些配对统计，本报告已从逐样本文件补算。

## 3. 下游微调结果

以下第3.1–3.7节的历史筛选实验均采用相同的100 epoch全参数微调、batch size 32、backbone LR 6e-5、head LR 2e-3，并按最佳 inner-val BA 选择权重；第3.8节 U1a 则采用新的50 epoch final-refit固定权重协议，不能与历史最佳权重结果混作同一统计总体。

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

因此，仅就这一轮历史 MR 筛选，最严谨的表述是：`rl3` 和 `h2` 在 held-out MR 上出现强候选信号；并非已经证明它们优于 SupLoss。由于逐样本预测文件未下载，目前也无法进行 paired bootstrap/McNemar；即使补做统计，单一受试者仍不能替代多折 LOSO。后续 U1a N 结果见第3.8节，其中 `rl3` 信号已反转。

### 3.8 U1a：N final-refit 15 类最小确认（2026-08-14 更新）

#### 3.8.1 协议与完整性

- 任务：`t15`，排除 `take/put`，K400 MViT-v2-S，`a0` mild augmentation，natural sampling；
- 测试对象：N；训练对象：M/J/MR；训练1207个样本，测试416个样本，sample overlap=0；
- 预训练：200 epoch；微调：50 epoch、无 validation、milestones `[25,37]`；统一汇总固定读取 epoch 50；
- 配置：`s0`、`rn3`、`rl3`、`h00_p1_k10`、`h11_p1_k10`，均为 seed 1；
- 完整性：5/5次运行齐全，`expected_missing=[]`；所有配置使用相同的 train/test manifest 哈希；
- 类别注意：N 没有 `close` 样本，因此 `num_present_classes=14`，BA/macro-F1只对其余14类求宏平均。

这里的 `h00_p1_k10` 是以 `h11` 为模板、仅把 `lambda_proto/lambda_rel` 设为0的 strict joint Null；它使用 Top-K10，不能用历史 Top-K3 `hn1_null_p1` 替代。`h11_p1_k10` 等价于历史 `h2_emg_both_p1_k10`，但本轮采用全量非测试人员重训练和固定 epoch 50，不是历史最佳验证权重的重复测试。

#### 3.8.2 总体结果

| 排名 | 配置 | N test BA | Macro-F1 | Accuracy | 正确数 | 说明 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `h11_p1_k10` | **84.30%** | **84.60%** | **84.38%** | 351/416 | P1 Proto+Rel active |
| 2 | `s0` | 84.17% | 83.61% | 83.65% | 348/416 | SupLoss-only 主基线 |
| 3 | `rn3` | 82.66% | 82.27% | 82.21% | 342/416 | P3 late-K3 exact Null |
| 4 | `h00_p1_k10` | 81.55% | 81.87% | 82.21% | 342/416 | P1 Top-K10 strict joint Null |
| 5 | `rl3` | 80.12% | 79.45% | 81.49% | 339/416 | P3 late-K3 Rel active |

自动汇总把每项写为“均值 ± 0.00”，但每个配置只有1个 subject × 1个 seed；这里直接报告单次点估计，不把0.00解释为标准差证据。

#### 3.8.3 Active–Null 与 SupLoss 双重比较

| 比较 | ΔBA | ΔF1 | ΔAccuracy | 正确数变化 | 判断 |
|---|---:|---:|---:|---:|---|
| `h11 - h00` | **+2.74 pp** | **+2.73 pp** | **+2.16 pp** | +9 | 联合损失相对严格同路径 Null 为正 |
| `h11 - s0` | +0.13 pp | +0.99 pp | +0.72 pp | +3 | 相对主基线近似持平，仅弱正信号 |
| `h00 - s0` | −2.62 pp | −1.74 pp | −1.44 pp | −6 | strict joint Null 路径本身弱于 `s0` |
| `rl3 - rn3` | **−2.53 pp** | **−2.82 pp** | **−0.72 pp** | −3 | Rel active 未超过 exact Null，MR 信号反转 |
| `rn3 - s0` | −1.51 pp | −1.34 pp | −1.44 pp | −6 | P3 Null 路径弱于 `s0` |
| `rl3 - s0` | −4.04 pp | −4.16 pp | −2.16 pp | −9 | Rel active 明显低于主基线 |

双重比较很重要。`h11` 超过 `h00` 说明 active objective 并非完全无效，但 `h00` 自身比 `s0` 低6个正确样本；`h11` 的大部分 Active–Null 增益是在补回 strict-Null 路径的退化，最终只比 `s0` 多3个正确样本。相反，`rl3` 既没有超过 `rn3`，也没有超过 `s0`。

#### 3.8.4 逐类变化

`h11-h00` 的 +9 个净正确样本主要来自：`insert` +8、`cap` +2、`tear` +2、`adjust` +1、`press` +1、`pull_out` +1；代价是 `wrap` −3、`label` −2、`cut` −1。由于这些变化相抵，不能只挑正向小类报告。

相对更重要的主基线 `s0`，`h11` 的变化明显收缩：`insert` +8、`cap` +2，但 `label` −3、`pull_out` −2、`wrap` −1、`cut` −1，净增仅3个样本。对应 recall 变化为 `cap` +11.11 pp、`insert` +9.09 pp、`label` −8.11 pp、`pull_out` −5.13 pp、`wrap` −2.86 pp、`cut` −2.33 pp。

`rl3-rn3` 的负向变化主要来自 `adjust` −2、`tear` −2、`insert` −2、`press` −1、`remove` −1；虽在 `label` +3、`cap` +1、`pull_out` +1，但净值仍为 −3。MR 上 `rl3` 曾改善 `adjust/press/tear/close`，而 N 上其中前三类全部下降，说明 RelLoss 的类间重分配具有明显 subject dependence。

#### 3.8.5 光照分层结果

N 测试集的三种光照数量接近平衡：left 137、normal 141、right 138；每个光照子集都覆盖相同的14个实际类别。分层指标如下：

| 配置 | Left BA/F1/Acc | Normal BA/F1/Acc | Right BA/F1/Acc |
|---|---:|---:|---:|
| `s0` | 81.97/81.01/78.83 | 80.27/79.39/82.27 | **90.34/90.00/89.86** |
| `rn3` | 83.00/81.63/80.29 | 80.92/80.40/82.98 | 83.97/84.24/83.33 |
| `rl3` | 81.27/80.34/78.83 | **70.93/68.82/78.01** | 88.13/86.57/87.68 |
| `h00_p1_k10` | 81.03/81.53/78.83 | 80.33/80.24/82.98 | 83.92/83.72/84.78 |
| `h11_p1_k10` | **85.23/84.75/83.94** | 78.91/80.31/82.27 | 89.08/88.28/86.96 |

`h11-h00` 在 left 和 right 上分别为 BA +4.21/+5.16 pp，但 normal 为 −1.42 pp；相对 `s0`，`h11` 只在 left 明显提高（BA +3.26、accuracy +5.11 pp），normal BA −1.37 pp，right BA/accuracy −1.26/−2.90 pp。它不是跨光照一致改善。

`rl3-rn3` 的总体负值主要由 normal 子集驱动：normal BA/F1/accuracy **−9.99/−11.58/−4.96 pp**；right 反而 +4.17/+2.33/+4.35 pp，left 略降。相对 `s0`，`rl3` 在三种光照的 BA 均未提高。由此可见，RelLoss 的错误重分配不仅依赖 subject，也依赖 lighting；现有结果不支持更强的光照不变性。

#### 3.8.6 预训练稳定性与 prototype 诊断

5个配置各有400条 debug 记录，均未出现 non-finite。epoch 200 的关键诊断如下：

| 配置 | Active prototypes | Near-dead | Assignment entropy | 同类 prototype cos | 最近异类 cos | Lighting NMI |
|---|---:|---:|---:|---:|---:|---:|
| `h00_p1_k10` | 15 | 0 | 1.000 | 不适用（P1） | 0.5185 | 0.0000 |
| `h11_p1_k10` | 15 | 0 | 1.000 | 不适用（P1） | **0.4140** | 0.0000 |
| `rn3` | 45 | 5 | 0.8560 | 0.9942 | 0.5528 | 0.0433 |
| `rl3` | 45 | 3 | 0.8817 | 0.9943 | **0.4801** | 0.0656 |

两组 active 都显著降低最近异类 prototype 相似度：`h11-h00` 为 −0.1045，`rl3-rn3` 为 −0.0727。可是只有 `h11` 的下游结果相对 Null 为正，`rl3` 反而更差，再次证明“prototype 分得更开”不是分类性能的充分条件。P3 的同类 prototype 仍以约0.994的余弦相似度高度重合；Lighting NMI/ARI 仍很低（`rl3` ARI 0.0199，`rn3` 0.0031），没有形成三光照语义。

新增损失的数值量级仍很小。最后一条 debug 记录中，`h11` 的加权 Proto+Rel 项合计约0.00536，而 SupLoss 为4.531；`rl3` 的加权 Rel 项约0.00059，而 SupLoss 为4.532。损失值占比不能代替梯度占比，但这些量级继续支持记录独立梯度范数与梯度夹角的必要性。

#### 3.8.7 统计解释

5份 `predictions.csv` 均有416个唯一 `original_key`，并能完全一一对齐。以下 paired bootstrap 使用固定随机种子20260814，重复10,000次；为避免少数类在重采样中消失，按14个实际出现的真实类别分别有放回采样，并保持原类别support。区间是未经多重比较校正的2.5%–97.5% percentile interval。McNemar 使用同一样本上“active正确/control错误”与“active错误/control正确”的 discordant counts 做双侧 exact binomial 检验。

| 比较 | ΔBA 95% CI (pp) | ΔF1 95% CI (pp) | ΔAcc 95% CI (pp) | Discordant active:control | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| `h11 - h00` | [−0.40, 6.19] | [−0.37, 6.46] | [−0.48, 4.81] | 21:12 | 0.163 |
| `h11 - s0` | [−3.29, 3.53] | [−2.74, 4.71] | [−1.92, 3.37] | 18:15 | 0.728 |
| `rl3 - rn3` | [−6.45, 1.11] | [−7.10, 0.92] | [−3.61, 2.16] | 17:20 | 0.743 |
| `rl3 - s0` | **[−7.76, −0.58]** | **[−8.36, −0.62]** | [−4.81, 0.48] | 12:21 | 0.163 |

`h11-h00` 的点估计为正，但三个区间都跨0，McNemar也未达到常用显著性阈值；相对 `s0` 的证据更弱。`rl3-rn3` 的负点估计同样不显著，但 `rl3` 相对 `s0` 的类别平衡指标 bootstrap 区间为负，支持“至少在固定 N/s1 模型上没有优势”，而不是支持 RelLoss。

这些区间只量化测试样本重采样误差。它们没有重训模型，不能覆盖 seed、训练轨迹或跨 subject 变化。特别是 N 的 `adjust` 仅8个样本，一个样本就改变12.5 pp recall；`cap/press/tear` 也只有12–18个样本。因此即使某个条件区间不跨0，也不能替代多 seed、多折 LOSO。

因此 U1a 的严格结论是：**管线和锁定评估协议运行正常；`h11` 保留弱正信号，`rl3` 的 MR 优势没有跨到 N。没有任何旧版损失在本轮表现出足以宣称稳定优于 SupLoss 的幅度。**

## 4. 对比预训练损失分析

### 4.1 训练稳定性

- 历史29个预训练日志和 U1a 新增5个日志中均没有发现 NaN/Inf；U1a 每配置400条 debug 记录也均为 finite。
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

U1a 已落实更严格的固定 epoch 50做法，因此消除了“每配置从验证曲线挑峰值”的偏差；但5个模型的最终训练 BA 仍为99.90%–100%，固定 epoch 只统一了评估规则，并没有消除训练集饱和或单 seed 方差。

## 7. ProtoLoss 是否发挥作用

更新后的综合判断：**ProtoLoss 正常执行，但历史 MR 外层测试和新增 N final-refit 都没有提供可单独归因于旧版 ProtoLoss 的泛化证据。**

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
- U1a 的 `h11` 同时启用了 ProtoLoss 与 RelLoss；虽然它在 N 上超过 strict joint Null `h00`，但没有运行 `h10`（Proto-only）与 `h01`（Rel-only），因此不能把该增益拆分或归因给 ProtoLoss。

因此，就当前旧版定义而言，应把 ProtoLoss 结论写成“未获得支持”，而不是“弱有效”。这不否定 prototype 主思路，但说明需要先解决同类 prototype 重合、assignment 语义和梯度目标问题。

## 8. RelLoss 是否发挥作用

更新后的综合判断：**RelLoss 明显改变 prototype 几何，但其分类收益具有强 subject dependence；MR 上的正信号已在 N 上反转，当前不支持稳定泛化增益。**

支持 RelLoss 的证据：

- active Rel 显著降低最近异类 prototype 的余弦相似度；
- exact matched P3 late-K3 active `rl3` 相对 `rn3`：test BA +5.11 pp、F1 +2.42 pp；
- `rl3` 达到全组最高 test BA/F1，而且它并不是 inner-val 第一名，降低了“只复现验证尖峰”的可能；
- exact P2 `rl2` 相对 `rn2` 虽 BA −0.14 pp，但 F1 +1.81 pp、accuracy +5.97 pp；P2/P3 都显示 Rel 会改变错误在类别之间的分配；
- P1 Rel-only `h1` 在 test 上相对 P1 Null 的 BA +2.43 pp，与 inner-val 的负向结果不同。
- U1a 中 `h11` 相对完全匹配的 joint Null `h00` 在 N 上 BA/F1/accuracy +2.74/+2.73/+2.16 pp，说明包含 RelLoss 的联合目标仍可能对部分受试者有效；但该比较不能分离 Proto 与 Rel。

仍需谨慎的证据：

- `rl3` 与 `rn3` accuracy 完全相同，BA 改善来自小类 recall 重分配，并非更多总正确样本；
- `re2` 验证第1却测试第15，early K10 P2 并不稳定；
- RelLoss 的加权量长期只有 SupLoss 的约 0.02%–0.09%，其效果可能高度依赖训练轨迹；
- 几何分离程度与分类性能不单调；
- U1a N 折 `rl3-rn3` 为 BA/F1/accuracy −2.53/−2.82/−0.72 pp，和 MR 折的 +5.11/+2.42/0.00 pp 方向相反；
- N 折 `rl3` 相对 `s0` 为 BA/F1/accuracy −4.04/−4.16/−2.16 pp；
- MR 与 N 都只有 seed 1，而且采用不同 checkpoint 选择协议，尚不能估计跨 seed 方差或用两个点做可靠总体效应估计。

因此当前最合理的表述是：**P3 late-K3 RelLoss 是一个已确认会改变类间几何、但分类效果跨人反向的机制候选，不再是优先性能候选。**若继续研究它，目的应是解释 subject dependence 或获得预注册的多折效应估计，而不是复现 MR 的单点最高值。

## 9. 确认实验进展与下一步

原建议中的最小因果确认已经以统一 U1a 协议执行了一部分：

| U1a 配置 | 目的 | N/s1 结果 |
|---|---|---|
| `s0` | 同 seed 主基线 | BA 84.17% |
| `rn3` | P3 late-K3 exact Null | BA 82.66% |
| `rl3` | 复现 MR 的 Rel 信号 | BA 80.12%；相对 `rn3` −2.53 pp，未复现 |
| `h00_p1_k10` | 与 H2 完全匹配的 Top-K10 strict joint Null | BA 81.55% |
| `h11_p1_k10` | 原 H2 Proto+Rel active | BA 84.30%；相对 `h00` +2.74 pp、相对 `s0` +0.13 pp |

U1a 已证明统一 runner、strict Null 与固定 epoch 测试链路可用，但没有达到“active 明显同时超过 SupLoss 和 exact Null”的性能门槛。接下来有两种不同目的，不能混在一起：

1. **如果目标是尽快推进主项目：**停止围绕旧版 `rl3`/Proto-only 扩大搜索，保留 U1a 为否定/异质性证据，转入预定的 t17 采样与 SupLoss/增强筛选；这些开发只能使用 `subject_dev`，不能根据已经查看过的 N 结果再称 N 为全新独立测试。
2. **如果目标是给旧版损失一个最终统计结论：**保持这5个配置完全冻结，执行 U1b 的其余受试者与至少一个额外 seed，并报告每个 subject/seed 的 Active–Null 和 Active–`s0` 配对差、均值、最差被试和方向一致率。此时 N 只是预先已查看的一折，不再是独立最终测试。

无论选择哪条路线，统一统计程序都应自动读取已经存在的逐样本预测并输出 paired bootstrap 与 exact McNemar，避免每轮手工补算。`h10/h01` 只有在研究问题确实需要拆分 H11 的 Proto/Rel 贡献时才值得加入；它们属于新的配置开发，应放在 `subject_dev`，而不是继续接触 N。

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
4. **不再把 P3 late-K3 当作默认性能起点**：MR 支持 `rl3`，但 N 上 exact Active–Null 反转。若保留该设置，应把它作为 subject-dependence 机制对照，并在冻结的多折/多 seed 设计中报告方向一致率。
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
- 不建议把 `h11` 的 N 折84.30%写成稳定提升；它只比 `s0` 高0.13 pp BA、3个正确样本；
- 不建议继续使用 MR 或 N 的已查看结果调参，并随后把它们称为全新独立测试证据。

### 当前建议

1. 冻结并保留 U1a 结果，不根据 N 继续修改这5个配置；
2. 主项目若强调效率，优先进入 t17 的 `subject_dev` 采样/SupLoss/增强实验，不扩大旧版 Loss 网格；
3. 若需要旧版损失的最终统计结论，只运行冻结 U1b，多折、多 seed 同时比较 `active-null` 与 `active-s0`，不再选择单折最高值；
4. 逐样本预测已经存在；下一步应把配对统计接入统一报告，并补独立梯度范数/夹角与 soft responsibility entropy；
5. Proto-only 旧版不再进入组合，除非改进版本先在 `subject_dev` 超过 exact Null；
6. `rl3` 只作为跨人异质性/机制对照保留，不再作为默认性能候选。

## 12. 最终回答

- **SupLoss：仍是最可靠主基线。**inner-val 相对 Direct BA +2.81 pp，held-out MR test BA +6.68 pp、F1 +7.21 pp；U1a N final-refit 的 `s0` 也达到84.17% BA、83.61% F1、83.65% accuracy。
- **ProtoLoss：旧版 active 没有得到外层测试支持。**hard P2 的验证优势在 test 反转；所有 Proto-only Active–Null 比较均未出现可信的三指标共同改善。多 prototype 高度重合仍是核心不足。
- **RelLoss：机制有效，但性能不稳定。**`rl3-rn3` 在 MR 为 BA/F1 +5.11/+2.42 pp，在 N 却为 −2.53/−2.82 pp；几何分离在两折都发生，但分类效应跨人反向。
- **Proto+Rel：`h11_p1_k10` 保留弱信号。**N 上相对完全匹配的 strict `h00` 为 BA/F1/accuracy +2.74/+2.73/+2.16 pp，但相对 `s0` 仅 +0.13/+0.99/+0.72 pp。P1 本质上是类中心，且没有 `h10/h01`，不能拆分 Proto 与 Rel 的贡献。
- **更新后的候选状态**：性能主线=`s0`；弱联合候选=`h11_p1_k10`；机制/异质性候选=`rl3/rn3`；Proto-only=无。现有旧版损失结果都不能直接写成稳定优于 SupLoss 的论文结论。

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

U1a 更新另外直接复核了：

- `results/rgb_mvit_pr_unified_followup_20260813/manifests/u1a_confirm15_min.meta.json`
- `results/rgb_mvit_pr_unified_followup_20260813/analysis_unified/analysis_audit.json`
- `results/rgb_mvit_pr_unified_followup_20260813/analysis_unified/UNIFIED_STATISTICAL_REPORT.md`
- 5个配置各自的 `run_meta/.../resolved_config.json`、`test/.../last_test_metrics.json`、`pretrain/.../debug_train_log.jsonl` 与 epoch 200 prototype/environment 诊断。

上述文件支持 U1a 的运行完整性、锁定配置、整体/逐类结果、Active–Null 差值和 prototype 几何结论。5个配置的 `predictions.csv` 还支持本报告新增的10,000次分层 paired bootstrap 和 exact McNemar；缺口是统一汇总程序尚未把这些统计自动写入报告。
