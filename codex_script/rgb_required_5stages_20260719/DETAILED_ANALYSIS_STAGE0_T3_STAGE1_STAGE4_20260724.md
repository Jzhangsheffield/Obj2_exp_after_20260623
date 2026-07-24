# RGB 必做实验详细分析报告：Stage 0、T3、Stage 1 与 Stage 4

生成日期：2026-07-24  
分析对象：00143 摄像头、16 帧 RGB、3D ResNet-18、15 类动作识别  
模型选择指标：validation Balanced Accuracy（BA）  
重要约束：本报告没有读取或使用测试集结果。

---

## 1. 关键结论

### 1.1 时间下采样不是目前的主要瓶颈

T3 将网络深层的时间长度从当前结构的 `16→16→8→8→4→2→1` 改为
`16→16→16→16→8→8→8`，但结果没有提升：

- T3 scratch：`88.33 ± 4.98%` BA；
- T3 SupLoss 预训练：`86.64 ± 2.43%` BA；
- 当前结构 Stage 0 scratch：`90.32 ± 0.64%` BA；
- 当前结构 Stage 0 SupLoss：`91.58 ± 1.84%` BA。

按相同 seed 比较，T3 SupLoss 比当前结构平均低约 5.23 个百分点。因此，“16 帧最终压成 1 个时间位置”不能解释当前的主要性能问题。保留 8 个时间位置后仍使用全局平均池化，反而可能增加优化难度，却没有增加显式的时序建模能力。后续主线应继续使用 current temporal mode；T3 不进入 ProtoLoss/RelLoss 组合实验。

### 1.2 ProtoLoss 有正面信号，但尚不能证明稳定优于 SupLoss 或 scratch

Stage 1 的单 seed 最佳结果是：

- `P6 all-positive, 3 prototypes/class`：`91.41%` BA；
- `P7 soft-positive, 3 prototypes/class`：`91.12%` BA；
- `P4 soft-positive, 2 prototypes/class`：`91.04%` BA；
- matched `P0 SupLoss-only`：`89.40%` BA。

所以 P6、P7、P4 相对同一次 Stage 1 的 P0 分别提高 `+2.01`、`+1.71`、`+1.63` 个百分点。与此同时：

- P6 的 `91.41%` 仍略低于 Stage 0 SupLoss 四次运行均值 `91.58%`；
- P6、P7、P4 之间只有 `0.08–0.37` 个百分点差距；
- 全部 Stage 1 只有一个 seed；
- `P2 single-positive, 2 prototypes/class` 下降到 `87.01%`，比 P0 低 `2.40` 个百分点。

因此，当前最严谨的结论是：**ProtoLoss 有值得继续确认的正面信号，但还不是已经得到多 seed 证明的稳定增益。**

### 1.3 当前 prototype 的主要不足不是严格 dead，而是“近死、失衡和高度重合”

从最终 checkpoint 可精确恢复 1,026 个训练样本的 assignment：

- 所有实验都覆盖全部 1,026 个训练样本；
- 所有有效 prototype 的计数都大于 0，因此严格 dead prototype 数量均为 0；
- 但 P6 的 `remove` 为 `24/31/1`，P7 的 `remove` 为 `24/1/31`；
- R9 的 `cap` 为 `30/10/1`，`remove` 为 `17/38/1`；
- R12 的 `remove` 为 `28/27/1`。

这些计数为 1 的槽位虽然不是严格 dead，却在统计意义上接近失效。

更重要的是，多个 prototype 没有形成清晰子模态：

- Stage 1 多 prototype 配置的类内 prototype 平均余弦相似度为 `0.9635–0.9791`；
- P4 的 deterministic soft responsibility 归一化熵为 `0.9385`，平均有效 prototype 数为 `1.92/2`；
- P7 的归一化熵为 `0.9595`，平均有效 prototype 数为 `2.88/3`；
- 不同类别最近 prototype 的平均余弦相似度仍达到约 `0.965–0.987`。

这说明 soft 权重通常接近平均分配，多个中心更像同一类别中心的重复副本，而不是不同速度、阶段、方向或动作外观的子模态。

### 1.4 RelLoss 最佳观察配置是 R9，但仍存在明显因果混杂

Stage 4 最佳为：

```text
R9:
diff-only
3 prototypes/class
top-k different classes = 3
prototype refresh starts at epoch 50
relation starts at epoch 125
preview EMA = 0.5
lambda_rel = 0.5
```

R9 达到：

- BA：`94.96%`；
- Accuracy：`95.03%`；
- macro-F1：`95.49%`；
- 最后 10 个 epoch 的 BA 均值：`94.44%`。

它相对同阶段 R0 提高 `+6.68` 个 BA 百分点，并且最后 10 个 epoch 的优势仍然存在。R9 是目前的**最佳观察设置**。

但不能仅凭该单 seed 结果宣称 RelLoss 已被证明有效，原因有三点：

1. Stage 4 的 R0 只有 `88.28%`，明显低于 Stage 0 SupLoss 均值 `91.58%`，是一个偏弱的单次基线；
2. R9 的 BA 增益中，`close` 类由 `25%` 提高到 `100%`。该类验证集只有 4 个样本，这一项单独贡献了 5.0 个 BA 百分点；
3. `contrastive_rel` 从 epoch 50 就会建立和刷新 prototype state，而 R9 的 relation 梯度到 epoch 125 才启用。R9 与 R0 的差异不只包含 relation 梯度，还包含 refresh、EMA、计算路径和潜在随机数轨迹差异。

因此必须用已新增的 Stage 4B `Null-rel` 对照，把 relation 梯度贡献和 prototype refresh/RNG 轨迹贡献分开。

### 1.5 RelLoss 的机制证据支持 R9，不支持通过增大 EMA 或 lambda 补救

在最终 checkpoint 上，用全部 1,026 个训练样本做 deterministic feature-level 梯度诊断：

- R9：加权 Rel 梯度约为 SupLoss 梯度的 `2.03%`，88.24% batch 非零；
- R7：约为 `4.24%`，100% batch 非零；
- R12（preview EMA 0.9）：只有 `0.061%`，仅 29.41% batch 非零；
- R13（preview EMA 0.99）：RelLoss 和梯度均为 0；
- R15（EMA 0.9、lambda 2）：只有 `0.070%`，仅 11.76% batch 非零。

这说明较大的 preview EMA 使新旧 prototype 距离变化太小，固定 `0.01` margin 的 ReLU 大量进入零区间。将 `lambda_rel` 从 0.5 增大到 2 或 5 并不能挽救一个已经失活的 loss。

---

## 2. 数据、配置和分析范围

### 2.1 已检查的结果

| 阶段 | 对比预训练内容 | 微调内容 |
|---|---:|---:|
| Stage 0 | 4 个 checkpoint + 4 份 debug JSONL | 8 份 summary |
| T3 | 3 个 checkpoint + 3 份 debug JSONL | 6 份 summary |
| Stage 1 | 8 个 checkpoint + 8 份 debug JSONL | 8 份 summary |
| Stage 4 | 17 个 checkpoint + 17 份 debug JSONL | 17 份 summary |

结果来源：

```text
results/cl_rgb_req_s0_20260719
results/ft_rgb_req_s0_20260719
results/cl_rgb_req_t3_20260721
results/ft_rgb_req_t3_20260721
results/cl_rgb_req_s1_20260719
results/ft_rgb_req_s1_20260719
results/cl_rgb_req_s4_20260719
results/ft_rgb_req_s4_20260719
```

### 2.2 共同训练设置

- 对比预训练：200 epochs，batch 64，AdamW；
- 预训练 LR `1e-3`，weight decay `1e-4`，milestones `50/100/150`；
- projection dimension 128，queue 1088，temperature 0.07；
- prototype temperature 0.07，每 10 epochs refresh，真实 prototype EMA 0.99；
- 微调：full fine-tune 100 epochs；
- backbone LR `3e-4`，head LR `1e-3`；
- 微调模型不是只训练分类头，而是全局微调；
- 所有结果按验证集 BA 选择 checkpoint。

### 2.3 诊断方法

本报告使用三类证据：

1. **微调 summary**：最佳 BA、Accuracy、macro-F1、最后 10 epoch BA、逐类别 recall；
2. **预训练 debug JSONL**：每个 epoch 记录的首个诊断 batch，用于观察 SupLoss、ProtoLoss、RelLoss、总梯度、feature std、prototype 有效比例；
3. **最终 checkpoint 离线诊断**：
   - 读取 `prototype_bank`、`sample_to_proto`、`sample_to_class`；
   - 统计每个 prototype 的最终 assignment 数量、dead/near-dead、hard assignment 熵；
   - 计算同类和异类 prototype 余弦相似度；
   - 对全部 1,026 个训练样本使用 deterministic transform，重新提取特征；
   - 计算 soft responsibility entropy；
   - 分别对 SupLoss 和辅助 loss 求特征层梯度，计算加权梯度范数比和方向余弦。

### 2.4 必须注意的限制

- debug JSONL 的 loss 是每个 epoch 首个诊断 batch，不是完整 epoch 所有 batch 的均值；
- 离线梯度是对最终 checkpoint 和 deterministic 输入的诊断，不是带随机增强的在线训练重放；
- 梯度比例是 projection feature `q` 上的梯度比例，不是所有网络参数的全局梯度比例；
- 当前下载的每个实验只有最终 prototype assignment map。因此可以精确检查最终计数，也能从日志确认 epoch 50 后状态是否激活，但**不能精确重建每次 refresh 后每个 prototype 的历史计数变化**；
- 所有 Stage 1 和 Stage 4 结果仍是单 seed，不能把小差异解释为稳定排序。

---

## 3. Stage 0：基线复现与随机性

### 3.1 微调结果

| 方法 | 运行 | 最佳 BA (%) |
|---|---|---:|
| scratch | seed1-a | 90.99 |
| scratch | seed1-b | 90.47 |
| scratch | seed2 | 89.45 |
| scratch | seed3 | 90.37 |
| SupLoss | seed1-a | 89.44 |
| SupLoss | seed1-b | 90.70 |
| SupLoss | seed2 | 92.71 |
| SupLoss | seed3 | 93.46 |

汇总：

| 方法 | 最佳 BA 均值 ± 样本标准差 (%) | 最后 10 epoch BA 均值 (%) |
|---|---:|---:|
| scratch | `90.32 ± 0.64` | 88.50 |
| SupLoss | `91.58 ± 1.84` | 90.56 |

SupLoss 平均比 scratch 高 `1.26` 个百分点，但方差明显更大。同为 seed1 的两次 SupLoss 微调相差 `1.26` 个百分点，说明只设置 seed 并没有使整个训练完全确定。

### 3.2 对比预训练轨迹

| 预训练 | epochs 151–200 SupLoss | grad norm | feature std |
|---|---:|---:|---:|
| sup_s1_a | 5.7804 | 0.9840 | 0.05284 |
| sup_s1_b | 6.0499 | 0.9290 | 0.04710 |
| sup_s2 | 6.0818 | 1.0804 | 0.04763 |
| sup_s3 | 5.8909 | 0.8040 | 0.05047 |

全部运行没有非有限值，但同 seed 的 `s1_a` 和 `s1_b` 已在早期出现不同 feature std，后期 SupLoss 也相差约 0.27。这支持之前的判断：相同摄像头基线差距大，最大可能性不是摄像头不同，而是训练链中仍有未完全锁定的随机性，以及小验证集下最佳 checkpoint 选择对少数样本非常敏感。

后续确认实验必须报告多 seed 均值和标准差，不能再以单次最高 BA 作为最终结论。

---

## 4. T3：保留时间维度的消融

### 4.1 微调结果

| 方法 | seed1 BA | seed2 BA | seed3 BA | 均值 ± 标准差 |
|---|---:|---:|---:|---:|
| T3 scratch | 82.74 | 92.28 | 89.98 | `88.33 ± 4.98` |
| T3 SupLoss | 87.33 | 83.94 | 88.66 | `86.64 ± 2.43` |

与当前结构按 seed 配对：

| seed | current SupLoss BA | T3 SupLoss BA | T3-current |
|---:|---:|---:|---:|
| 1（使用 current s1-a） | 89.44 | 87.33 | -2.12 |
| 2 | 92.71 | 83.94 | -8.77 |
| 3 | 93.46 | 88.66 | -4.80 |
| 平均 | 91.87 | 86.64 | **-5.23** |

### 4.2 预训练机制

T3 三个 seed 均正常下降且无非有限值，但末段特征并不优于 current：

| 预训练 | epochs 151–200 SupLoss | grad norm | feature std |
|---|---:|---:|---:|
| T3 seed1 | 6.0204 | 1.3355 | 0.04907 |
| T3 seed2 | 6.0550 | 0.9689 | 0.04514 |
| T3 seed3 | 6.1744 | 1.4632 | 0.04672 |

T3 seed3 的后期 SupLoss 较高、梯度较大，而 feature std 偏低。它没有表现出“保留时间分辨率后特征自然更可分”的迹象。

### 4.3 判断

当前任务不能简单类比长视频 LFB 网络。这里只有 16 帧，且最后仍用全局平均池化；保留 8 个深层时间位置并不会自动学习动作顺序。若未来重新研究时间建模，更合理的是：

- current backbone 后增加轻量 temporal attention/TCN；
- 对帧顺序做显式监督或 temporal contrast；
- 比较 16/32 帧并保持相同实际时间跨度；
- 不再只改 stride 后继续全局平均。

但这些不应阻塞当前 ProtoLoss/RelLoss 主线。

---

## 5. Stage 1：ProtoLoss 详细分析

### 5.1 对比预训练中的 ProtoLoss 数值

下表为 ProtoLoss 激活后的首 batch 日志均值。`weighted/Sup` 已包含 `lambda_proto=0.1`。

| 配置 | ProtoLoss | 加权 ProtoLoss | 加权值/SupLoss |
|---|---:|---:|---:|
| P1 single-P1 | 1.2701 | 0.1270 | 2.11% |
| P2 single-P2 | 2.1791 | 0.2179 | 3.41% |
| P3 all-P2 | 2.1800 | 0.2180 | 3.60% |
| P4 soft-P2 | 2.0637 | 0.2064 | 3.41% |
| P5 single-P3 | 2.0367 | 0.2037 | 3.39% |
| P6 all-P3 | 2.5184 | 0.2518 | 4.23% |
| P7 soft-P3 | 2.4136 | 0.2414 | 4.01% |

ProtoLoss 不是数值上消失的辅助项，也没有非有限值。增加 prototype 数量会提高 loss 量级，但仍明显小于 SupLoss。

### 5.2 epoch 50 启动后的变化

| 配置 | 区间 | SupLoss | ProtoLoss | 加权 ProtoLoss | grad norm | feature std |
|---|---|---:|---:|---:|---:|---:|
| P2 | 1–50 | 7.0454 | 0 | 0 | 0.1760 | 0.00978 |
| P2 | 51–100 | 6.6881 | 2.5597 | 0.2560 | 0.7169 | 0.03363 |
| P2 | 101–150 | 6.3096 | 2.1035 | 0.2104 | 1.0517 | 0.04200 |
| P2 | 151–200 | 6.0697 | 1.8742 | 0.1874 | 1.7845 | 0.04638 |
| P4 | 1–50 | 6.9816 | 0 | 0 | 0.2529 | 0.01678 |
| P4 | 51–100 | 6.3026 | 2.2837 | 0.2284 | 0.6774 | 0.04419 |
| P4 | 151–200 | 5.8493 | 1.9110 | 0.1911 | 1.2106 | 0.05189 |
| P6 | 1–50 | 6.8897 | 0 | 0 | 0.3313 | 0.02181 |
| P6 | 51–100 | 6.1789 | 2.8021 | 0.2802 | 0.6516 | 0.04530 |
| P6 | 151–200 | 5.7581 | 2.3017 | 0.2302 | 1.1493 | 0.05449 |

从 epoch 51 开始：

- `valid_proto_ratio=1.0`；
- P2/P4 的诊断 batch 均观察到 2 个 prototype ID；
- P6/P7 均观察到 3 个 prototype ID；
- 没有出现启动失败或非法 assignment。

P2 后期梯度明显高于 P4/P6，但 feature std 较低，且最终微调最差。说明 hard single assignment 在中心高度相似时可能放大不稳定的聚类边界，而不是缺少梯度。

当前文件不能恢复 epoch 50、60、70……每次 refresh 后的完整逐 prototype 计数。后续训练应把每类 assignment histogram 和 prototype 相似度写入每次 refresh 日志。

### 5.3 最终 assignment、dead prototype 和相似度

`H(assign)` 是基于最终 hard assignment 计数的类别内归一化熵；1 表示完全均衡。

| 配置 | 严格 dead | assignment CV | H(assign) | 类内 proto cos | 类内最小 cos | 异类最近 cos 均值 |
|---|---:|---:|---:|---:|---:|---:|
| P1 single-P1 | 0 | 0.0000 | 1.0000 | — | — | 0.9482 |
| P2 single-P2 | 0 | 0.3009 | 0.8926 | 0.9748 | 0.9441 | 0.9871 |
| P3 all-P2 | 0 | 0.3302 | 0.8913 | 0.9791 | 0.9554 | 0.9770 |
| P4 soft-P2 | 0 | 0.3521 | 0.8181 | 0.9708 | 0.9001 | 0.9808 |
| P5 single-P3 | 0 | 0.3937 | 0.8975 | 0.9664 | 0.7708 | 0.9732 |
| P6 all-P3 | 0 | 0.3154 | 0.9318 | 0.9690 | 0.7604 | 0.9651 |
| P7 soft-P3 | 0 | 0.4903 | 0.8654 | 0.9635 | 0.6197 | 0.9747 |

结论：

- 严格 dead 数量都是 0，但 P4/P6/P7 存在近死槽位；
- 同类 prototype 平均相似度接近 1，多 prototype 的结构分离不足；
- 异类最近 prototype 同样非常相似，现有 prototype 没有形成强判别边界；
- P6 的 assignment 最均衡，但它的 all-positive 目标会同时拉近全部同类 prototype，理论上本身就倾向于中心重合。

### 5.4 Stage 1 全部逐类 assignment

| class | n | P2 single-P2 | P4 soft-P2 | P6 all-P3 | P7 soft-P3 |
|---|---:|---|---|---|---|
| adjust | 31 | 14/17 | 14/17 | 14/5/12 | 11/17/3 |
| cap | 41 | 20/21 | 4/37 | 13/13/15 | 8/30/3 |
| close | 21 | 15/6 | 9/12 | 8/10/3 | 3/12/6 |
| cut | 108 | 86/22 | 103/5 | 26/65/17 | 35/47/26 |
| insert | 200 | 112/88 | 88/112 | 75/78/47 | 93/29/78 |
| label | 93 | 34/59 | 73/20 | 32/34/27 | 20/28/45 |
| measure | 62 | 26/36 | 32/30 | 17/25/20 | 19/7/36 |
| move | 71 | 54/17 | 11/60 | 34/23/14 | 45/16/10 |
| open | 54 | 38/16 | 19/35 | 17/26/11 | 23/12/19 |
| position | 31 | 16/15 | 19/12 | 10/12/9 | 12/5/14 |
| press | 31 | 16/15 | 13/18 | 12/9/10 | 6/8/17 |
| pull_out | 91 | 75/16 | 41/50 | 41/35/15 | 44/41/6 |
| remove | 56 | 39/17 | 52/4 | 24/31/1 | 24/1/31 |
| tear | 45 | 37/8 | 17/28 | 28/9/8 | 4/20/21 |
| wrap | 91 | 52/39 | 44/47 | 40/27/24 | 39/36/16 |

按“计数不超过 2，或不超过该类样本的 5%”定义 near-dead：

- P2：0 个 near-dead；
- P4：1 个，`cut=5/108`；
- P6：1 个，`remove=1/56`；
- P7：2 个，`cap=3/41`、`remove=1/56`。

### 5.5 soft responsibility entropy 与特征梯度

下表基于最终 checkpoint 的全部 1,026 个 deterministic 训练样本。

| 配置 | soft H/最大熵 | 平均最大责任 | 有效 proto 数 | 加权 loss/Sup | 加权梯度/Sup | 梯度方向 cos |
|---|---:|---:|---:|---:|---:|---:|
| P2 single-P2 | 0.9502 | 0.5998 | 1.935/2 | 2.85% | 11.24% | 0.627 |
| P4 soft-P2 | 0.9385 | 0.6020 | 1.921/2 | 3.15% | 9.78% | 0.748 |
| P6 all-P3 | 0.9598 | 0.4214 | 2.878/3 | 3.80% | 11.97% | 0.820 |
| P7 soft-P3 | 0.9595 | 0.4060 | 2.881/3 | 3.73% | 9.55% | 0.733 |

关键解释：

- ProtoLoss 加权后只有 SupLoss 标量的约 3–4%，但特征梯度达到约 9.5–12%，所以它并不“太弱”；
- ProtoLoss 与 SupLoss 梯度方向总体同向，尤其 P6 达到 0.82，更像加强类内聚合；
- soft responsibility 的熵过高，P4 近似同时使用 1.92/2 个中心，P7 近似同时使用 2.88/3 个中心；
- 因此当前 soft-positive 实际上接近 all-positive，而不是清晰选择一个子模态；
- P2 的失败不能通过简单增大 `lambda_proto` 修复。它已有 11.24% 的相对梯度，问题更可能是 hard assignment 边界不稳定。

### 5.6 微调结果

| 排名 | 配置 | BA (%) | 相对 P0 | Accuracy (%) | macro-F1 (%) | 最后 10 epoch BA (%) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | P6 all-P3 | 91.41 | +2.01 | 92.27 | 91.97 | 90.66 |
| 2 | P7 soft-P3 | 91.12 | +1.71 | 92.82 | 91.55 | 90.01 |
| 3 | P4 soft-P2 | 91.04 | +1.63 | 92.27 | 91.51 | 90.35 |
| 4 | P3 all-P2 | 90.43 | +1.03 | 91.71 | 91.35 | 88.93 |
| 5 | P1 single-P1 | 90.41 | +1.01 | 91.16 | 90.97 | 89.75 |
| 6 | P5 single-P3 | 90.32 | +0.92 | 92.27 | 90.94 | 88.25 |
| 7 | P0 SupLoss | 89.40 | 0 | 88.95 | 87.46 | 86.44 |
| 8 | P2 single-P2 | 87.01 | -2.40 | 88.40 | 87.07 | 86.14 |

P6 相对 P0 的主要逐类变化：

- `open`：70% → 100%；
- `pull_out`：75% → 100%；
- `position`：83.33% → 100%；
- `remove`：90% → 100%；
- 但 `close`：100% → 75%，`cap`：85.71% → 71.43%。

提升并不是所有类别一致改善，且小类每错对一个样本就会显著改变 BA。

### 5.7 ProtoLoss 是否有用

当前证据等级：

- **有用的信号：有。** P4/P6/P7 均比 matched P0 高，且 ProtoLoss 梯度稳定非零；
- **已证明稳定有效：没有。** 只有一个 seed，最佳 P6 没有超过 Stage 0 SupLoss 均值；
- **当前最佳观察值：P6 all-positive、3 prototypes、lambda 0.1、epoch 50 启动；**
- **更适合继续改进的结构：P4 soft-positive、2 prototypes。** 它更简单、near-dead 较少，并能在加入显式多样性约束后真正形成 mixture。

P6 应作为确认基准保留，但不建议直接把 all-positive 当作最终机制，因为其目标与“多个 prototype 表示不同子模态”存在内在冲突。

---

## 6. Stage 4：RelLoss 详细分析

### 6.1 微调结果完整排序

| 排名 | 配置 | BA (%) | 相对 R0 | Accuracy (%) | macro-F1 (%) | 最后 10 epoch BA (%) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | R9 diff-K3-start125-EMA0.5-lambda0.5 | 94.96 | +6.68 | 95.03 | 95.49 | 94.44 |
| 2 | R7 diff-K10-start50-EMA0.5-lambda0.5 | 93.18 | +4.90 | 93.37 | 92.19 | 91.35 |
| 3 | R12 diff-K3-start125-EMA0.9-lambda0.5 | 93.03 | +4.75 | 92.27 | 93.02 | 91.37 |
| 4 | R3 same+diff-K5 | 92.72 | +4.44 | 92.82 | 92.27 | 90.45 |
| 5 | R11 diff-K3-start125-EMA0.8 | 92.30 | +4.02 | 92.82 | 92.62 | 91.30 |
| 6 | R16 diff-K3-start125-EMA0.9-lambda5 | 92.14 | +3.86 | 93.37 | 91.98 | 91.12 |
| 7 | R5 diff-K3-start50 | 91.96 | +3.67 | 92.82 | 92.33 | 90.28 |
| 8 | R15 diff-K3-start125-EMA0.9-lambda2 | 91.73 | +3.45 | 93.37 | 92.41 | 90.04 |
| 9 | R6 diff-K5-start50 | 91.32 | +3.04 | 93.37 | 91.48 | 89.78 |
| 10 | R13 diff-K3-start125-EMA0.99 | 90.92 | +2.63 | 92.27 | 91.29 | 89.90 |
| 11 | R1 current same+diff-all classes | 90.74 | +2.46 | 91.71 | 91.23 | 88.16 |
| 12 | R8 diff-K3-start100 | 89.63 | +1.35 | 91.71 | 91.16 | 88.05 |
| 13 | R10 diff-K3-start150 | 89.34 | +1.06 | 90.61 | 90.61 | 88.67 |
| 14 | R14 diff-K3-start125-EMA0.9-lambda1 | 89.16 | +0.88 | 91.71 | 90.27 | 86.34 |
| 15 | R4 same+diff-K10 | 88.90 | +0.62 | 89.50 | 88.82 | 87.60 |
| 16 | R0 SupLoss-only | 88.28 | 0 | 92.27 | 88.11 | 87.40 |
| 17 | R2 same+diff-K3 | 86.56 | -1.72 | 90.06 | 87.11 | 84.51 |

从排序看：

- diff-only 整体比同时约束 same+diff 更可信；
- top-k、启动时刻和 preview EMA 存在明显交互，不能分别独立解释；
- `lambda_rel` 并不随数值增大而单调改善；
- R9 的最后 10 epoch BA 仍为 94.44%，不是单个 epoch 的瞬时尖峰。

### 6.2 RelLoss 在线日志中的实际数值

| 配置 | 区间 | RelLoss | 加权 RelLoss |
|---|---|---:|---:|
| R7 | 51–100 | 0.001427 | 0.000713 |
| R7 | 101–150 | 0.003330 | 0.001665 |
| R7 | 151–200 | 0.005072 | 0.002536 |
| R9 | 51–100 | 0 | 0 |
| R9 | 101–150 | 0.000860 | 0.000430 |
| R9 | 151–200 | 0.002451 | 0.001225 |
| R12 | 101–150 | 0.000061 | 0.000031 |
| R12 | 151–200 | 0.000124 | 0.000062 |
| R13 | 101–200 | 0 | 0 |
| R15 | 101–150 | 0.000024 | 0.000048 |
| R15 | 151–200 | 0.000085 | 0.000169 |

R9 在 epoch 125 才启动，所以 101–150 区间被前 25 个无 RelLoss 的 epoch 稀释。R12/R13/R15 的量级则确实接近失活。

### 6.3 最终特征层梯度诊断

| 配置 | 加权 RelLoss/SupLoss | 加权 Rel 梯度/Sup 梯度 | 梯度方向 cos | 非零 batch |
|---|---:|---:|---:|---:|
| R7 EMA0.5/K10 | 0.01860% | 4.240% | 0.139 | 100% |
| R9 EMA0.5/K3 | 0.00558% | 2.031% | 0.137 | 88.24% |
| R12 EMA0.9/K3 | 0.00013% | 0.061% | 0.099 | 29.41% |
| R13 EMA0.99/K3 | 0 | 0 | — | 0% |
| R15 EMA0.9/K3/lambda2 | 0.00002% | 0.070% | 0.170 | 11.76% |

虽然 RelLoss 标量很小，R9/R7 的梯度并非完全可以忽略。ReLU/hinge 类损失在少数 active pair 上可产生有限梯度，所以不能只看 loss 标量判断作用。

同时，RelLoss 与 SupLoss 梯度方向余弦只有约 0.10–0.17，远低于 ProtoLoss 的 0.63–0.82。RelLoss 提供的是近似正交的几何正则，可能补充 SupLoss，也可能在配置不当时成为噪声。这与 Stage 4 中不同设置大幅波动一致。

### 6.4 prototype assignment 与几何

| 配置 | 严格 dead | assignment CV | H(assign) | 类内 proto cos | 类内最小 cos | 异类最近 cos 均值 |
|---|---:|---:|---:|---:|---:|---:|
| R7 | 0 | 0.3979 | 0.9132 | 0.9770 | 0.9258 | 0.9741 |
| R9 | 0 | 0.4715 | 0.8634 | 0.9643 | 0.8242 | 0.9805 |
| R12 | 0 | 0.4843 | 0.8636 | 0.9592 | 0.6950 | 0.9801 |
| R15 | 0 | 0.4115 | 0.9072 | 0.9727 | 0.8332 | 0.9657 |

R9 并没有带来更均衡的 prototype，也没有把异类最近中心明显推远。因此其下游增益不能简单解释为“prototype 几何全面改善”。

### 6.5 Stage 4 关键配置逐类 assignment

| class | n | R7 K10 | R9 K3/start125 | R12 EMA0.9 | R15 EMA0.9/lambda2 |
|---|---:|---|---|---|---|
| adjust | 31 | 15/12/4 | 23/3/5 | 10/15/6 | 3/19/9 |
| cap | 41 | 15/16/10 | 30/10/1 | 27/2/12 | 10/21/10 |
| close | 21 | 9/7/5 | 8/7/6 | 10/6/5 | 12/5/4 |
| cut | 108 | 22/25/61 | 17/70/21 | 70/17/21 | 44/54/10 |
| insert | 200 | 78/77/45 | 63/92/45 | 78/99/23 | 44/66/90 |
| label | 93 | 39/49/5 | 28/23/42 | 56/28/9 | 41/30/22 |
| measure | 62 | 27/15/20 | 17/36/9 | 27/20/15 | 16/30/16 |
| move | 71 | 38/22/11 | 29/16/26 | 27/10/34 | 38/21/12 |
| open | 54 | 22/24/8 | 26/23/5 | 27/16/11 | 33/18/3 |
| position | 31 | 5/14/12 | 10/18/3 | 12/15/4 | 5/14/12 |
| press | 31 | 5/20/6 | 8/8/15 | 3/9/19 | 6/10/15 |
| pull_out | 91 | 45/34/12 | 30/22/39 | 33/55/3 | 34/31/26 |
| remove | 56 | 32/16/8 | 17/38/1 | 28/27/1 | 25/8/23 |
| tear | 45 | 7/20/18 | 14/25/6 | 7/18/20 | 17/23/5 |
| wrap | 91 | 39/20/32 | 39/28/24 | 22/36/33 | 30/45/16 |

near-dead 数量：

- R7：1 个；
- R9：2 个；
- R12：3 个；
- R15：1 个。

R9/R12 的高 BA 与 assignment 平衡度没有正相关，进一步表明单 seed 下游结果可能同时受到训练轨迹和小验证集波动影响。

### 6.6 R9 的逐类别改善来自哪里

相对 R0：

- `close`：25% → 100%，增加 75 个百分点；
- `remove`：80% → 100%；
- `position`：83.33% → 100%；
- `measure`：81.82% → 90.91%；
- `cap`：85.71% → 71.43%；
- `pull_out`：100% → 93.75%。

验证集每类等权计算 BA。`close` 只有 4 个样本，从 1/4 正确变成 4/4 正确，单独贡献：

```text
(1.00 - 0.25) / 15 = 0.05
```

即 5.0 个 BA 百分点。R9 的总 BA 增益为 6.68 点，所以大部分 BA 提升来自这个极小类别。不过 Accuracy 仍由 92.27% 提高到 95.03%，说明提升也不完全是 BA 计算假象。

### 6.7 RelLoss 是否有用

当前证据等级：

- **最有希望的配置：R9。** 它的下游结果最好，Rel 梯度非零且约占 Sup 梯度 2%；
- **辅助候选：R7。** 梯度更稳定、第二高 BA，但 start50/K10 改动同时存在；
- **R12 的高 BA 不能用其最终 Rel 梯度解释。** 最终有效梯度只有 0.061%；
- **R13 已确认机制失活。**
- **R15/R16 说明增大 lambda 不是正确补救方式。**
- **是否“确实有用”仍待 Stage 4B 多 seed + Null-rel。**

在 Stage 4B 结果出来前，R9 应称为“最佳观察设置”，而不是“最终被证明有效的设置”。

---

## 7. 当前最佳设置与实验决策

### 7.1 当前观察到的最佳配置

ProtoLoss：

```text
P6
positive mode = all
num_prototypes = 3
lambda_proto = 0.1
proto_start = 50
```

机制改进起点：

```text
P4
positive mode = soft
num_prototypes = 2
lambda_proto = 0.1
proto_start = 50
```

RelLoss：

```text
R9
same weight = 0
diff weight = 1
top-k different classes = 3
rel_start = 125
preview EMA = 0.5
lambda_rel = 0.5
```

预算允许时的第二候选：

```text
R7
diff-only
top-k = 10
rel_start = 50
preview EMA = 0.5
lambda_rel = 0.5
```

### 7.2 暂时不推荐

- T3 temporal mode；
- multi-prototype hard single-positive；
- same+diff relation 同时开启；
- preview EMA `0.9/0.99` 作为 RelLoss 默认值；
- 通过把 `lambda_rel` 提高到 2 或 5 修复 loss 失活；
- 在没有多 seed 和 Null-rel 前直接进入大规模 Stage 5。

### 7.3 必须先完成的确认实验

当前实验包已新增：

```text
stage4b_small_confirmation
```

必须完成 seed2、seed3：

- relation：R0、R9、R12；
- Null-rel：R9 完全相同计算路径，但 `lambda_rel=0`，运行 seed1/2/3；
- prototype：P0、P4、P6；
- 可选：R7 seed2/3。

判定标准：

1. 比较候选与同 seed matched baseline 的 BA 差值；
2. 报告三 seed 均值、样本标准差和每 seed 配对差；
3. R9/R12 必须显著高于 Null-rel，才能把提升归因于 relation 梯度；
4. P4/P6 必须在均值上高于 matched P0，且不是单个 seed 驱动；
5. 所有选择冻结后才允许运行测试集。

---

## 8. ProtoLoss 后续如何改进

### 8.1 把“多中心”目标从重复类中心改为真正 mixture

当前 all-positive 会把一个样本同时拉向同类全部中心，这与子模态分离相冲突。建议把下一版主目标改为：

```text
L_proto = -log sum_m r_im * exp(sim(q_i, p_yi,m) / tau)
```

其中 `r_im` 是 stop-gradient responsibility。每个样本主要拉近相关中心，而不是全部中心。

### 8.2 加入轻量类内多样性约束

可以加入：

```text
L_div = mean ReLU(cos(p_c,i, p_c,j) - margin_div)
```

建议初始小网格：

- `margin_div ∈ {0.7, 0.8}`;
- `lambda_div ∈ {0.01, 0.03}`;
- 先在 P4 soft-P2 上验证；
- 监控类内平均 cos 是否从约 0.97 降到 0.7–0.9，而不损伤类别间隔。

不要直接用很大的排斥权重，否则同一类样本会被人为撕裂。

### 8.3 用平衡 assignment 防止 near-dead

建议按优先级尝试：

1. 每类独立的 balanced Sinkhorn assignment；
2. 每次 refresh 设最小簇大小，例如 `max(3, 0.05*n_class)`；
3. 小于阈值的 prototype 从该类高损失样本或最远样本重新初始化；
4. 按类别样本量动态选择 prototype 数，而不是所有类固定为 3。

验收指标：

- strict dead = 0；
- near-dead = 0；
- assignment entropy 不低于 0.8；
- 但 soft responsibility entropy 不能继续接近 1。

### 8.4 控制 soft responsibility 的尖锐度

当前 P4/P7 soft entropy 太高。建议在加入多样性和 balanced assignment 后再比较：

- prototype temperature：`0.03 / 0.05 / 0.07`；
- 目标归一化 responsibility entropy：约 `0.55–0.85`；
- 同时监控平均最大 responsibility，目标大致 `0.65–0.90`。

只降低 temperature 而不先分离中心，可能只是把数值噪声变成硬 assignment。

### 8.5 不要继续简单增大 lambda_proto

当前 `lambda_proto=0.1` 已产生约 9.5–12% 的相对特征梯度。下一轮优先比较：

- `lambda_proto=0.05`；
- `lambda_proto=0.1`；
- 以加权梯度/Sup 梯度约 5–10% 为目标。

如果加入 diversity loss，总辅助梯度还需重新标定，不能机械叠加原权重。

### 8.6 增加动作子模态约束

RGB 动作的 prototype 应尽量对应可解释的变化，如动作阶段、运动方向、速度或视角。可加入：

- 两个 augmentation view 的 responsibility consistency；
- 相邻时间片的 prototype transition consistency；
- 同一原视频 clip 的 assignment 稳定性；
- 以动作阶段为弱监督的 temporal prototype。

这样可以避免 K-means 仅按背景、手部位置或光照形成簇。

---

## 9. RelLoss 后续如何改进

### 9.1 首先解决 hinge 大量为零

当前 `rel_diff_margin=0.01` 配合高 preview EMA，使大量 pair 进入 ReLU 零区间。两种可选改法：

1. 用平滑版本替代硬 ReLU：

```text
softplus((D_old - D_new - margin) / temperature)
```

2. 使用动态 margin：

```text
margin = 某个 batch 内 |D_new-D_old| 的分位数
```

推荐先试 softplus，因为它保留小变化的梯度且更容易稳定记录。

### 9.2 按 active pair 数量归一化

当前不同 batch 的 active pair 比例差异很大。建议记录并使用：

- active pair ratio；
- active pair 上的平均 loss；
- 每类入选 top-k 次数；
- 对 active pair 数量而不是全部候选 pair 归一化。

否则 loss 大小同时受到“几何变化”和“多少 pair 被 margin 截断”影响。

### 9.3 默认保留 EMA 0.5，不用高 EMA

现有证据清晰表明：

- EMA 0.5 的 R7/R9 有有效梯度；
- EMA 0.9 基本失活；
- EMA 0.99 完全失活。

下一版可只在 `0.3/0.5/0.7` 内小范围搜索。不要再把 0.9 作为主配置。

### 9.4 用梯度比例控制 lambda，而不是看 loss 标量

RelLoss 标量很小，但 R9 的梯度仍约 2%。建议使用自适应权重或离线校准，使：

```text
||lambda_rel * grad_rel|| / ||grad_sup|| ≈ 1%–3%
```

R7 的 4.24% 可作为较强上限。若 active gradient 为 0，增大 lambda 没有意义；应先修改 margin/EMA。

### 9.5 只约束真正困难的异类关系

Stage 4 支持 diff-only。后续可以用微调混淆矩阵定义候选边：

- 优先约束 `close/open`、`insert/pull_out` 等高混淆动作；
- top-k 由“当前 prototype 距离 + 历史混淆频率”共同确定；
- 避免同一易分类类别反复占据 top-k；
- 保留 stop-gradient target，避免两个距离矩阵同时漂移。

### 9.6 彻底隔离 refresh/RNG 混杂

Null-rel 是当前最重要的对照。训练器后续还应：

- 在 prototype refresh 前保存 RNG state，refresh 后恢复；
- 确保 `lambda_rel=0` 与 `contrastive_only` 的模型更新完全一致；
- 单独记录 refresh 耗时、调用次数和随机数消耗；
- 保存每次 refresh 的 assignment histogram 和 prototype bank；
- 对同一 checkpoint 离线重放 RelLoss，验证实现一致性。

---

## 10. 下一步实验顺序

1. **先运行 Stage 4B required pipeline**，不要运行测试；
2. 汇总 P0/P4/P6 和 R0/R9/R12/Null-rel 的三 seed 配对结果；
3. 若 P4/P6 均值不能超过 P0，先改 ProtoLoss，不进入组合；
4. 若 R9 与 Null-rel 接近，先修复 refresh/RNG 混杂，不把增益归于 RelLoss；
5. 若 R9 稳定高于 R0 和 Null-rel，保留 R9；预算允许再确认 R7；
6. 只有两个单项都通过后，再重新配置 Stage 5：
   - ProtoLoss 从 P4 或确认后的 P6 启动；
   - RelLoss 使用 R9，而不是旧的 EMA0.9/lambda2；
   - proto epoch 50，rel epoch 125；
   - 组合实验必须同时包含 Sup-only、Proto-only、Rel-only、Proto+Rel；
7. 最终模型和所有超参数冻结后，才对测试集运行一次。

---

## 11. 最终回答

### ProtoLoss 是否有用？

**有正面信号，但未完成统计确认。** P6/P7/P4 在 matched 单 seed P0 上提高约 1.6–2.0 个 BA 点，且 ProtoLoss 梯度真实存在；但中心高度重合、soft responsibility 近似均匀、存在 near-dead，最佳 P6 也没有超过 Stage 0 SupLoss 多次运行均值。当前应把它视为需要改进和多 seed 确认的有效候选。

### RelLoss 是否有用？

**R9 很有希望，但目前不能完成因果归因。** R9 达到全实验最高的 94.96% BA，且 Rel 梯度约为 Sup 梯度的 2%；不过单 seed、弱 R0、小类 `close` 和 refresh/RNG 混杂都可能放大结果。必须以 Stage 4B 的 Null-rel 和多 seed 为最终判断依据。

### 当前最佳设置是什么？

- 最佳观察 Proto：`P6 all-positive / 3 prototypes / lambda 0.1 / start50`；
- 最适合继续机制改进的 Proto：`P4 soft-positive / 2 prototypes`，加入平衡 assignment 和轻量 diversity；
- 最佳观察 Rel：`R9 diff-only / K3 / start125 / preview EMA0.5 / lambda0.5`；
- 可选 Rel：`R7 diff-only / K10 / start50 / preview EMA0.5 / lambda0.5`；
- 不建议：T3、hard single 多 prototype、EMA≥0.9、用大 lambda 补救失活 RelLoss。
