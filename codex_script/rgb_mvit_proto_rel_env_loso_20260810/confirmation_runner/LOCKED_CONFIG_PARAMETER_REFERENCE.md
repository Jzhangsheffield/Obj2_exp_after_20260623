# 锁定配置完整参数字典

> 本文件保留旧15类确认协议的配置含义。2026-08-13之后的新15/17类任务、增强、采样和50轮固定epoch协议请以[`UNIFIED_EXPERIMENT_CONFIGS.md`](./UNIFIED_EXPERIMENT_CONFIGS.md)为准；Loss简写仍与本文件兼容。

本文件回答“简称究竟代表什么”。训练脚本实际读取 `config/locked_config_registry.json`；每次运行还会在 `run_meta/fold_<对象>/<配置>/s<seed>/resolved_config.json` 保存最终解析参数。

## 快速索引

| 简称 | 完整名称 | SupLoss | ProtoLoss | RelLoss | P | Top-K | 严格对照 |
|---|---|---|---|---|---:|---:|---|
| `d0` | `d0_k400_direct` | 无对比预训练 | Off | Off | — | — | 直接微调基准 |
| `s0` | `s0_sup` | Active | Off | Off | 1（名义） | 未使用 | 主基线 |
| `rn3` | `rn3_k3_s125` | Active | Off | Null，λ=0 | 3 | 3 | `rl3` |
| `rl3` | `rl3_k3_s125` | Active | Off | Active，λ=0.5 | 3 | 3 | `rn3` |
| `rn2` | `rn2_k3_s125` | Active | Off | Null，λ=0 | 2 | 3 | `rl2` |
| `rl2` | `rl2_k3_s125` | Active | Off | Active，λ=0.5 | 2 | 3 | `rn2` |
| `h00_p1_k10` | `h00_p1_k10_strict_null` | Active | Null，λ=0 | Null，λ=0 | 1 | 10 | `h11_p1_k10` |
| `h10_p1_k10` | `h10_p1_k10_proto_only` | Active | Active，λ=1 | Null，λ=0 | 1 | 10 | `h00_p1_k10` |
| `h01_p1_k10` | `h01_p1_k10_rel_only` | Active | Null，λ=0 | Active，λ=1 | 1 | 10 | `h00_p1_k10` |
| `h11_p1_k10` | `h11_p1_k10_proto_rel` | Active | Active，λ=1 | Active，λ=1 | 1 | 10 | `h00_p1_k10` |

**Off**表示机制不作为目标使用；**Null**表示保留同一机制路径和诊断设置但权重为0；**Active**表示参与反向传播。

## 第一阶段锁定配置

### `d0`：K400直接全局微调（可选）

- 不运行对比预训练，Kinetics-400初始化的MViT-v2-S直接做下游全局微调。
- 它是“无对比预训练的direct baseline”，不是随机初始化意义上的true scratch。
- 默认最小锁定组不包含它；使用`phase1_with_direct`可在相同fold/seed下补充该基准。

### `s0`：SupLoss-only

- 来源：`stage1/s0_sup`；`ablation_mode=contrastive_only`。
- SupLoss Active，`lambda_proto=0`，`lambda_rel=0`。
- `num_prototypes=1`只是统一接口的名义值，不表示启用ProtoLoss。
- `num_positive=6`对SupLoss不起作用；SupLoss使用同标签正样本集合，该参数只对KCL路径有意义。
- 目的：每个fold、每个seed都重新训练的同期主基线。

### `rn3`：P3 late-K3 Null-rel

- 来源：`stage3a/rn3_k3_s125`；`ablation_mode=contrastive_rel`。
- SupLoss Active；ProtoLoss不参与梯度；`lambda_rel=0`。
- `num_prototypes=3`，`proto_positive_mode=all`。
- diff-only：`rel_same_weight=0`，`rel_diff_weight=1`。
- `rel_topk_diff_classes=3`，`rel_start=125`，end=200。
- `preview_ema_momentum=0.5`，`rel_lambda_schedule=constant`。
- 目的：`rl3`的严格Null control。

### `rl3`：P3 late-K3 Active RelLoss

- 所有设置与`rn3`相同，唯一确认性差异是`lambda_rel=0.5`。
- epoch 125开始启用RelLoss；constant权重；只优化不同类别关系。
- 目的：复现MR上BA/Macro-F1的RelLoss候选信号。

### `h00_p1_k10`：H2严格Null

- 来源模板：`stage4/h2_emg_both_p1_k10`，不是历史`hn1_null_p1`。
- `ablation_mode=contrastive_proto_rel`，P=1，all-positive。
- `lambda_proto=0`，`lambda_rel=0`。
- Proto/Rel均从epoch 50开始；EMA=0.5。
- same+diff：两个relation权重均为1；Top-K=10；cosine日程。
- 历史`hn1_null_p1`使用Top-K3，而历史h2使用Top-K10，因此并非完全严格配对。新配置修正了这个问题。

### `h11_p1_k10`：H2 Proto+Rel Active

- 除损失权重外与`h00_p1_k10`完全相同。
- `lambda_proto=1`，`lambda_rel=1`。
- P1本质为每类单中心，不是多prototype子簇。
- 目的：确认历史H2三项指标共同改善能否跨对象、跨seed重复。

## 可选P2 Rel确认

`rn2/rl2`与`rn3/rl3`的结构差异是P=2。`rn2`与`rl2`的唯一Active–Null差异仍为`lambda_rel=0/0.5`，用于确认P2是否偏向Accuracy、P3是否偏向BA/F1。

## H2 2×2消融

| 配置 | λproto | λrel | 问题 |
|---|---:|---:|---|
| `h00_p1_k10` | 0 | 0 | 严格空对照 |
| `h10_p1_k10` | 1 | 0 | ProtoLoss独立贡献 |
| `h01_p1_k10` | 0 | 1 | RelLoss独立贡献 |
| `h11_p1_k10` | 1 | 1 | 联合效果与交互作用 |

四项的P、Top-K、启动时间、EMA、same/diff权重和cosine日程完全相同。

## 共享对比预训练参数

| 参数 | 值 |
|---|---|
| Backbone/初始化 | MViT-v2-S / Kinetics-400 |
| 输入 | RGB摄像头00143，16帧，224×224 |
| patch embedding | 可训练 |
| 类别数 | 15 |
| epochs / batch size | 200 / 32 |
| optimizer / LR / WD | AdamW / 0.00006 / 0.0001 |
| projection dim / queue | 128 / 1088 |
| SupLoss/prototype temperature | 0.07 / 0.07 |
| LR warm-up / schedule | 10 / cosine |
| 权重保存 | 每50 epochs |
| prototype诊断/重聚类 | 每10 / 每10 epochs |
| prototype EMA | 0.99 |
| KMeans random state/n_init/max_iter | 42 / 10 / 300 |
| Rel same/diff margin | 0.01 / 0.01 |

## 预训练增强

- 时间采样`global_uniform`，两个视图采用shared temporal view。
- RandomResizedCrop scale `[0.85,1.0]`，ratio `[0.9,1.1]`。
- 水平/垂直翻转概率0.5/0。
- jitter概率0.2，brightness/contrast/saturation/hue=`0.1/0.1/0.1/0.02`。
- grayscale 0；blur概率0.1、kernel 5、sigma `[0.1,1.0]`。
- RGB mean `[0.45,0.45,0.45]`，std `[0.225,0.225,0.225]`。

## 微调与测试

| 参数 | 值 |
|---|---|
| 策略 | 全局微调 |
| epochs / batch size | 100 / 32 |
| optimizer | AdamW |
| backbone/head LR | 0.00006 / 0.002 |
| weight decay / milestones | 0.0001 / 50、75 |
| 保存频率 | 每25 epochs |
| 模型选择 | inner-val BA最佳的`best_val_balanced.pth` |
| 测试 | batch 32，AMP开启 |

微调增强为RRC `[0.85,1.0]`、ratio `[0.9,1.1]`、水平翻转0.5；jitter、gray和blur关闭。外层测试不参与checkpoint选择。

## 配置组

- `phase1_locked`：`s0,rn3,rl3,h00_p1_k10,h11_p1_k10`。
- `phase1_with_direct`：在锁定组前加入`d0`，共6项。
- `phase1_rel_p2`：`s0,rn2,rl2`。
- `phase1_all`：`d0`、第一阶段锁定组及P2 Rel组，共8项。
- `phase2_h2_ablation`：`s0,h00,h10,h01,h11`。
