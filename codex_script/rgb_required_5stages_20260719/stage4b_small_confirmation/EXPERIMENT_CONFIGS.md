# Stage 4B 详细实验配置

## 数据与模型

| 项目 | 配置 |
|---|---|
| 模态 | RGB |
| 摄像头 | 00143 |
| 划分 | N as test |
| 训练/验证清单 | `N_as_test/train_manifest_except_take_put.jsonl` / `val_manifest_except_take_put.jsonl` |
| 类别数 | 15 |
| 输入 | 16 帧，224×224 |
| 骨干 | ResNet3D-18，current temporal mode |
| 微调 | full fine-tune；backbone LR=0.0003；head LR=0.001 |

## 共同预训练参数

| 参数 | 值 |
|---|---:|
| epochs / batch size | 200 / 64 |
| optimizer / LR / weight decay | AdamW / 0.001 / 0.0001 |
| LR milestones | 50, 100, 150 |
| queue / projection dim | 1088 / 128 |
| SupCon temperature | 0.07 |
| recluster interval | 10 |
| prototype temperature / EMA | 0.07 / 0.99 |

## 预训练增强

| 增强 | 参数 |
|---|---|
| Temporal sampling | shared，minimum overlap 1.0 |
| Random resized crop | scale 0.85–1.0，ratio 0.9–1.1 |
| Horizontal / vertical flip | p=0.5 / 0 |
| Color jitter | p=0.2；brightness/contrast/saturation=0.1，hue=0.02 |
| Grayscale | p=0 |
| Gaussian blur | p=0.1；kernel=5；sigma=0.1–1.0 |

## 默认确认矩阵

| Index | ID | Seed | 核心配置 |
|---:|---|---:|---|
| 0–1 | rel_r0_s2/s3 | 2/3 | SupLoss-only，3-prototype matched control |
| 2–3 | rel_r9_s2/s3 | 2/3 | diff-only，K3，start125，EMA0.5，λrel=0.5 |
| 4–5 | rel_r12_s2/s3 | 2/3 | diff-only，K3，start125，EMA0.9，λrel=0.5 |
| 6–8 | rel_null_s1/s2/s3 | 1/2/3 | R9 计算路径；prototype state 从50开始，relation 从125开始，λrel=0 |
| 9–10 | proto_p0_s2/s3 | 2/3 | SupLoss-only，1 prototype |
| 11–12 | proto_p4_s2/s3 | 2/3 | soft-positive，P2，λproto=0.1，start50 |
| 13–14 | proto_p6_s2/s3 | 2/3 | all-positive，P3，λproto=0.1，start50 |
| 17–19 | proto_null_p2_s1/s2/s3 | 1/2/3 | P4 完全匹配路径；soft-positive，P2，start50，λproto=0 |
| 20–22 | proto_null_p3_s1/s2/s3 | 1/2/3 | P6 完全匹配路径；all-positive，P3，start50，λproto=0 |

Null-P2 / Null-P3 必须使用 `contrastive_proto`，不能使用 `contrastive_only`。这样才会保留 prototype state、聚类、refresh、EMA 和 ProtoLoss 计算路径，同时将 ProtoLoss 对总损失的梯度贡献置零。

## 可选 R7

| Index | ID | Seed | 核心配置 |
|---:|---|---:|---|
| 15–16 | rel_r7_s2/s3 | 2/3 | diff-only，K10，start50，EMA0.5，λrel=0.5 |

## 微调参数

| 参数 | 值 |
|---|---:|
| epochs / batch size | 100 / 64 |
| optimizer / weight decay | AdamW / 0.0001 |
| backbone LR / head LR | 0.0003 / 0.001 |
| LR milestones | 50, 75 |
| crop / horizontal flip | scale 0.85–1.0，ratio 0.9–1.1 / p=0.5 |
| jitter / gray / blur | disabled |
| primary selection metric | validation Balanced Accuracy |

机器实际使用的唯一配置源是 `config/confirmation_plan.json`；本文用于人工核对。

## 增强汇总输出

| 文件 | 内容 |
|---|---|
| `confirmation_validation_runs.csv` | 每个运行的 best/final/last-10 validation 指标 |
| `confirmation_family_summary.csv` | 三 seed family 均值和标准差 |
| `confirmation_paired_deltas.csv` | 每个 seed 的严格配对差 |
| `confirmation_pair_summary.csv` | 配对差均值、标准差、正向 seed 数 |
| `confirmation_per_class_runs.csv` | 最佳 BA epoch 的逐类 recall |
| `confirmation_per_class_pair_summary.csv` | 各 comparison 的逐类 recall 配对变化 |
| `confirmation_pretrain_diagnostics.csv` | 预训练 loss、非零率、non-finite、assignment、dead/near-dead 和 prototype cosine |
| `confirmation_summary.md` | 面向决策的完整汇总与进入 Stage 5 的判据 |
