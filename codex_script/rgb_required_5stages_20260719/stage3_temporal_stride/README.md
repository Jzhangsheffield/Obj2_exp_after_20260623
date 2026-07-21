# Stage T3：LFB式时间下采样消融

## 目的

验证当前 ResNet3D-18 将16帧压缩为1个深层时间位置，是否是 RGB 分类性能受限的重要原因。本阶段只改变时间 stride，不改变数据、空间增强、SupLoss、训练轮数、优化器或模型参数形状。

## T3结构

当前结构：

```text
input 16 → conv1 16 → maxpool 8 → layer1 8 → layer2 4 → layer3 2 → layer4 1
```

T3结构：

```text
input 16 → conv1 16 → maxpool 16 → layer1 16 → layer2 8 → layer3 8 → layer4 8
```

具体修改：

- `maxpool` 改为 kernel `(1,3,3)`、stride `(1,2,2)`、padding `(0,1,1)`；
- `layer2` 保留 `(2,2,2)`，作为唯一一次时间下采样；
- `layer3`、`layer4` 的首个 block 及 shortcut 改为 `(1,2,2)`；
- 最终 `AdaptiveAvgPool3d((1,1,1))` 保持不变。

这对应 LFB 的“总时间下采样倍率2”思想，但属于适配当前 BasicBlock ResNet3D-18 的版本，不是原论文 ResNet-50 I3D-NL 的逐层复制。

## 实验矩阵

预训练3项：`t3_sup_s1`、`t3_sup_s2`、`t3_sup_s3`。全部使用 A2 shared temporal views 和 SupLoss-only。

微调6项：

- `scratch_t3_s1/s2/s3`：T3 backbone 从随机初始化训练；
- `t3_sup_s1_ft/s2_ft/s3_ft`：加载对应 T3 SupLoss checkpoint，backbone LR `3e-4`。

公平对照使用 Stage 0：

| 当前T0 | T3 |
|---|---|
| `scratch_s1_a/s2/s3` | `scratch_t3_s1/s2/s3` |
| `sup_s1_a_ft/s2_ft/s3_ft` | `t3_sup_s1_ft/s2_ft/s3_ft` |

不要用 `sup_s1_b` 参与三seed配对；它是 Stage 0 的 same-seed复现副本。

## 提交

```bash
bash slurm/submit_pipeline.sh
```

输出：

```text
results/cl_rgb_req_t3_20260721/
results/ft_rgb_req_t3_20260721/
```

T3保留更多中间激活，因此预训练时限提高到24小时、微调提高到18小时。若显存不足，先记录峰值显存，不要单独修改某个seed的batch size；应统一调整本阶段全部任务后重跑。

## 判断标准

分别计算：

```text
Δscratch = mean(T3 scratch) - mean(T0 scratch)
Δsup     = mean(T3 SupLoss) - mean(T0 SupLoss)
```

- 两者都稳定为正：强烈支持时间过度压缩限制了性能；
- 只有 `Δsup` 为正：保留时间维可能主要改善对比预训练表征；
- 只有 `Δscratch` 为正：预训练目标或增强可能没有利用新增时间分辨率；
- 两者都不提升：时间压缩不是当前主要瓶颈，或 T3 计算/优化代价抵消了收益。

主要指标仍为 validation Balanced Accuracy。最终测试只能在所有决定冻结后通过测试锁执行。

