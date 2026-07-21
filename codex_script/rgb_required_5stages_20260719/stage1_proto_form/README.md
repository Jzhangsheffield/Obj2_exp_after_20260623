# Stage 1：Prototype loss 形式

目的：验证当前 all-positive 是否会强迫样本同时靠近同类全部 prototype，从而抵消多 prototype 的意义。固定 A2 shared、seed 1、λproto 0.1、epoch 50 启动，只改变 positive 定义和 prototype 数。

运行：

```bash
bash slurm/submit_pipeline.sh
```

矩阵：8 个预训练 + 8 个微调。输出为 `results/cl_rgb_req_s1_20260719` 和 `results/ft_rgb_req_s1_20260719`。

决策：先排除明显低于 p0 SupLoss 的形式；再看 p2/p3 是否稳定优于 p1。若 all 随 prototype 数增大而变差而 soft/single 不变差，说明主要不足是正样本定义。若三者都无提升，优先检查聚类质量、assignment 覆盖率、prototype 使用率和 λproto 的梯度占比。

