# Stage 4：Relation loss 机制

目的：完全关闭 prototype contrastive loss，单独判断 rel loss 是否有效，以及现有 same-class 项、全类别 diff 项、过早启动和 preview EMA 0.5 是否造成信号过弱或噪声过大。

运行：

```bash
bash slurm/submit_pipeline.sh
```

矩阵：17 个预训练 + 自动生成的 17 个 full fine-tune。输出为 `results/cl_rgb_req_s4_20260719` 和 `results/ft_rgb_req_s4_20260719`。

推荐阅读顺序：R0/R1 判断当前 rel；R2–R4 判断 hard-negative top-k；R5–R7 判断 diff-only；R8–R10 判断启动时间；R11–R13 判断 preview EMA；R14–R16 判断 λrel。只有 rel-only 能稳定超过 R0，才把它带入 Stage 5；否则将 rel 视为需要重写，而不是继续放大权重。

