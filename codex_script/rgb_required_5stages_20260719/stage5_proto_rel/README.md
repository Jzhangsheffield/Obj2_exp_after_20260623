# Stage 5：Proto + Rel 组合

目的：在已隔离机制的前提下，检查两种新损失是否互补。固定默认 soft-p2、diff-only top3、preview EMA 0.9，再比较单项消融、同时/分阶段启动及权重灵敏度。

运行：

```bash
bash slurm/submit_pipeline.sh
```

矩阵：12 个预训练 + 自动生成的 12 个 full fine-tune。输出为 `results/cl_rgb_req_s5_20260719` 和 `results/ft_rgb_req_s5_20260719`。

判据：`c1_proto_only > c0_sup`、`c2_rel_only > c0_sup` 是单项有效证据；`c4_p50_r125` 等组合必须进一步超过最强单项才算互补。若组合不如单项，重点检查 loss 梯度方向冲突、rel 启动时 prototype 是否稳定，以及 λproto/λrel 的有效贡献比例。

