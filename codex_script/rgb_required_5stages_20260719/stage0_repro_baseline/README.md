# Stage 0：复现性与基线

目的：判断 00143 基线大差距是否主要来自训练协议、随机性或结果选择，而不是相机 ID 本身。先做两次完全相同 seed 1，再做 seed 2/3；同时比较 scratch 与 A2 shared SupLoss-only。

重点记录：same-seed 最大差、3-seed Balanced Accuracy 均值±标准差、scratch→SupLoss 的配对提升、是否所有任务都由 `best_val_balanced.pth` 进入最终测试。

运行：

```bash
bash slurm/submit_pipeline.sh
```

矩阵：4 个预训练任务，8 个微调任务。输出为 `results/cl_rgb_req_s0_20260719` 与 `results/ft_rgb_req_s0_20260719`。Stage 0 后若 same-seed 差距仍很大，先检查 GPU 型号、数据清单哈希、源码哈希和 resume 起点，不进入损失调参。

