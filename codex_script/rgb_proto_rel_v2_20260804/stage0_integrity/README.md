# Stage 0 — integrity audit

目的：验证 V2 运行时源码注入可解析，并确认 `lambda=0` 的 Proto/Rel 分支不会改变 SupLoss-only 的模型更新。

运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage0_integrity/slurm"
bash submit_pipeline.sh
```

共 3 个 seed-1、60-epoch 预训练任务，不进行微调或测试。完成后先读 `results/cl_rgb_v2_s0_integrity_20260804/analysis/null_path_audit.md`。若出现非零权重差异，停止后续阶段并检查随机数消耗、prototype 初始化和恢复点；不要把该差异误认为损失效果。
