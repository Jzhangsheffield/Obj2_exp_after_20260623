# Stage 3 — RelLoss v2 three-seed confirmation

目的：比较 Null-rel、hard-negative rank 和 rank+direction，确认 RelLoss 是否真正降低最易混淆负类的相似度与 margin violation，并能转化为下游收益。

运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage3_rel_confirm/slurm"
bash submit_pipeline.sh
```

共 12 个预训练 + 12 个 full fine-tuning。Rel 从 epoch 75 启动，在 epoch 100 达到完整权重并持续到 epoch 200，top-K=3。先看 `RRANK-RN`，再看 `RV2-RRANK`；若 direction 只增加 loss 而不改善 gap，应保留 rank、删除 direction，而不是直接提高 lambda。
