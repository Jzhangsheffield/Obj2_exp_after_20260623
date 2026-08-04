# Stage 4 — Proto v2 × Rel v2 factorial confirmation

目的：用 seeds 1/2/3 的 2×2 设计测量 Proto 主效应、Rel 主效应以及两者交互。

运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage4_factorial/slurm"
bash submit_pipeline.sh
```

共 12 个预训练 + 12 个 full fine-tuning。只应在 Stage 2 和 Stage 3 都给出正向证据后运行。自动流水线不会运行测试；全部设置冻结后，使用 `ALLOW_LOCKED_TEST=YES` 明确解锁，并优先只测试最终候选与 matched SupLoss family。
