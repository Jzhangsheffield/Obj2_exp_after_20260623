# Stage 1 — ProtoLoss v2 mechanism screen

目的：用 seed 1 分解 teacher soft target、类内 Sinkhorn 平衡、prototype diversity 约束各自的作用。

运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage1_proto_screen/slurm"
bash submit_pipeline.sh
```

5 个预训练任务会自动接 5 个 full fine-tuning 任务，再生成 validation-only 汇总。主要比较 P0–P4；同时查看 assignment entropy、balance、diversity，而不是只选单 seed 的最高准确率。此阶段不运行锁定测试。
