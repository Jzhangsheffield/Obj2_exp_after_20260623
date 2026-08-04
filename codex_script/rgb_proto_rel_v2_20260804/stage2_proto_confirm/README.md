# Stage 2 — ProtoLoss v2 three-seed confirmation

目的：用 seeds 1/2/3 比较 SupLoss、Null-proto 和完整 ProtoLoss v2，排除单 seed 波动及 prototype 代码路径本身的影响。

运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage2_proto_confirm/slurm"
bash submit_pipeline.sh
```

共 9 个预训练 + 9 个 full fine-tuning。主要效应为 `PV2 - PN`，而 `PN - P0` 是实现控制。必须看 validation balanced accuracy 的均值、标准差和逐 seed 方向一致性；测试脚本只保留给最终冻结后的评估，不参与参数选择。
