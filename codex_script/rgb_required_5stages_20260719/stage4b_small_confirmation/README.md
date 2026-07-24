# Stage 4B：Prototype / Relation 小规模确认实验

本阶段是在 Stage 1 和 Stage 4 单 seed 筛选之后增加的确认实验。它不修改原有 Stage 0、T3、Stage 1、Stage 4、Stage 5 或 Stage 7 的任何文件。

## 目标

1. 判断 R9 和 R12 的验证集提升能否在 seed 2、3 复现。
2. 用 Null-rel 区分真正的 relation 梯度贡献与 prototype refresh / EMA / 数值轨迹带来的影响。
3. 用 Null-P2 / Null-P3 区分真正的 ProtoLoss 梯度贡献与 prototype state / refresh / EMA 路径带来的影响。
4. 判断 P4 和 P6 是否在三 seed 均值和逐 seed 配对差上优于各自匹配的 Null-proto 与 P0。
5. 在结果确认前，不进入原 Stage 5，不使用测试集。

## 默认实验规模

- 21 个必做预训练任务，数组 index 为 `0-14,17-22`。
- 21 个对应的 full fine-tune，数组 index 为 `0-14,17-22`。
- R7 是可选扩展，另外包含 2 个预训练和 2 个微调，index 为 `15-16`。
- 已有 seed 1 结果不会重新训练；汇总脚本会从原 Stage 1/4 结果目录读取。
- Null-rel 此前没有 seed 1，因此本阶段为它运行 seed 1、2、3。
- Null-P2 和 Null-P3 此前没有结果，因此本阶段均运行 seed 1、2、3。

## 最重要的 Null-rel 定义

Null-rel 使用：

```text
ablation_mode = contrastive_rel
num_prototypes = 3
proto_start = 50
rel_start = 125
preview_ema_momentum = 0.5
rel_same_weight = 0
rel_diff_weight = 1
rel_topk_diff_classes = 3
lambda_rel = 0
```

这里不能改成 `contrastive_only`。`contrastive_rel + lambda_rel=0` 会保留 R9 的 prototype refresh、prototype state 和 relation 计算路径，但 relation 对总损失的加权贡献严格为零。按现有训练器的阶段调度，prototype state/refresh 从 epoch 50 开始，relation 计算路径从 epoch 125 开始。

## Null-proto 定义

Null-P2 使用：

```text
ablation_mode = contrastive_proto
proto_positive_mode = soft
num_prototypes = 2
proto_start = 50
lambda_proto = 0
lambda_rel = 0
```

Null-P3 使用：

```text
ablation_mode = contrastive_proto
proto_positive_mode = all
num_prototypes = 3
proto_start = 50
lambda_proto = 0
lambda_rel = 0
```

二者都不能改成 `contrastive_only`。`contrastive_proto + lambda_proto=0` 会保留与 P4/P6 一致的 prototype 聚类、refresh、state、EMA 和 ProtoLoss 计算路径，但 ProtoLoss 对总梯度的加权贡献严格为零。

## 输出目录

```text
results/
├── cl_rgb_req_s4b_confirm_20260724/
│   ├── rel_r0_s2/
│   ├── rel_r0_s3/
│   ├── rel_r9_s2/
│   ├── proto_null_p2_s1/
│   ├── proto_null_p2_s2/
│   ├── proto_null_p2_s3/
│   ├── proto_null_p3_s1/
│   ├── proto_null_p3_s2/
│   ├── proto_null_p3_s3/
│   ├── ...
│   └── rel_r7_s3/                 # 只有运行可选扩展后才存在
└── ft_rgb_req_s4b_confirm_20260724/
    ├── weights/
    ├── datamaps/
    ├── analysis/
    │   ├── confirmation_validation_runs.csv
    │   ├── confirmation_family_summary.csv
    │   ├── confirmation_paired_deltas.csv
    │   ├── confirmation_pair_summary.csv
    │   ├── confirmation_per_class_runs.csv
    │   ├── confirmation_per_class_pair_summary.csv
    │   ├── confirmation_pretrain_diagnostics.csv
    │   └── confirmation_summary.md
    └── test/                       # 当前阶段不要生成
```

## 第一次运行

从项目根目录执行：

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
module load Anaconda3/2022.05
source activate pytorch
export PROJECT_ROOT=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
export DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
```

先检查配置：

```bash
"${CONDA_PREFIX}/bin/python" \
  codex_script/rgb_required_5stages_20260719/common/validate_package.py \
  --config codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/config/stage_config.json
```

查看完整任务表：

```bash
"${CONDA_PREFIX}/bin/python" \
  codex_script/rgb_required_5stages_20260719/common/show_plan.py \
  --config codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/config/stage_config.json
```

## 推荐运行顺序

提交默认的 21+21 个确认任务：

```bash
bash codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/slurm/submit_pipeline.sh
```

脚本会依次提交：

1. `01_pretrain_required_array.slurm`：index `0-14,17-22`；
2. `02_finetune_required_array.slurm`：等待全部预训练成功；
3. `03_summarize_validation.slurm`：等待全部微调成功，并合并原 seed 1。

只有预算充足时再运行 R7：

```bash
bash codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/slurm/submit_optional_r7.sh
```

如果部分数组任务失败，修复问题后直接对失败 index 重新提交即可。例如：

```bash
sbatch --array=2,7,12 \
  codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/slurm/01_pretrain_required_array.slurm
```

预训练运行器会从最新 `checkpoint_*.pth` 继续；已经完成的微调目录存在 `last.pth` 时会自动跳过。

## 如何读取结果

默认流水线完成后查看：

```text
results/ft_rgb_req_s4b_confirm_20260724/analysis/confirmation_summary.md
```

增强汇总包含：

- 每个 seed 的 best/final/last-10 validation 指标；
- 三 seed family 均值和样本标准差；
- R9−R0、R9−Null-rel、P4−Null-P2、P6−Null-P3 等严格逐 seed 配对差；
- 最佳 BA epoch 的逐类别 recall 和逐类配对变化；
- 预训练 SupLoss、ProtoLoss、RelLoss、非零比例、非有限值；
- 最终 prototype assignment、dead/near-dead、assignment entropy 和 prototype cosine；
- 同配置 `rel_r0` 与 `proto_p0` 的重复性审计。

需要手动重新汇总时：

```bash
"${CONDA_PREFIX}/bin/python" \
  codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/scripts/summarize_confirmation.py \
  --project-root "$PROJECT_ROOT"
```

默认会逐个读取最终 checkpoint 计算 prototype 几何。如果只想快速预览已有微调和日志结果，可加：

```bash
"${CONDA_PREFIX}/bin/python" \
  codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/scripts/summarize_confirmation.py \
  --project-root "$PROJECT_ROOT" \
  --skip-checkpoints
```

## 进入 Stage 5 的判据

1. 所有必做 family 都完成三个 seed。
2. 候选的三 seed平均 best BA 与 last-10 BA 均高于匹配对照。
3. 提升至少在 2/3 seed 为正，不是单个 seed 驱动。
4. R9/R12 明显高于 Null-rel，才能把提升归因于 relation 梯度。
5. P4 明显高于 Null-P2、P6 明显高于 Null-P3，才能把提升归因于 ProtoLoss 梯度。
6. 若 Null-proto 高于 P0 而候选与 Null-proto 接近，则收益来自 prototype state/refresh 路径，不来自 ProtoLoss。
7. 提升不能只来自 support 很小的单一类别。
8. 若同配置 `rel_r0` 与 `proto_p0` 的重复波动和候选增益同量级，应先处理确定性问题。

## 测试集锁

本阶段只负责验证确认。`90_test_array.slurm` 保留用于最终模型完全冻结后的完整性，但当前不要提交。测试脚本仍要求显式设置 `ALLOW_LOCKED_TEST=YES`。

Stage 5 的配置应等本阶段三 seed 结果完成后再决定。
