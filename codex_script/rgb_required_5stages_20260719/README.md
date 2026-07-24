# RGB 必做五阶段实验包 + T3时间下采样消融

本包原有五个阶段保持不变，并于2026-07-21增加 `stage3_temporal_stride`，于2026-07-24增加 `stage4b_small_confirmation`。目录名继续保留 `rgb_required_5stages_20260719`，避免破坏已有脚本路径。

## 目录和执行顺序

1. `stage0_repro_baseline`：确认 00143 相机 scratch 与 SupLoss-only 基线可复现，量化同 seed 重复误差和跨 seed 方差。
2. `stage3_temporal_stride`：比较当前 `16→1` 与T3 `16→8`时间压缩，运行T3 scratch和T3 SupLoss三seed。
3. `stage1_proto_form`：比较 single / all / soft 三种 prototype 正样本定义和 1/2/3 prototype 数。
4. `stage4_rel_mechanism`：关闭 proto loss，只研究 rel loss 的同类项、异类项、top-k、启动时间、preview EMA 和权重。
5. `stage4b_small_confirmation`：用三 seed、Null-rel、Null-P2 和 Null-P3 对 R9/R12、P4/P6 做完整因果确认。
6. `stage5_proto_rel`：只在 Stage 4B 确认单项有效后，组合 SupLoss + proto + rel。
7. `stage7_finetune`：固定候选预训练权重，比较 head-only（线性探测）和 full fine-tune 的 backbone LR，最后一次性锁定测试集。

所有详细参数以 [ALL_EXPERIMENT_CONFIGS.md](ALL_EXPERIMENT_CONFIGS.md) 和 `common/experiment_plan.json` 为准。JSON 是运行器实际读取的唯一配置源，Markdown 是便于核对的说明。

## 第一次使用

把整个项目同步到集群项目根目录，然后进入本包：

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/codex_script/rgb_required_5stages_20260719
module load Anaconda3/2022.05
source activate pytorch
export PROJECT_ROOT=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
export DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
```

先做静态检查；把 `stage0_repro_baseline` 换成其它阶段即可逐个检查：

```bash
"${CONDA_PREFIX}/bin/python" common/validate_package.py \
  --config stage0_repro_baseline/config/stage_config.json
```

在集群训练环境下可增加 `--commands`，它会执行 parse-only 检查，不会开始训练或创建结果：

```bash
"${CONDA_PREFIX}/bin/python" common/validate_package.py \
  --config stage0_repro_baseline/config/stage_config.json --commands
```

## 提交一个阶段

每个阶段的 `submit_pipeline.sh` 只自动提交预训练和依赖它的微调，不会自动测试：

```bash
bash stage0_repro_baseline/slurm/submit_pipeline.sh
bash stage3_temporal_stride/slurm/submit_pipeline.sh
bash stage4b_small_confirmation/slurm/submit_pipeline.sh
```

训练中断后，重新提交相同脚本即可。预训练会选择输出目录中最新的 `checkpoint_*.pth` 续训；微调入口会选择最新的 `epoch_*.pth`。完整的最终 checkpoint 或 `last.pth` 已存在时会跳过。

## 结果选择与测试集锁

在调参阶段只用验证集 Balanced Accuracy。可生成验证选择清单：

```bash
"${CONDA_PREFIX}/bin/python" common/select_best.py \
  --config stage1_proto_form/config/stage_config.json
```

Stage 7 前，编辑 `stage7_finetune/config/selected_sources.json`，核对T3路径，并将 proto、rel、proto_rel 路径换成 Stage 1/4/5 根据验证集选出的预训练 checkpoint。随后提交 Stage 7：

```bash
bash stage7_finetune/slurm/submit_pipeline.sh
```

只有所有模型、学习率和 checkpoint 选择已经冻结后，才解锁测试。假设 Stage 7 微调作业号为 `123456`：

```bash
sbatch --export=ALL,ALLOW_LOCKED_TEST=YES --dependency=afterok:123456 \
  stage7_finetune/slurm/03_test_array.slurm
```

测试作业完成后，假设测试数组作业号为 `123789`：

```bash
sbatch --dependency=afterok:123789 stage7_finetune/slurm/04_summarize.slurm
```

最终汇总位于：

`results/ft_rgb_req_s7_20260719/test/rgb_test_results_ranked.csv`

## 单独运行和排错

打印某个实验的完整命令但不运行：

```bash
"${CONDA_PREFIX}/bin/python" common/run_pretrain.py \
  --config stage1_proto_form/config/stage_config.json --index 4 --dry-run
```

只重跑一个微调任务：

```bash
"${CONDA_PREFIX}/bin/python" common/run_finetune.py \
  --config stage7_finetune/config/stage_config.json --index 10
```

查看自动展开后的完整任务表（包括 Stage 4/5/7 的微调 index）：

```bash
"${CONDA_PREFIX}/bin/python" common/show_plan.py \
  --config stage7_finetune/config/stage_config.json
```

每次训练会在输出目录写 `codex_run_provenance.json`，记录配置、完整命令、Git commit、训练入口 SHA256、Slurm 作业号和 GPU 可见性。预训练还写 `required_wrapper_args.json` 和 `debug_train_log.jsonl`。

## 重要约束

- 不要在 Stage 0/T3/1/4/5 用测试集筛选超参数；这些阶段的 test 脚本仅为完整性保留。
- Stage 4B 的必做数组是 `0-14,17-22`；`15-16` 为可选 R7。
- Stage 4B 只有在 R9 高于 Null-rel、P4/P6 高于各自 Null-proto 后，才能把增益归因于新损失。
- 正式测试只寻找 `best_val_balanced.pth`，不会退回 `best_val.pth` 或 `last.pth`。
- Stage 7 的 `head_only` 就是本项目当前实现能够支持的线性探测。
- 若集群路径不同，只需在提交前设置 `PROJECT_ROOT` 与 `DATASET_ROOT`，无需修改 JSON。
- 如果 GPU 型号变化，同 seed 仍可能存在微小数值差异；Stage 0 的 same-seed 两次重复就是用来测量这一部分。
