# MViT Proto/Rel 统一后续实验包

> 2026-08-13更新：新实验统一使用`run_unified.py`，同时支持15类最小确认、含take/put的17类任务、对比学习增强筛选、类别采样诊断、被试级开发验证以及无验证集的50轮最终重训练。详细配置见[UNIFIED_EXPERIMENT_CONFIGS.md](./UNIFIED_EXPERIMENT_CONFIGS.md)，完整命令见[UNIFIED_USAGE.md](./UNIFIED_USAGE.md)。旧`run_confirmation.py`和旧脚本继续保留，只用于历史复现。

新实验默认输出到：

```text
results/rgb_mvit_pr_unified_followup_20260813
```

最重要的协议约束：增强和Loss筛选必须使用`subject_dev + evaluate`；配置锁定后才允许使用`final_refit + test`，最终测试固定读取`epoch_050.pth`。

---

## 以下为旧版确认runner说明（历史复现）

这是在原实验包上新增的独立确认运行器。原Stage 1–7、历史结果和原配置均未修改。完整简称含义见 `LOCKED_CONFIG_PARAMETER_REFERENCE.md`。

## 能力

- 任意选择一个或多个配置、随机种子和LOSO测试对象。
- 一次提交全部任务，或只运行pretrain、finetune、test中的某些阶段。
- 为每个样本保存标签、预测、完整logits、完整概率、光照与对象元数据。
- 按相同fold、seed和sample ID严格比较Active与Null。
- 同时对相同fold/seed的候选配置与`s0`进行配对比较；选择`d0`时也输出相对direct baseline的差值。
- 汇总多fold/seed均值、标准差、置信区间、paired bootstrap、McNemar、逐类和逐环境结果。

## 第一阶段规模

默认`phase1_locked`有5项：`s0,rn3,rl3,h00_p1_k10,h11_p1_k10`。

如果需要同期复现“无对比预训练、K400直接微调”的`d0`，把配置组改为`phase1_with_direct`。`d0`不是随机初始化true scratch。

| 选择 | 独立训练运行数 |
|---|---:|
| 1对象 × 1 seed | 5 |
| 1对象 × 2 seeds | 10 |
| M/J/N × 2 seeds | 30 |
| M/J/MR/N × 2 seeds | 40 |

每个运行包括一次对比预训练和一次微调。MR已参与历史筛选，所以默认使用`M,J,N`作为新确认对象；MR仅适合补充重复性。

## 运行前

本包复用原包已审核的split：

```text
results/rgb_mvit_pr_env_loso_20260810/runtime/splits
```

如果缺少`protocol_audit.json`，先按原包说明手动执行一次`01_prepare`。

Windows检查：

```bat
confirmation_runner\scripts\windows\01_validate.bat
```

HPC检查：

```bash
python confirmation_runner/run_confirmation.py validate --platform hpc
```

## Stanage运行

先进入：

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/codex_script/rgb_mvit_proto_rel_env_loso_20260810
```

推荐先运行一个对象、一个seed：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs phase1_locked --seeds 2 --subjects J --name p1_j_s2
```

确认完整输出后补seed 3：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs phase1_locked --seeds 3 --subjects J --name p1_j_s3
```

再运行其他对象：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs phase1_locked --seeds 2,3 --subjects M,N --name p1_mn_s23
```

也可以一次提交全部30个运行：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs phase1_locked --seeds 2,3 --subjects M,J,N \
  --name phase1_all30 --max-parallel 4
```

自定义配置和阶段：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs s0,rn3,rl3 --seeds 2,3 --subjects M,J \
  --stages pretrain,finetune,test,summarize --name rel_p3_mj
```

若已有预训练权重，只补微调和测试：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs rn3,rl3 --seeds 2 --subjects N \
  --stages finetune,test,summarize --name rel_p3_n_fttest
```

默认断点续跑；完整checkpoint或测试结果存在时跳过。只查看提交内容：

```bash
bash confirmation_runner/scripts/slurm/submit_selected.sh \
  --configs phase1_locked --seeds 2 --subjects J --name check_j --dry-run
```

dry-run会生成manifest，但不会提交作业。

## Windows运行

顺序运行一个小组：

```bat
confirmation_runner\scripts\windows\03_run_selected.bat --configs s0,rn3,rl3 --seeds 2 --subjects J --name win_check --phases pretrain,proto-env,finetune,test
```

仅生成manifest：

```bat
confirmation_runner\scripts\windows\02_build_manifest.bat --configs phase1_locked --seeds 2,3 --subjects M,J,N --name phase1_all30
```

随后按manifest运行：

```bat
confirmation_runner\scripts\windows\04_run_manifest.bat "D:\Junxi_data\Obj2_experiments_after_260623\results\rgb_mvit_pr_confirm_20260812\manifests\phase1_all30.csv"
```

## 输出目录

```text
results/rgb_mvit_pr_confirm_20260812/
├── manifests/phase1_all30.csv
├── run_meta/fold_J/rl3/s2/resolved_config.json
├── pretrain/fold_J/rl3/s2/
├── finetune/fold_J/rl3/s2/
├── test/fold_J/rl3/s2/
│   ├── test_results.csv
│   ├── best_val_balanced_per_sample_test.csv
│   ├── best_val_balanced_test_metrics.json
│   └── predictions.csv
└── analysis_confirmation/
    ├── overall_runs.csv
    ├── config_mean_std_ci.csv
    ├── config_by_fold_seed_summary.csv
    ├── paired_run_differences.csv
    ├── paired_fold_seed_summary.csv
    ├── paired_difference_summary.csv
    ├── paired_bootstrap.csv
    ├── mcnemar_exact.csv
    ├── per_class_metrics.csv
    ├── per_environment_metrics.csv
    ├── prediction_flips_by_class.csv
    ├── analysis_audit.json
    └── CONFIRMATION_STATISTICAL_REPORT.md
```

短路径`fold_J/rl3/s2`用于避免Windows路径过长；完整参数保存在`resolved_config.json`。

## 汇总与失败重跑

完整HPC提交会自动汇总。手动汇总：

```bash
python confirmation_runner/tools/analyze_confirmation.py \
  --results-root /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/results/rgb_mvit_pr_confirm_20260812 \
  --manifest /path/to/phase1_all30.csv
```

检查缺失测试并生成重跑manifest：

```bash
python confirmation_runner/tools/inspect_missing_runs.py \
  --manifest /path/to/phase1_all30.csv --phase test
```

## 统计边界

- seed先在同一fold内汇总，跨对象结论以fold为主要重复单位；不能用样本量冒充独立实验数。
- paired bootstrap在同一fold/seed中按真实类别分层，对相同样本重采样。
- McNemar检验配对正确性变化，主要用于解释Accuracy。
- BA和Macro-F1结合paired bootstrap与跨fold/seed方向判断。
- 单一对象改善不足以证明损失有效；建议至少3/4 fold方向一致。
- MR参与过历史选择，最终结论应优先依据M/J/N的新结果。
