# RGB ProtoLoss v2 + RelLoss v2 实验包

创建日期：2026-08-04  
位置：`codex_script/rgb_proto_rel_v2_20260804`

## 1. 这个实验包解决什么问题

旧实验已经表明：原 ProtoLoss/RelLoss 对下游性能的提升小且不稳定。本包保持“类内多 prototype + prototype relation”主思路不变，但把两个最可能的失败源改掉：

1. ProtoLoss v2 不再使用固定硬 assignment；改用 momentum teacher 产生的软 assignment，并可在类内执行 Sinkhorn 平衡。
2. prototype bank 不再周期性硬重聚类；只在启动时初始化，随后用 teacher responsibility 做软 EMA 更新。
3. 增加类内 prototype 使用平衡和多样性约束，直接监控 assignment entropy、dead prototype 风险及类内 prototype 相似度。
4. RelLoss v2 不再平均处理所有关系；仅处理每个样本最危险的 top-K 负类，并优化正类 prototype 与 hard-negative class prototype 的 margin。
5. `rank_direction` 额外要求当前 batch 的 preview update 改善困难样本的正负 gap，但仅对违反 margin 的样本生效。

本包是一个全新的目录，不会修改旧包 `rgb_required_5stages_20260719`。微调和测试仍调用旧包中已经验证过的 runner，以减少无关变量。

## 2. 首次使用前必须做的检查

在集群项目根目录执行：

```bash
PROJECT_ROOT=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
CFG="$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage0_integrity/config/stage_config.json"
python "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/common/run_pretrain.py" \
  --config "$CFG" --index 0 --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT" --validate-command
```

看到 `Proto/Rel V2 source patch validation: OK` 才能提交训练。V2 通过运行时注入复用原训练程序；若原文件结构以后发生改变，检查会明确失败，而不是静默使用错误的损失。

## 3. 推荐运行顺序

### Stage 0：实现一致性审计（必须首先运行）

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage0_integrity/slurm"
bash submit_pipeline.sh
```

检查：

- `results/cl_rgb_v2_s0_integrity_20260804/analysis/null_path_audit.md`
- `results/cl_rgb_v2_s0_integrity_20260804/analysis/v2_pretrain_diagnostics.csv`

如果零权重 Proto/Rel 分支与 SupLoss-only 的权重不一致，先检查 RNG、bank 初始化和 resume，不应直接开始 Stage 1。

### Stage 1：ProtoLoss v2 机制筛选

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage1_proto_screen/slurm"
bash submit_pipeline.sh
```

这是单 seed 机制筛选，只回答 teacher、balanced assignment、多样性约束是否有方向性价值。查看：

- `results/ft_rgb_v2_s1_proto_screen_20260804/analysis/validation_summary.md`
- `results/cl_rgb_v2_s1_proto_screen_20260804/analysis/v2_pretrain_diagnostics.csv`

若所有 V2 配置均低于 P0，且出现高 assignment entropy 但 prototype 相似度仍很高，说明多 prototype 没有形成可用子模态；先修机制，不进入大规模确认。

### Stage 2：ProtoLoss v2 三 seed 确认

仅在 Stage 1 找到合理候选后运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage2_proto_confirm/slurm"
bash submit_pipeline.sh
```

主要比较 SupLoss-only、Null-proto 和 ProtoLoss-v2-full 的验证集 balanced accuracy 均值与标准差。Null-proto 用于排除“启用 prototype 代码路径本身”造成的差异。

### Stage 3：RelLoss v2 三 seed确认

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage3_rel_confirm/slurm"
bash submit_pipeline.sh
```

比较 SupLoss-only、Null-rel、hard-negative rank、rank+direction。重点看 validation BA，同时检查 margin violation 是否下降、hard-negative similarity 是否下降。

### Stage 4：2×2 因子确认

只有 ProtoLoss v2 与 RelLoss v2 至少各自通过机制确认时才运行：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage4_factorial/slurm"
bash submit_pipeline.sh
```

四组是 SupLoss、SupLoss+Proto-v2、SupLoss+Rel-v2、SupLoss+Proto-v2+Rel-v2，各 3 seeds。它回答两个损失是互补、冗余还是相互干扰。

## 4. 测试集使用规则

`submit_pipeline.sh` 永远不会提交测试。所有结构和超参数必须只根据训练/验证结果冻结。最终只测试冻结后的候选与 matched SupLoss baseline：

```bash
cd "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage4_factorial/slurm"
export ALLOW_LOCKED_TEST=YES
TEST_JOB=$(sbatch --parsable 90_test_array.slurm)
sbatch --dependency="afterok:${TEST_JOB%%;*}" 91_summarize_test.slurm
```

如果最终只需两个 family，不建议直接运行整个 array；可使用 `sbatch --array=指定索引 90_test_array.slurm`。不要用 Stage 1/2/3 的测试结果选择参数。

## 5. 恢复训练与输出

- checkpoint 每 50 epoch 保存，最终为 `checkpoint_0200.pth`；Stage 0 最终为 `checkpoint_0060.pth`。
- 微调周期 checkpoint 每 25 epoch 保存，即 `epoch_025.pth`、`epoch_050.pth`、`epoch_075.pth`、`epoch_100.pth`；最佳 BA、最佳 Accuracy、最佳 Macro-F1 和最终 `last.pth` 仍照常保存。
- 同一任务重新提交会自动寻找最新 `checkpoint_*.pth` 恢复；若最终 checkpoint 已存在则直接跳过。
- 原项目 prototype 诊断参数仍为每 10 epoch；V2 另外把在线损失/assignment 诊断写到 `v2_diagnostics.jsonl`（每个 epoch 的第一个 step，以及每 50 step）。
- RelLoss 启动前后 checkpoint 继续由原项目的 `rel_checkpoint_after_epochs=10` 机制保存。
- 每次运行会写 `codex_run_provenance.json` 和 `v2_effective_config.json`，用于追溯源码 hash 与有效参数。

## 6. 重要边界

- 所有微调都是 full fine-tuning，不是只训练分类头；backbone LR=`3e-4`，head LR=`1e-3`。
- temporal views 为 shared，16 帧位置一致；本包不混入 T3 的 temporal-stride 改动。
- sampler 保持 `none`，与原 matched SupLoss 设置一致；Sinkhorn 只在当前 batch 内每个已出现类别上运行。
- 当前版本有固定损失权重，尚未加入 GradNorm/自适应权重。先确认损失机制确实有正信号，再做权重自适应，否则会把“机制无效”和“权重不合适”混在一起。
- 完整数学定义、数据增强、路径和全部实验表见 [ALL_EXPERIMENT_CONFIGS.md](ALL_EXPERIMENT_CONFIGS.md)。

## 7. 目录结构

```text
rgb_proto_rel_v2_20260804/
├── README.md
├── ALL_EXPERIMENT_CONFIGS.md
├── common/
│   ├── experiment_plan.json
│   ├── v2_losses.py
│   ├── pretrain_v2_entry.py
│   ├── run_pretrain.py
│   ├── delegate_legacy.py
│   ├── audit_null_paths.py
│   ├── summarize_v2.py
│   └── summarize_validation.py
├── stage0_integrity/{README.md,config/,slurm/}
├── stage1_proto_screen/{README.md,config/,slurm/}
├── stage2_proto_confirm/{README.md,config/,slurm/}
├── stage3_rel_confirm/{README.md,config/,slurm/}
└── stage4_factorial/{README.md,config/,slurm/}
```
