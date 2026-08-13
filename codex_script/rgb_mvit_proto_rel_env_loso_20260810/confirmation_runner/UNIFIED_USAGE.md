# 统一实验包使用说明

## 1. 入口与输出

- 运行入口：`confirmation_runner/run_unified.py`
- Windows脚本：`confirmation_runner/scripts/windows/u*.bat`
- Stanage脚本：`confirmation_runner/scripts/slurm/u*.slurm`
- 主提交脚本：`confirmation_runner/scripts/slurm/submit_unified.sh`
- 结果目录：`results/rgb_mvit_pr_unified_followup_20260813`

旧的`run_confirmation.py`及无`u`前缀脚本仍保留用于历史复现；新实验只使用统一入口。

## 2. 第一次使用：准备manifest

准备过程只读取原数据manifest，不载入RGB张量，也不需要GPU。它会把每个LOSO目录中的完整train/val/test合并、去重并重新生成严格的协议文件。

Windows：

```bat
confirmation_runner\scripts\windows\u00_prepare_unified.bat
confirmation_runner\scripts\windows\u01_validate_unified.bat
```

Stanage上如果数据只存在HPC，则手动运行：

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/codex_script/rgb_mvit_proto_rel_env_loso_20260810
python confirmation_runner/run_unified.py prepare --platform hpc
python confirmation_runner/run_unified.py validate --platform hpc
```

也可以单独提交纯CPU准备脚本：

```bash
sbatch confirmation_runner/scripts/slurm/u00_prepare.slurm
```

`submit_unified.sh`只提交GPU训练/评估和最后汇总，不会自动提交CPU准备任务。这延续了之前“prepare手动运行”的做法。

## 3. 查看可选内容

```bash
python confirmation_runner/run_unified.py list-configs
```

## 4. 推荐运行顺序

### U1a：15类最小确认（推荐先运行）

这组配置已经锁定，可以使用最终协议。先运行N、seed 1的5个配置：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --preset confirm15_min_n_s1 \
  --stages pretrain,finetune,test,summarize \
  --name u1a_confirm15_min \
  --max-parallel 4
```

规模为5次。

### U1b：15类多被试、多seed确认（可选扩展）

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --preset confirm15_final \
  --stages pretrain,finetune,test,summarize \
  --name u1b_confirm15_full \
  --max-parallel 4
```

完整规模40次。也可以用`--subjects N`覆盖预设，只补N的seed 2、3，共10次。

### U2：17类入口与采样诊断

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --preset full17_entry_dev \
  --stages pretrain,finetune,evaluate,summarize \
  --name u2_full17_entry \
  --max-parallel 4
```

共18次微调，其中12次SupLoss预训练；Direct FT无效的预训练采样重复会自动去除。根据三个整人验证折的平均BA、最差折BA以及原15类/take-put子集指标选择一个采样策略。

### U3：SupLoss数据增强筛选

把下例`balanced_pre`替换成U2选出的采样策略：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --preset aug17_s0_screen \
  --samplings balanced_pre \
  --stages pretrain,finetune,evaluate,summarize \
  --name u3_aug_s0 \
  --max-parallel 4
```

共21次。选择A0以外最多两个增强候选；选择时同时检查最差被试和take/put，不按单折最高值选择。

### U4：增强与四类Loss交互

假设U3选中`a2_photo`和`a4_combined`：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --preset aug17_cross_loss \
  --augmentations a0,a2_photo,a4_combined \
  --samplings balanced_pre \
  --stages pretrain,finetune,evaluate,summarize \
  --name u4_aug_loss \
  --max-parallel 4
```

规模为4个Loss×3个增强×3个开发折=36次。

### U5：严格Null确认

假设最终增强为`a2_photo`：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --task t17 --protocol subject_dev \
  --configs augmentation_controls \
  --augmentations a2_photo \
  --samplings balanced_pre \
  --seeds 1 --subjects M,J,MR \
  --stages pretrain,finetune,evaluate,summarize \
  --name u5_null_controls
```

比较`rl3-rn3`和`h11-h00`，共12次。

### U6：锁定后的最终测试

只有完成U2–U5并把所有超参数写入实验记录后才能执行。下面仅为示例，不能把示例候选当成已确定胜者：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --task t17 --protocol final_refit \
  --configs s0,rl3,h11_p1_k10 \
  --augmentations a2_photo \
  --samplings balanced_pre \
  --seeds 2,3 --subjects N \
  --stages pretrain,finetune,test,summarize \
  --name u6_locked_n
```

最终阶段使用全部M+J+MR训练数据、无validation，并只测试epoch 50。看到N结果后不能再改变配置并把同一个N结果称作独立测试。

## 5. 自由选择配置

任何阶段都可以选择一个或多个维度，运行数为各维度笛卡尔积；Direct FT会自动去除对它无效的预训练增强/采样重复。

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --task t17 --protocol subject_dev \
  --configs s0,h10_p1_k10,rl3,h11_p1_k10 \
  --augmentations a0,a2_photo \
  --samplings natural,balanced_pre \
  --seeds 1,2 --subjects M,J \
  --stages pretrain,finetune,evaluate,summarize \
  --name custom_grid
```

也可以只运行后续阶段：

```bash
bash confirmation_runner/scripts/slurm/submit_unified.sh \
  --task t17 --protocol subject_dev \
  --configs s0 --augmentations a0 --samplings natural \
  --seeds 1 --subjects M \
  --stages finetune,evaluate,summarize \
  --name resume_ft_eval
```

前提是相同输出路径下已经存在所需的`checkpoint_0200.pth`。

## 6. Windows运行示例

```bat
confirmation_runner\scripts\windows\u03_run_selected.bat --preset full17_entry_dev --phases pretrain,finetune,evaluate --name win_u2
```

仅构建任务清单而不训练：

```bat
confirmation_runner\scripts\windows\u02_build_manifest.bat --preset aug17_s0_screen --samplings balanced_pre --name u3_manifest
```

随后按已有manifest运行：

```bat
confirmation_runner\scripts\windows\u04_run_manifest.bat D:\path\u3_manifest.csv pretrain,finetune,evaluate
```

## 7. 检查缺失任务与汇总

```bat
confirmation_runner\scripts\windows\u06_inspect_missing.bat D:\path\manifest.csv finetune
confirmation_runner\scripts\windows\u05_summarize.bat D:\path\manifest.csv
```

HPC汇总通常已由`submit_unified.sh`作为最后一个依赖任务自动提交。

## 8. checkpoint规则

对比预训练保存：

```text
checkpoint_0050.pth
checkpoint_0100.pth
checkpoint_0150.pth
checkpoint_0200.pth
```

prototype诊断每10轮单独保存，不需要为了诊断而保存完整模型权重。

微调保存：

```text
epoch_025.pth
epoch_050.pth
last.pth
```

`subject_dev`还会产生best验证权重，但统一统计不用它们。`final_refit`没有验证集，因此没有best权重。`evaluate`和`test`都强制读取`epoch_050.pth`。

## 9. 重要防错规则

- `subject_dev`只能运行`evaluate`，不能运行`test`。
- `final_refit`只能运行`test`，不能运行`evaluate`。
- t15使用15类label map，t17使用17类label map，分类头维度由runner同步设置。
- 不要把旧15类checkpoint加载到17类分类头；对比预训练backbone可加载，但分类微调必须重新开始。
- 输出路径包含task、protocol、对象、Loss、增强、采样和seed，不会覆盖旧实验。
- 每个运行目录保存`resolved_config.json`及三份manifest的SHA256，便于复核。
