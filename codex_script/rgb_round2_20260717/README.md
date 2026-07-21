# RGB 对比学习第二轮 HPC 实验包

本实验包针对第一轮分析中发现的 RGB 表示问题，依次验证：

1. vertical flip 和强空间/颜色增强是否破坏动作语义；
2. 两个 temporal view 是否需要共享时间索引或保持至少 75% 相同帧；
3. relation loss 是否应推迟到 epoch 50，并降低到 0.25/0.5；
4. 在投影头前 512D 特征上增加辅助交叉熵是否改善类别结构；
5. full 微调时 backbone LR 使用 1e-4 还是 3e-4。

原始 `train/` 和 `ft_and_test/` Python 文件不会被修改。实验入口在内存中加入所需功能；如果原始脚本结构与本机版本不一致，验证程序会安全退出，而不会执行错误补丁。

## 目录内容

- `config/round2_config.json`：路径、9个预训练实验和9个微调任务。
- `rgb_round2_pretrain_entry.py`：时间视图、512D CE 和预训练断点恢复。
- `rgb_round2_finetune_entry.py`：微调周期 checkpoint 恢复。
- `run_pretrain.py`：运行单个预训练配置。
- `run_finetune.py`：运行单个 full 微调配置。
- `run_test.py`：测试所有已完成的 best_val 权重。
- `summarize_results.py`：生成按 balanced accuracy 排序的 CSV。
- `validate_package.py`：配置、时间采样和代码注入检查。
- `slurm/`：按阶段提交的 Slurm 文件。

## 实验矩阵

预训练数组索引：

| Index | ID | 主要变化 |
|---:|---|---|
| 0 | a0_current_independent_supcon | 当前强增强、独立时间视图、SupCon-only |
| 1 | a1_no_vflip_independent_supcon | 只关闭 vertical flip |
| 2 | a2_weak_shared_supcon | 弱增强、两个视图使用相同帧 |
| 3 | a3_weak_overlap75_supcon | 弱增强、至少75%相同帧 |
| 4 | b1_weak_overlap75_rel025_all | A3 + 延迟 relation，lambda=0.25 |
| 5 | b2_weak_overlap75_rel050_all | A3 + 延迟 relation，lambda=0.5 |
| 6 | b3_weak_overlap75_rel050_topk3 | B2，但不同类只约束 top-3 |
| 7 | c1_weak_overlap75_rel050_ce050 | B2 + 512D CE，权重0.5 |
| 8 | c2_weak_overlap75_rel050_ce100 | B2 + 512D CE，权重1.0 |

预设微调只覆盖 scratch、A2、A3、B2 和 C1；每个预训练候选比较 backbone LR 1e-4/3e-4。scratch 使用全网络 LR 1e-3，避免把随机初始化 backbone 的学习率压得过低而人为削弱基线。RGB 不再把 head-only 作为主要实验。

## HPC 使用步骤

### 1. 上传并检查路径

将整个项目同步到：

```text
/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
```

数据集默认位于：

```text
/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
```

如果路径不同，修改 `config/round2_config.json`，或提交时设置 `PROJECT_ROOT`、`DATASET_ROOT` 环境变量。

提交前保证 Slurm 日志目录已经存在：

```bash
mkdir -p /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/train/out
mkdir -p /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out
```

### 2. 登录节点验证

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python codex_script/rgb_round2_20260717/validate_package.py
python codex_script/rgb_round2_20260717/run_pretrain.py --index 2 --dry-run
python codex_script/rgb_round2_20260717/run_finetune.py --index 1 --dry-run
```

只有看到 `Round-2 package validation: OK` 后再提交训练。

### 3. 第一阶段：视图消融

```bash
sbatch codex_script/rgb_round2_20260717/slurm/01_pretrain_view_array.slurm
```

该作业运行 index 0–3，同时最多2张 GPU。建议先完成这一阶段，再决定是否继续全部目标函数实验。

预训练权重输出到：

```text
results/cl_rgb_round2_action_preserving_20260717/<experiment_id>/
```

每10个 epoch 保存一个 checkpoint。重新提交同一个任务时会自动读取编号最大的 checkpoint；如果 `checkpoint_0200.pth` 已存在，则直接跳过。

### 4. 对第一阶段候选进行 full 微调

```bash
sbatch codex_script/rgb_round2_20260717/slurm/03_finetune_shortlist_array.slurm
```

第一次提交时，尚未完成 B2/C1 的任务会显示 prerequisite missing 并安全退出；scratch、A2 和 A3 会正常运行。微调每10个 epoch 保存一次，重新提交时自动恢复。

微调输出到：

```text
results/ft_rgb_round2_action_preserving_20260717/
```

### 5. 第二阶段：relation 和辅助 CE

如果 A2/A3 相比 A0/A1 有改善，再提交：

```bash
sbatch codex_script/rgb_round2_20260717/slurm/02_pretrain_objective_array.slurm
```

完成后再次提交微调数组。已经存在 `last.pth` 的任务会跳过，只运行新增的 B2/C1 候选：

```bash
sbatch codex_script/rgb_round2_20260717/slurm/03_finetune_shortlist_array.slurm
```

### 6. 测试并排名

默认只测试 `best_val.pth`：

```bash
sbatch codex_script/rgb_round2_20260717/slurm/04_test_all.slurm
```

如需同时测试 `last.pth`：

```bash
sbatch --export=ALL,INCLUDE_LAST=1 codex_script/rgb_round2_20260717/slurm/04_test_all.slurm
```

排名结果位于：

```text
results/ft_rgb_round2_action_preserving_20260717/weights/_batch_test/round2_fixed/summary/rgb_test_results_ranked.csv
```

逐样本预测 CSV 仍保存在每个权重旁边，可继续用于混淆矩阵和 McNemar 检验。

## 断点和重排说明

- Slurm 脚本在时间结束前180秒接收 `USR1`，终止当前 step 并调用 `scontrol requeue`。
- 对比预训练恢复模型、MoCo queue、优化器以及已有 prototype state。
- 微调恢复模型、优化器和 AMP scaler。
- checkpoint 间隔为10 epoch，所以异常终止最多损失最近10个 epoch。
- 如果集群不允许用户自行 requeue，去掉 Slurm 文件中的 `--requeue`、`--signal` 和 trap；手工重新提交同一脚本仍会自动恢复。

## 结果判断标准

建议优先看 balanced accuracy，并保持同一 scratch checkpoint 作为比较对象。进入更大规模实验前，至少要求：

1. A2/A3 的 full 微调优于新的固定 scratch；
2. 配对 McNemar 检验方向为正；
3. CL-512 训练集 silhouette 从负值转为正值；
4. C1 的增益不是仅通过更高 backbone LR 获得。

本轮没有加入跨摄像头正样本。该实验需要先确认两路摄像头严格同步，建议在本轮确定有效的空间/时间增强后再单独实现。
