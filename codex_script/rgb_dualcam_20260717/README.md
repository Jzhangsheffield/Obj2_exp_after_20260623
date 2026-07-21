# RGB 双摄像头正样本实验包

本实验使用 `N_as_test` 的 except-take-put 划分。每条 manifest 记录同时包含：

- `rgb_cam_00143`
- `rgb_cam_00152`

本机检查结果：训练集1026个样本全部具有两路路径；两路帧数差最大4帧，793/1026个样本差不超过1帧，1018/1026个样本差不超过2帧。

`.pt` 文件没有逐帧时间戳，因此不能证明硬件级严格同步。本实验将同一分段 clip 内的相同归一化时间位置视为对齐位置。例如00143的50%位置与00152的50%位置配对。每次训练先从归一化时间轴随机截取85%–100%的共同时间范围，再映射到两路实际帧号。

## 关键设计

两路相机使用各自的归一化统计。00152统计由 `N_as_test` 训练集估计：每个片段最多均匀抽8帧，空间每8个像素抽一个点。

| Index | ID | 正样本构造 |
|---:|---|---|
| 0 | a0_same143_sup | 两个view都来自00143，严格对照 |
| 1 | a1_same152_sup | 两个view都来自00152 |
| 2 | a2_cross_fixed_sup | query固定00143，key固定00152 |
| 3 | a3_cross_random_sup | 两路相机随机交换query/key角色 |
| 4 | a4_hybrid50_sup | 50%跨相机，50%同相机 |
| 5 | b1_cross_random_rel050 | A3 + epoch50后relation loss |
| 6 | b2_cross_random_rel050_ce050 | B1 + 512D辅助CE，权重0.5 |

所有配置使用相同的弱RGB增强：无垂直翻转、RRC scale 0.85–1.0、ratio 0.9–1.1、轻量颜色扰动和blur。这样主要变量是摄像头正样本，而不是增强强度。

固定顺序A2可以判断“00143作为query时跨视角key是否有效”；随机顺序A3避免MoCo的query/key角色和摄像头绑定，通常应作为主要候选；A4用于防止模型只学习视角不变性而损失同视角细粒度信息。

## 目录内容

- `config/dualcam_config.json`：路径、相机统计、7个预训练和9个微调任务。
- `dualcam_pretrain_entry.py`：双相机加载、归一化时间对齐、辅助CE和断点恢复。
- `dualcam_finetune_entry.py`：微调恢复和相机独立的逐样本测试文件。
- `run_pretrain.py`、`run_finetune.py`、`run_test.py`：单任务入口。
- `inspect_manifest.py`：双摄像头路径和帧数差检查。
- `compute_camera_stats.py`：重新计算相机均值和标准差。
- `summarize_results.py`：生成两路相机排名和camera gap。
- `slurm/`：HPC任务脚本。

运行时会复用同一项目中 `codex_script/rgb_round2_20260717` 的经过验证的辅助CE与断点恢复补丁，因此同步到HPC时应保留这两个脚本目录。

## 第一次提交前

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
mkdir -p train/out ft_and_test/out

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python codex_script/rgb_dualcam_20260717/validate_package.py \
  --dataset-root /mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
```

如果需要逐个确认两路tensor文件已经完整上传：

```bash
python codex_script/rgb_dualcam_20260717/inspect_manifest.py \
  --dataset-root /mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset \
  --check-files
```

## 推荐运行顺序

第一阶段只比较正样本构造，运行index 0–4：

```bash
sbatch codex_script/rgb_dualcam_20260717/slurm/01_pretrain_view_array.slurm
```

完成后进行微调：

```bash
sbatch codex_script/rgb_dualcam_20260717/slurm/03_finetune_array.slurm
```

此时依赖B1/B2的微调任务会提示 prerequisite missing 并安全退出。先测试第一阶段结果，重点判断A3/A4是否超过A0和scratch。

如果跨摄像头正样本有效，再运行第二阶段：

```bash
sbatch codex_script/rgb_dualcam_20260717/slurm/02_pretrain_objective_array.slurm
sbatch codex_script/rgb_dualcam_20260717/slurm/03_finetune_array.slurm
```

最后在两个摄像头测试：

```bash
sbatch codex_script/rgb_dualcam_20260717/slurm/04_test_both_cameras.slurm
```

## 输出路径

预训练：

```text
results/cl_rgb_dualcam_N_20260717/<experiment_id>/
```

微调：

```text
results/ft_rgb_dualcam_N_20260717/weights/<finetune_task>/run_01_<source>/
results/ft_rgb_dualcam_N_20260717/datamaps/<finetune_task>/run_01_<source>/
```

两个摄像头的逐样本结果不会互相覆盖：

```text
best_val_cam00143_per_sample_test.csv
best_val_cam00152_per_sample_test.csv
best_val_cam00143_test_metrics.json
best_val_cam00152_test_metrics.json
```

汇总结果：

```text
results/ft_rgb_dualcam_N_20260717/weights/_batch_test/dualcam/summary/rgb_dualcam_test_results.csv
results/ft_rgb_dualcam_N_20260717/weights/_batch_test/dualcam/summary/rgb_dualcam_test_results_ranked.csv
results/ft_rgb_dualcam_N_20260717/weights/_batch_test/dualcam/summary/rgb_dualcam_camera_gap.csv
```

## 判断标准

主要结论应基于00143测试集，因为它与原RGB基线一致。00152结果用于判断视角泛化：

1. A3相对A0在00143提升：跨相机正样本改善主任务。
2. A3只在00152提升：主要获得跨视角鲁棒性，但未改善原测试域。
3. A3优于A2：随机交换摄像头角色是必要的。
4. A4优于A3：纯跨摄像头约束过强，需要保留同相机正样本。
5. B1/B2只有在A3有效后才值得作为最终方法。

scratch使用全网络LR 1e-3；预训练模型默认backbone LR 1e-4，A3额外比较3e-4。预训练和微调均每10个epoch保存，重复提交同一任务会自动恢复。
