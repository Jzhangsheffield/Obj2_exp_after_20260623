# RGB MViT-V2-S 与 Motion-Crop 对照实验包

## 1. 实验目的

本实验包回答两个相互独立的问题：

1. 将 RGB backbone 从 ResNet3D-18 更换为随机初始化的 MViT-V2-S，能否改善动作类别表征？
2. 保持 ResNet3D-18 不变，只输入 motion-crop 关键运动区域，能否改善 SupLoss 学到的类别分离？

本包只使用：

- split：`N_as_test`
- task：`except_take_put`
- camera：`00143`
- tier：`tier1`，15 类
- clip：16 帧，224×224
- seed：1
- MViT 权重：`weights=None`，不使用 Kinetics 或其他预训练权重

所有改动都在本目录的 `src/` 副本中。项目原有 `train/`、`ft_and_test/`、`backbone/`、`utils_/` 和 `aug/` 文件没有被修改。

## 2. 实验矩阵

| 编号 | ID | Backbone | 输入 | 训练方式 |
|---|---|---|---|---|
| P0 | `mvit_v2_s_original_supcon` | MViT-V2-S | 原始 RGB | SupLoss 预训练 |
| P1 | `resnet3d18_motioncrop_supcon` | ResNet3D-18 | motion crop | SupLoss 预训练 |
| C0 | `mvit_v2_s_original_scratch` | MViT-V2-S | 原始 RGB | 随机初始化、全模型监督训练 |
| C1 | `mvit_v2_s_original_supcon_head` | MViT-V2-S | 原始 RGB | 加载 P0，冻结 backbone，只训练分类头 |
| C2 | `resnet3d18_motioncrop_scratch` | ResNet3D-18 | motion crop | 随机初始化、全模型监督训练 |
| C3 | `resnet3d18_motioncrop_supcon_head` | ResNet3D-18 | motion crop | 加载 P1，冻结 backbone，只训练分类头 |

这里的 head-only 只用于 SupLoss 预训练后的 backbone。没有“冻结随机 backbone 只训练头”的无效实验。

## 3. 固定的数据与增强策略

- train：`N_as_test/train_manifest_except_take_put.jsonl`
- validation：`N_as_test/val_manifest_except_take_put.jsonl`
- test：`N_as_test/test_manifest_except_take_put.jsonl`
- 原图字段：`rgb_cam_00143`
- 裁剪字段：`rgb_cam_00143_motion_crop_m32`
- SupLoss 两个视图使用相同的 temporal indices，避免两个视图抽到不同动作阶段。
- 训练空间增强采用弱增强：RRC scale 0.85–1.0、ratio 0.9–1.1、水平翻转 0.5、轻度 ColorJitter、轻度 blur，不使用垂直翻转和灰度化。
- motion crop 在 resize/RRC 前先按训练集均值 padding 成正方形，避免把不同比例的裁剪区域强行拉伸。
- motion-crop mean/std 不写死。流水线第一阶段扫描训练 split 并保存到：
  `results/rgb_mvit_motioncrop_seed1_20260721/runtime/motion_crop_train_rgb_stats.json`。

当前 manifest 审计结果为 train 1026、validation 181、test 416 个样本；test split 只包含 14 个实际类别，缺少 `close`，虽然分类头仍按全任务的 15 类构建。测试 Balanced Accuracy 和 Macro-F1 应按 test 中实际出现的类别解释，验证脚本会明确打印这一警告。

## 4. 目录结构

```text
rgb_mvit_motioncrop_seed1_20260721/
├── README.md
├── config/experiments.json          # 唯一实验配置
├── run_experiment.py                # Windows/HPC 共用启动器
├── scripts/
│   ├── windows/                     # .bat 脚本
│   └── slurm/                       # .slurm 与依赖提交脚本
├── tools/
│   ├── validate_runtime.py
│   ├── compute_rgb_stats.py
│   ├── evaluate_features.py
│   └── summarize_results.py
└── src/                             # 从原项目复制并隔离修改的训练代码
```

## 5. Python 环境要求

环境至少需要：

- PyTorch 与 torchvision，且 `torchvision.models.video.mvit_v2_s` 可导入；
- NumPy；
- scikit-learn；
- matplotlib；
- tqdm。

训练代码使用 `torch.amp`、torchvision transforms v2 和 MViT，因此必须先运行验证脚本。验证会打印 Python、PyTorch、torchvision、CUDA 状态和 MViT feature dimension。

## 6. Windows 运行方法

先进入 PyTorch 环境，然后进入本实验包：

```bat
conda activate pytorch
cd /d D:\Junxi_data\Obj2_experiments_after_260623\codex_script\rgb_mvit_motioncrop_seed1_20260721
```

如果 `python` 不是目标环境解释器，可先设置：

```bat
set PYTHON_BIN=C:\path\to\env\python.exe
```

数据集默认位置是：

```text
C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset
```

如位置不同：

```bat
set DATASET_ROOT=D:\your\Final_Mapstyle_Dataset
```

按阶段运行：

```bat
scripts\windows\00_validate.bat
scripts\windows\01_compute_crop_stats.bat
scripts\windows\02_pretrain_all.bat
scripts\windows\03_classifier_all.bat
scripts\windows\04_test_all.bat
scripts\windows\05_features_all.bat
scripts\windows\06_summarize.bat
```

完整串行运行：

```bat
scripts\windows\run_all.bat
```

Windows 完整串行训练可能持续很久。建议先用 dry-run 检查命令：

```bat
set DRY_RUN=1
scripts\windows\run_all.bat
```

确认后清除 dry-run：

```bat
set DRY_RUN=
```

运行单个实验的示例：

```bat
python run_experiment.py pretrain --platform windows --experiment mvit_v2_s_original_supcon
python run_experiment.py classifier --platform windows --experiment mvit_v2_s_original_supcon_head
python run_experiment.py test --platform windows --experiment mvit_v2_s_original_supcon_head
python run_experiment.py features --platform windows --experiment mvit_v2_s_original_supcon
```

## 7. HPC/SLURM 运行方法

### 7.1 提交前检查

将整个项目和更新后的数据集同步到 HPC。尤其要确认以下内容已经同步：

- 每个样本中的 `rgb_cam_00143_motion_crop_m32.pt`；
- N_as_test 的 train/val/test manifests 中的 motion-crop 字段；
- 本实验包目录。

默认 HPC 路径写在 `config/experiments.json` 和 `scripts/slurm/common_env.sh`。如实际路径不同，可以在提交前设置：

```bash
export PROJECT_ROOT=/your/project/path
export DATASET_ROOT=/your/Final_Mapstyle_Dataset
export CONDA_ENV=pytorch
```

### 7.2 推荐：一次提交整个依赖流水线

```bash
cd "$PROJECT_ROOT/codex_script/rgb_mvit_motioncrop_seed1_20260721/scripts/slurm"
bash submit_pipeline.sh
```

该脚本按依赖关系提交：

```text
validate → crop stats → pretrain array → classifier array → test array
                                  └────→ feature array ─────┘
                                                   ↓
                                                summary
```

上游失败后，下游不会启动。

### 7.3 手动分阶段提交

```bash
sbatch 00_validate.slurm
sbatch 01_compute_crop_stats.slurm
sbatch 02_pretrain_array.slurm
sbatch 03_classifier_array.slurm
sbatch 04_test_array.slurm
sbatch 05_features_array.slurm
sbatch 06_summarize.slurm
```

手动提交时必须等待前一阶段成功，尤其是：

- stats 完成后才能运行 motion-crop 训练；
- pretrain 完成后才能运行两个 head-only 分类任务；
- classifier 完成后才能 test。

Array index 对应关系：

| 脚本 | index | 实验 |
|---|---:|---|
| `02_pretrain_array.slurm` | 0 | MViT 原图 SupLoss |
|  | 1 | ResNet3D motion-crop SupLoss |
| `03_classifier_array.slurm` | 0 | MViT 原图 scratch |
|  | 1 | MViT SupLoss head-only |
|  | 2 | ResNet3D motion-crop scratch |
|  | 3 | ResNet3D motion-crop SupLoss head-only |
| `04_test_array.slurm` | 0–3 | 与 classifier index 相同 |
| `05_features_array.slurm` | 0–1 | 与 pretrain index 相同 |

## 8. 结果位置

所有结果写入：

```text
results/rgb_mvit_motioncrop_seed1_20260721/
```

主要子目录：

- `pretrain/<id>/checkpoint_0200.pth`：SupLoss 最终 checkpoint；
- `classifier/<id>/.../best_val_balanced.pth`：按 validation Balanced Accuracy 保存的最佳分类模型，也是 test 使用的模型；
- `classifier/<id>/.../last.pth`：分类训练完成标志；
- `test/<id>/test_results.csv`：独立 test 结果；
- `features/<id>/feature_metrics.json`：backbone 与 projection features 的分离指标；
- `features/<id>/*_pca.png`：按真实类别着色的 PCA 图；
- `summary/classifier_test_summary.csv`：四个分类实验汇总；
- `summary/feature_separation_summary.csv`：两个 SupLoss 表征汇总。

## 9. 特征分离评价口径

对 SupLoss checkpoint 同时评估：

- projection head 之前的 backbone feature；
- 128 维 projection feature；
- logistic-regression linear probe；
- cosine kNN，k=1/5/10；
- true-label cosine silhouette；
- nearest-centroid Balanced Accuracy；
- 类间质心距离/类内距离比；
- PCA 图。

判断 backbone 是否真正分离类别时，优先看 backbone feature 的 linear-probe、kNN、silhouette 和 between/within ratio；PCA 只作为辅助图，不单独作为结论。

## 10. 默认超参数与显存

- MViT SupLoss batch size：8；
- MViT classifier scratch batch size：8；
- MViT head-only batch size：16；
- ResNet3D batch size：64；
- queue size：1088，可被两个预训练 batch size 整除；
- SupLoss：200 epochs；
- classifier：100 epochs；
- optimizer：AdamW；
- MViT learning rate：1e-4，分类头：1e-3；
- ResNet3D learning rate：1e-3。

若 MViT 仍然 OOM，优先把配置中的 MViT batch size 从 8 改成 4 或 2；1088 对这两个数仍可整除。不要任意改成不能整除 queue size 的 batch size。

## 11. 中断与重复运行

- SupLoss 每 10 epochs 保存 checkpoint。再次启动同一预训练任务时，启动器会自动选取最新 checkpoint 继续；存在 `checkpoint_0200.pth` 时会跳过。
- 分类任务存在 `last.pth` 时会跳过，避免覆盖完整结果。
- 当前分类脚本不自动恢复未完成的中间 epoch；如果分类作业中断，应保留日志后重新运行该分类任务到一个新的输出目录，或先确认并移走未完成目录。
- test CSV 使用 append 模式；重复测试后汇总脚本只读取每个 CSV 的最后一行。

## 12. 实验纪律

- 所有模型选择只使用 validation split；
- test 阶段应在训练与配置确定后运行；
- 不根据 test 结果反向挑选 checkpoint 或超参数；
- 报告中注明这里只使用 seed 1，结论属于单次运行结果；
- 比较裁剪效果时，必须同时报告 motion-crop scratch 与 motion-crop SupLoss head-only；
- 比较 backbone 时，必须同时报告 MViT scratch 与 MViT SupLoss head-only。
