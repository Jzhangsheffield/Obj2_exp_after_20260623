# RGB SupLoss 表征失效诊断与修复实验包（2026-08-06）

## 1. 实验目标

本包用于解释并修复“IMU 使用 SupLoss 后类别清晰分离，而 ResNet3D RGB 表征仍严重混合”的问题。已有证据表明 RGB 表征有效秩极低、对倒序/乱序几乎不敏感、局部帧差比整帧外观更可分，并且旧 train/val 存在同一次录制片段交叉。因此，本包按因果优先级检验：评估协议、模型容量与优化、运动输入、时间建模、IMU 教师指导，最后才进行三种子微调和锁定测试。

本包不修改原五阶段包、V2 包或项目训练源码。运行时复用 `rgb_mvit_motioncrop_seed1_20260721/src`，所有新模型和输入变换只在当前 Python 进程内安装。

## 2. 必须遵守的运行顺序

1. Stage 0：校验环境，重建无受试者混合的训练/验证划分，并计算 motion-crop 统计量。
2. Stage 1：比较 ResNet3D-18/10、旧 step LR、10 epoch warm-up + cosine、辅助 CE。
3. Stage 2：比较整帧、motion-crop、绝对帧差、RGB+帧差双流。
4. Stage 3：比较 16/32 帧以及保留时间位置后的 attention pooling。
5. Stage 4：用冻结 IMU SupLoss 教师做实例对齐和类别关系蒸馏。
6. 根据 Stage 1–4 的 MR 验证诊断编辑 `config/final_selection.json`，将 `selection_ready` 改为 `true`。
7. Stage 5：三种子运行 scratch full、SupLoss head-only、SupLoss full。
8. 只有所有设计和 checkpoint 已固定后，手动解锁并测试 N；任何自动流程都不提交 N 测试。

独立追加的 Stage 6 用于比较 Kinetics-400 视频 backbone，并进一步拆分“强初始化”“下游迁移”和“SupCon”的贡献。它不会改写 Stage 0–5 的定义；完整顺序与选择门见 `stage6_backbone_transfer/README.md`。建议在 Stage 0 已生成新协议后运行 Stage 6。

不要把 Stage 1–4 全部无条件连续提交。每个阶段结束后先读取 `results/rgb_supcon_repair_20260806/summary/diagnostic_summary.csv`，通过停止规则再进入下一阶段。

筛选出最多两个候选后，可用同一启动器补三折受试者验证，例如：

```bash
python run.py pretrain --stage stage2 --experiment s2_m3_dual --split-profile fold_M
python run.py diagnose --stage stage2 --experiment s2_m3_dual --split-profile fold_M
```

把 `fold_M` 依次换成 `fold_J`、`fold_MR`。这些结果写入 `pretrain_crossval/`、`diagnostics_crossval/`，不会覆盖主筛选结果；汇总文件为 `crossval_diagnostic_summary.csv`。

## 3. Windows 快速开始

```bat
conda activate pytorch
cd /d D:\Junxi_data\Obj2_experiments_after_260623\codex_script\rgb_supcon_repair_20260806
stage0_protocol\windows\00_validate.bat
stage0_protocol\windows\01_prepare_protocol.bat
stage1_capacity_optim\windows\01_pretrain_all.bat
stage1_capacity_optim\windows\02_diagnose_all.bat
```

后续阶段同样先运行 `01_pretrain_all.bat`，再运行 `02_diagnose_all.bat`。Stage 4 在预训练前额外运行 `00_cache_teacher.bat`。

可先检查命令而不训练：

```bat
set DRY_RUN=1
stage2_motion_input\windows\01_pretrain_all.bat
set DRY_RUN=
```

若 Python 或数据路径不同，可在运行前设置 `PYTHON_BIN`、`DATASET_ROOT`。预训练存在最终 `checkpoint_0200.pth` 时自动跳过；中断后重跑会从最新的 50-epoch checkpoint 恢复。

## 4. HPC/SLURM 快速开始

```bash
export PROJECT_ROOT=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623
export DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset
cd "$PROJECT_ROOT/codex_script/rgb_supcon_repair_20260806"
sbatch stage0_protocol/slurm/00_validate.slurm
sbatch stage0_protocol/slurm/01_prepare_protocol.slurm
```

Stage 1 示例：

```bash
pre=$(sbatch --parsable stage1_capacity_optim/slurm/01_pretrain_array.slurm)
diag=$(sbatch --parsable --dependency="afterok:${pre}" stage1_capacity_optim/slurm/02_diagnose_array.slurm)
sbatch --dependency="afterok:${diag}" stage1_capacity_optim/slurm/03_summarize.slurm
```

Stage 2、3 使用同样方式。Stage 4 的依赖顺序是 cache teacher → pretrain array → diagnose array → summarize。SLURM 脚本明确加载 Anaconda、cuDNN 和 `pytorch` 环境，直接调用环境中的 Python，不把整条命令塞进单个引号参数。

## 5. Stage 5 和锁定测试

先编辑 `config/final_selection.json`，填入 MR 验证上胜出的 checkpoint 及其 backbone/input/frame/temporal 配置，然后设置：

```json
"selection_ready": true
```

再运行三种子微调。Windows：

```bat
stage5_final_confirm\windows\01_finetune_all.bat
```

HPC：

```bash
sbatch stage5_final_confirm/slurm/01_finetune_array.slurm
```

确认配置不再改变后才允许 N 测试。Windows 使用 `set ALLOW_LOCKED_TEST=YES`；HPC 使用 `export ALLOW_LOCKED_TEST=YES`。测试脚本没有被任何上游流水线自动调用。

## 6. 输出目录

```text
results/rgb_supcon_repair_20260806/
├── runtime/
│   ├── splits/                         # 新划分与审计
│   ├── motion_crop_train_stats.json
│   └── imu_teacher/train_MJ_features.pth
├── pretrain/stage1|stage2|stage3|stage4/<experiment>/
├── pretrain/stage6c/<experiment>/
├── diagnostics/stage1|stage2|stage3|stage4|stage6a|stage6c/<experiment>/diagnostics.json
├── classifier/stage5/<mode_seed>/
├── classifier/stage6b|stage6b_confirm|stage6c|stage6c_confirm/<experiment>/
├── test/stage5/<mode_seed>/test_results.csv
└── summary/
    ├── diagnostic_summary.csv
    ├── diagnostic_ranking.json
    └── locked_test_summary.csv
```

完整参数含义、对照关系、判断阈值和资源配置见 `ALL_EXPERIMENT_CONFIGS.md`。每个 stage 的子目录也有独立 README。

## 7. 已知边界

- Stage 1–4 是单种子筛选，不能用来报告最终均值和标准差。
- Stage 4 使用成对 IMU 作为训练期教师，推理仍只需要 RGB；它是“跨模态训练”，应与纯 RGB 结果分栏报告。
- `absdiff` 在完成同一空间变换和标准化后计算相邻帧绝对差；双流将标准 RGB 与该差分在通道维拼接，然后由两个独立 ResNet3D 分支编码。
- motion-crop 缺失的样本会被数据加载器过滤；Stage 0 的审计文件必须报告实际样本数，比较时需说明这一点。
