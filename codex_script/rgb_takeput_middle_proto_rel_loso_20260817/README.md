# RGB Take/Put + Middle Proto/Rel 实验包

本包用于新的 2 类与 11 类 RGB 实验，统一支持 Windows `.bat` 与 HPC Slurm。所有路径、训练超参数、增强参数和实验网格集中在 [`config/experiment_config.json`](config/experiment_config.json)；换电脑时优先只修改该文件中的 `paths.windows` 或 `paths.hpc`。

## 核心协议

- `take_put`：`take, put`。
- `middle`：`insert, cut, label, pull_out, wrap, move, measure, remove, open, tear, cap`。
- `full`：全部 17 类；本包先生成 manifest，当前实验网格不自动训练 full。
- 参数开发固定使用 `dev_N`：M+MR+J 训练，N 每个分类 epoch 都评估并记录。
- N 已参与模型、超参数和 epoch 选择，因此报告中必须称为开发被试，不能称为无偏最终测试。
- 参数锁定后才运行 `test_M/test_J/test_MR`。每个 fold 都由其余三人训练、目标一人测试，不复用 N 上训练的权重。

预训练固定 200 epochs。只在 50/100/150/200 保存完整权重；其余 epoch 不做下游微调评估。训练过程仍持续记录 loss、特征、batch 标签、梯度、参数更新、非有限值检查；Proto/Rel 状态每 epoch 写诊断文件。分类阶段每个 epoch 都在当前 held-out subject 上评估，直接分类 100 epochs，下游微调 50 epochs。

## 第一次使用

1. 修改中心配置中的路径。
2. Windows 运行 `scripts/windows/00_prepare_validate.bat`；HPC 在 `scripts/slurm` 目录运行 `./submit.sh 00_prepare_validate.slurm`。
3. 检查 `runtime/manifests/manifest_audit.json`，确认样本数、类别数、人员划分和零重叠。
4. 用 `run.py list --stage <stage>` 查看行号与配置，再提交对应脚本。

Windows 启动器本身需要一个能运行脚本的 Python。若命令名不是 `python`，先设置 `PYTHON_BOOTSTRAP`；真正训练用的 Python 仍由中心配置的 `python_bin` 控制。HPC 也可在作业环境中设置同名变量。

Slurm 推荐始终通过 `scripts/slurm/submit.sh` 提交，使 `SLURM_SUBMIT_DIR` 可稳定定位包目录。例如 `./submit.sh 20_middle_augmentation_devN.slurm`。若集群需要 module/conda 初始化，可在 `common_env.sh` 中加入环境初始化，或在提交前导出 `PYTHON_BOOTSTRAP`。

## 推荐执行顺序

1. `10_takeput_direct_devN`：6 个直接分类配置。
2. `11_takeput_supcon_devN`：4 个 SupLoss 预训练配置；每个预训练 checkpoint 接 full/head-only 两种下游微调。
3. `20_middle_augmentation_devN`：7 个增强候选；先固定最佳增强。
4. `30_middle_loss_screen_devN`：SupLoss、严格 null、P=1 Proto 权重与 Rel 权重初筛。
5. `31_middle_rel_topk_devN`：topK=3/5/10；11 类中每类只有 10 个异类，因此 K=10 就是 all。
6. `32_middle_followups_devN`：只在前一步证据支持时手动运行 Rel 起点、联合损失和 P=2 sentinel。
7. 锁定配置后，用 `80_locked_generalization` 在 M/J/MR 三个独立 LOSO fold 运行。

不建议铺开 P=2/P=3 主网格。主实验统一 P=1；P=2 仅保留 null/active 两个小型 sentinel，用于验证“多 prototype 高相似且无增益”的结论是否在新 11 类任务仍成立；不再安排 P=3。

## 常用命令

```text
python run.py validate --platform windows
python run.py prepare --platform windows
python run.py list --stage middle_loss_screen --platform windows
python run.py pipeline --stage middle_loss_screen --index 2 --fold dev_N --platform windows
python run.py summarize --platform windows
```

`pipeline` 对预训练实验依次执行：200-epoch 预训练、full 微调、head-only 微调、训练诊断汇总。直接分类实验只执行其网格中指定的 full/head-only 策略。

## 结果与分析

- `tools/analyze_training_diagnostics.py`：汇总 loss、梯度范数、参数更新、特征统计、nonfinite、prototype 状态和 checkpoint 完整性。
- `tools/summarize_results.py`：汇总分类运行、逐 epoch 曲线、最佳 epoch 与末 epoch 的性能下降。
- `tools/analyze_features.py`：支持预训练或分类 checkpoint；计算线性探针、1-NN、silhouette、Davies–Bouldin、类内/类间距离、有效秩，并生成 PCA/UMAP 图。降维器只在训练被试上拟合，再 transform held-out subject，避免联合拟合泄漏。

训练输出中的 `resolved_experiment.json` 保存了当次解析后的完整配置和命令，便于复现实验。实验包不会写入旧包的 `confirmation_runner`，也不会导入或调用 `rgb_proto_rel_v2_20260804`。

详细网格、增强来源和分析判据见 [`EXPERIMENT_CONFIGS.md`](EXPERIMENT_CONFIGS.md)。
