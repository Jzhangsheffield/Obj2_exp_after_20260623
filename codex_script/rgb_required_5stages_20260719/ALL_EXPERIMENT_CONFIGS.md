# 五阶段全部实验配置及T3时间下采样附加实验

## 1. 全阶段固定条件

| 类别 | 配置 |
|---|---|
| 项目根目录 | `/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623` |
| 数据集根目录 | `/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset` |
| train | `N_as_test/train_manifest_except_take_put.jsonl` |
| val | `N_as_test/val_manifest_except_take_put.jsonl` |
| test | `N_as_test/test_manifest_except_take_put.jsonl` |
| label map | `label_map_except_take_put.json` |
| RGB 相机 | `00143` |
| 类别 / 帧数 | 15 类 / 16 帧 |
| RGB 归一化 | mean `[0.3752,0.3864,0.3960]`; std `[0.2934,0.2724,0.2644]` |
| Backbone | 3D ResNet-18 |
| 选择标准 | validation Balanced Accuracy；测试只使用 `best_val_balanced.pth` |

默认 `backbone_temporal_mode=current`，即时间维 `16→16→8→8→4→2→1`（分别对应输入、conv1、maxpool、layer1、layer2、layer3、layer4）。T3附加实验使用 `backbone_temporal_mode=t3_lfb`，即 `16→16→16→16→8→8→8`。两种模式参数形状相同，但训练、微调和测试必须使用与checkpoint一致的时间模式。

## 2. 对比预训练固定参数

200 epochs，batch 64，AdamW，LR `1e-3`，weight decay `1e-4`，milestones `50/100/150`，projection 128，queue 1088，temperature 0.07，SupLoss，queue positives 6，prototype refresh 每 10 epochs，prototype temperature 0.07，真实 prototype EMA 0.99。每 10 epochs 保存 checkpoint，并自动从最新 checkpoint 续训。

动作保持增强 A2：两个 view 使用完全相同的帧索引；RRC scale `[0.85,1.0]`，ratio `[0.9,1.1]`；水平翻转 0.5，垂直翻转 0；color jitter 概率 0.2、强度 `[0.1,0.1,0.1,0.02]`；灰度 0；Gaussian blur 概率 0.1、kernel 5、sigma `[0.1,1.0]`。

所有预训练开启现有 debug JSONL、prototype/feature/non-finite/gradient 总体诊断。prototype 正样本模式：`single` 只拉近分配到的 prototype；`all` 等权拉近同类全部 prototype；`soft` 用相似度产生停止梯度的责任权重，避免多 prototype 被强迫坍缩。

## 3. 微调与测试固定参数

微调 100 epochs，batch 64，AdamW，head LR `1e-3`，weight decay `1e-4`，milestones `50/75`。预训练模型默认 backbone LR `3e-4`，Stage 7 额外比较 `1e-4/3e-4/1e-3`。微调增强为 RRC `[0.85,1.0]`、ratio `[0.9,1.1]`、水平翻转 0.5，其余 RGB 增强关闭。测试时所有增强关闭，batch 64。

## 4. Stage 0：复现与基线

预训练：`sup_s1_a`、`sup_s1_b` 使用相同 seed 1；`sup_s2`、`sup_s3` 使用 seed 2/3。全部是 SupLoss-only。微调同时做四个 scratch 和四个对应 SupLoss checkpoint。重点看同 seed 重复差、3-seed 均值±标准差，以及 scratch 与 SupLoss 的配对差。

## 5. Stage 1：proto 形式

| index | ID | 正样本定义 | prototype/类 | λproto |
|---:|---|---|---:|---:|
| 0 | p0_sup | 无 proto loss | 1 | 0 |
| 1 | p1_single_p1 | single | 1 | 0.1 |
| 2 | p2_single_p2 | single | 2 | 0.1 |
| 3 | p3_all_p2 | all | 2 | 0.1 |
| 4 | p4_soft_p2 | soft | 2 | 0.1 |
| 5 | p5_single_p3 | single | 3 | 0.1 |
| 6 | p6_all_p3 | all | 3 | 0.1 |
| 7 | p7_soft_p3 | soft | 3 | 0.1 |

proto 从 epoch 50 启动，rel 完全关闭。每个预训练 checkpoint 均用 backbone LR `3e-4` 微调。首先判断 all-positive 是否随 prototype 数增加而恶化，再比较 soft 是否保留多模态而不坍缩。

## 6. Stage T3：时间压缩消融

T3只改变ResNet3D-18的时间步幅：stem maxpool使用 kernel `(1,3,3)`、stride `(1,2,2)`；layer2保留唯一一次时间stride 2；layer3和layer4只做空间stride `(1,2,2)`。因此16帧在深层输出保留8个时间位置，再由最终AdaptiveAvgPool聚合。

预训练任务为 `t3_sup_s1/s2/s3`，均使用A2 shared、SupLoss-only及seed 1/2/3。微调任务为 `scratch_t3_s1/s2/s3` 和 `t3_sup_s1_ft/s2_ft/s3_ft`。与Stage 0相同seed的current结构进行配对比较。输出路径为：

```text
results/cl_rgb_req_t3_20260721
results/ft_rgb_req_t3_20260721
```

T3预训练Slurm时限24小时、微调18小时；其它超参数与Stage 0完全相同。

## 7. Stage 4：rel 机制

共 17 个实验：R0 SupLoss-only；R1 复现当前 p3、same+diff、全类别、preview EMA 0.5；R2–R4 比较 top-k 3/5/10；R5–R7 关闭 same，仅保留 diff 并比较 top-k；R8–R10 比较 rel 启动 epoch 100/125/150；R11–R13 比较 preview EMA 0.8/0.9/0.99；R14–R16 比较 λrel 1/2/5。所有 rel 实验均关闭 proto contrastive loss，使用 3 prototypes/类，真实 EMA 0.99。

默认推荐候选为 `r15_diff_k3_s125_pm09_l2`，但 Stage 7 前必须以实际验证结果替换，不把推荐当结论。

## 8. Stage 5：proto + rel

| index | ID | 作用 |
|---:|---|---|
| 0 | c0_sup | SupLoss-only |
| 1 | c1_proto_only | soft-p2，λproto 0.1 |
| 2 | c2_rel_only | diff top3，epoch125，λrel 2 |
| 3 | c3_both_s50 | 两项同时从 epoch50 启动 |
| 4 | c4_p50_r125 | proto 50、rel 125，主推荐 |
| 5 | c5_p50_r100 | rel 提前到 100 |
| 6 | c6_p50_r150 | rel 延后到 150 |
| 7 | c7_p25_r125 | proto 提前到 25 |
| 8 | c8_lp005 | λproto 0.05 |
| 9 | c9_lp020 | λproto 0.2 |
| 10 | c10_lr050 | λrel 0.5 |
| 11 | c11_lr500 | λrel 5.0 |

除单项消融外，默认 soft-p2、diff-only top3、preview EMA 0.9、真实 EMA 0.99。重点不是只找最高分，还要检查：加 proto 后是否稳定提升 SupLoss；再加 rel 后是否在不牺牲类内结构的前提下继续提升。

## 9. Stage 7：微调协议

候选为 scratch、current SupLoss-only、T3 SupLoss-only、best proto、best rel、best proto+rel。scratch 做 head-only 与 full；其余五个候选做 head-only，以及 full backbone LR `1e-4/3e-4/1e-3`，共22个微调任务。T3候选自动设置 `backbone_temporal_mode=t3_lfb`；其它候选为 `current`。实际候选路径和时间模式单独存放在 `stage7_finetune/config/selected_sources.json`，便于冻结和审计。

最终报告至少给出：每种预训练目标的线性探测、最佳 full fine-tune、相对 scratch 和 SupLoss-only 的绝对提升；Stage 0 多 seed 均值±标准差；混淆矩阵或 per-class recall；失败模型的 proto/rel 平均量级和 debug 日志异常。

## 10. Slurm 与环境

GPU 脚本使用 partitions `gpu,gpu-h100,gpu-h100-nvl`、qos `gpu`、1 GPU、12 CPU、60 GB；预训练 18h、微调 14h、测试 6h。环境固定为：

```bash
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
```

脚本始终使用 `${CONDA_PREFIX}/bin/python`。为提高复现性还设置 `PYTHONHASHSEED=0` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。所有实际数值仍以 `common/experiment_plan.json` 为最终准则。
