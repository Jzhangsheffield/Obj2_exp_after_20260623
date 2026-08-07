# Stage 6：强视频骨干、迁移策略与 SupCon 因果确认

## 目的

Stage 6 回答三个不同问题，不能把结果混在一起解释：

1. **6A**：不训练 backbone 时，哪种现成的视频表征已经能跨受试者区分类别？
2. **6B**：不经过本数据集 SupCon，Kinetics-400 backbone 采用哪种下游迁移方式最好？这就是强 no-SupCon 基线。
3. **6C**：在同一个胜出 backbone、同一个微调策略下，本数据集 SupCon 相对 K400 直接微调到底改善还是破坏表征？

Stage 6 不使用锁定 N，只使用 M+J train / MR validation。所有模型选择结束后才能沿用 Stage 5 的锁定测试原则。

## 6A：冻结特征筛选

| index | ID | 初始化 | 输入 |
|---:|---|---|---|
| 0 | s6a_a0_r3d10_random | 当前自定义 R3D-10 随机初始化，负控制 | 16×224² |
| 1 | s6a_a1_r3d18_k400 | torchvision R3D-18 K400 | 16×224² |
| 2 | s6a_a2_r2plus1d18_k400 | torchvision R(2+1)D-18 K400 | 16×224² |
| 3 | s6a_a3_mvitv2s_k400 | torchvision MViT-v2-S K400 | 16×224² |
| 4 | s6a_a4_swin3dt_k400 | torchvision Swin3D-T K400 | 32×224² |

诊断输出包括 frozen linear BA/Macro-F1、1-NN BA、动作与受试者 cosine silhouette、effective rank、top-5 variance、between/within、reverse/shuffle/repeat-center、线性探针 confusion matrix、PCA 坐标；环境安装 `umap-learn` 时还会保存 UMAP 坐标。6A 是确定性特征抽取，不做随机空间增强。

HPC：

```bash
sbatch stage6_backbone_transfer/slurm/00_stage6a_array.slurm
sbatch stage6_backbone_transfer/slurm/90_summarize.slurm
```

Windows：运行 `windows\00_stage6a_all.bat`，随后运行 `windows\90_summarize.bat`。

阅读 `summary/diagnostic_summary.csv` 后，把前两名 ID 写入 `config/stage6_selection.json` 的 `stage6b_candidates`，并设 `stage6a_ready=true`。优先级为 MR frozen linear BA、Macro-F1、动作 silhouette；person silhouette 越高表示越依赖身份，应作为惩罚项。不能仅凭 UMAP 选择。

## 6B：K400 直接迁移

对 6A 前两名各运行三种 seed-1 策略，共 6 项：

- `head`：冻结整个 backbone，只训练线性头，25 epoch；回答现成特征是否已足够。
- `partial`：只解冻最后语义 stage、归一化层和分类头，50 epoch；控制小数据全量微调造成的遗忘。
- `full`：全部解冻，100 epoch；作为最大适配能力对照。

CNN 使用 backbone LR `1e-4`，Transformer 使用 `3e-5`，分类头均为 `1e-3`，AdamW、weight decay `1e-4`，每 25 epoch 保存。`partial` 对 R3D/R(2+1)D 解冻 `layer4`；MViT 解冻最后 2 个 blocks + norm；Swin3D 解冻最后 stage + norm。

先设好 6A 选择门，再提交：

```bash
sbatch stage6_backbone_transfer/slurm/10_stage6b_screen_array.slurm
```

选出一个 backbone/policy 后，在 `stage6_selection.json` 填 `stage6b_winner`、`stage6b_policy` 并设 `stage6b_ready=true`。然后补 seed 2/3：

```bash
sbatch stage6_backbone_transfer/slurm/11_stage6b_confirm_array.slurm
```

6B 的三种子均值/标准差是 Stage 6 的 **no-SupCon 强基线**。

## 6C：SupCon 是否真正有用

固定 6B 胜出的 backbone 和迁移策略，seed 1 比较：

- `C0 direct_k400`：6B 中的 K400 直接迁移胜出结果，不重复训练。
- `C1 random_supcon`：随机初始化 → 本数据集 SupCon 200 epoch → 相同微调。
- `C2 k400_supcon`：K400 → 本数据集 SupCon 200 epoch → 相同微调。
- `C3 k400_supcon_patch_frozen`：只对 MViT/Swin3D运行；SupCon 阶段冻结 patch projection，其余网络可训练，再按相同策略微调。

冻结对象严格为 MViT `conv_proj` 或 Swin3D `patch_embed.proj`，不是“第一个 Transformer block”。C3 用于检验小数据 SupCon 是否在最底层破坏已学到的视频局部表征。

SupCon 使用 cosine + 10 epoch warm-up、200 epoch、每 50 epoch保存。K400 模型用较小 LR：Transformer `3e-5`，CNN `1e-4`。不启用辅助 CE、ProtoLoss、RelLoss，避免改变因果问题。

运行顺序：

```bash
cl=$(sbatch --parsable stage6_backbone_transfer/slurm/20_stage6c_pretrain_array.slurm)
diag=$(sbatch --parsable --dependency="afterok:${cl}" stage6_backbone_transfer/slurm/21_stage6c_diagnose_array.slurm)
sbatch --dependency="afterok:${cl}" stage6_backbone_transfer/slurm/22_stage6c_finetune_array.slurm
```

若 6B 胜出的是 CNN，C3 不适用，提交 20/21/22 时用 `sbatch --array=0-1 ...` 覆盖脚本默认的 `0-2`。Windows 对应的三个 all 脚本默认也是 Transformer 的 3 项；CNN 时只手动运行 `run.py ... --index 0` 和 `--index 1`。

比较 C0–C3 的 MR BA/Macro-F1，以及 SupCon 前后 frozen diagnostics。若选择 SupCon 方案，将 `stage6c_winner` 写为对应名称并设 `stage6c_ready=true`，再运行 seed 2/3：

```bash
cl=$(sbatch --parsable stage6_backbone_transfer/slurm/30_stage6c_confirm_pretrain_array.slurm)
sbatch --dependency="afterok:${cl}" stage6_backbone_transfer/slurm/31_stage6c_confirm_finetune_array.slurm
```

如果 `direct_k400` 最好，不运行 6C confirm 脚本；直接使用 6B confirm 的 seed 2/3 与 seed 1 组成三种子结果。

## 权重与输出

- 6A 不训练、不保存 checkpoint，只保存诊断数组和 JSON。
- 6B/6C 微调每 25 epoch 保存周期权重；训练器还保存 `best_val_balanced.pth` 与 `last.pth`。
- 6C SupCon 每 50 epoch 保存，共 `0050/0100/0150/0200` 四个周期 checkpoint；最终权重就是 `checkpoint_0200.pth`，不额外重复保存。
- 输出位于 `results/rgb_supcon_repair_20260806/diagnostics/stage6a|stage6c`、`pretrain/stage6c`、`classifier/stage6b|stage6b_confirm|stage6c|stage6c_confirm`。

首次调用 K400 权重时 torchvision 会把官方权重下载到 PyTorch cache。若 Stanage 计算节点不能联网，应先在可联网节点预缓存相同 torchvision 权重，并保持相同 `TORCH_HOME`。
