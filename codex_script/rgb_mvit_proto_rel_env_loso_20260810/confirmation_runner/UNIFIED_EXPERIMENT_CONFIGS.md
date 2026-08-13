# 统一后续实验配置说明（2026-08-13）

本文件是新实验的配置真值说明。机器可读配置位于`config/unified_experiment_registry.json`，运行入口为`run_unified.py`。原来的`run_confirmation.py`与历史结果仍保留，仅用于复现旧协议。

## 1. 两个任务

| ID | 类别 | manifest来源 | label map | 用途 |
|---|---:|---|---|---|
| `t15` | 15 | 完整manifest过滤tier1=`take/put` | `label_map_except_take_put.json` | 完成历史最小确认，保持与旧实验可比 |
| `t17` | 17 | 每个LOSO目录的完整train/val/test manifest去重合并 | `label_map.json` | 后续主要研究任务，包含take与put |

经审计的完整数据共有4543条：t15为1623条，take/put为2920条，占64.3%。因此t15和t17必须作为两个独立任务报告，不能直接比较绝对Accuracy。

## 2. 两种协议

### `subject_dev`：配置开发

- N完全保留，不参与增强、采样方式或Loss参数选择。
- 三个开发折：
  - `holdout_M`：J+MR训练，M验证；
  - `holdout_J`：M+MR训练，J验证；
  - `holdout_MR`：M+J训练，MR验证。
- 微调50轮，同时保存best、epoch 25和epoch 50。
- 配置排名统一使用epoch 50在整人验证对象上的结果，不使用best权重排名。
- 运行阶段使用`evaluate`，脚本会拒绝`test`。

### `final_refit`：锁定配置后的最终LOSO

- 测试对象之外的三个人全部用于训练，原train与val不再分开。
- 不传validation manifest，并显式启用`disable_val`。
- 微调50轮，保存epoch 25、epoch 50及last；此阶段没有有统计意义的best checkpoint。
- 最终只测试`epoch_050.pth`。
- 运行阶段使用`test`，脚本会拒绝`evaluate`。

## 3. 共同模型与优化参数

| 项目 | 设置 |
|---|---|
| backbone | Kinetics-400初始化的MViT-V2-S |
| 输入 | RGB摄像头00143，16帧，224×224 |
| 对比预训练 | 200 epochs，batch size 32，AdamW，LR 6e-5，WD 1e-4 |
| 对比LR | 10轮warm-up + cosine |
| projection / queue | 128 / 1088 |
| SupLoss温度 | 0.07 |
| 对比权重 | 每50轮：50/100/150/200 |
| prototype诊断 | 每10轮；prototype重聚类每10轮 |
| 微调 | 全局微调，50 epochs，batch size 32 |
| backbone/head LR | 6e-5 / 2e-3 |
| 微调LR里程碑 | epoch 25和37 |
| 微调权重 | epoch 25、epoch 50、last；开发协议另外保存best |
| 唯一测试权重 | epoch 50 |

## 4. Loss配置

| ID | SupLoss | ProtoLoss | RelLoss | 关键参数 | 角色 |
|---|---|---|---|---|---|
| `d0` | 否 | 否 | 否 | K400直接全局微调 | 无对比预训练基线 |
| `s0` | 是 | 否 | 否 | `lambda_proto=0, lambda_rel=0` | SupLoss-only基线 |
| `rn3` | 是 | 0 | Null | P3、K3、start125、Rel梯度为零 | `rl3`严格Null |
| `rl3` | 是 | 否 | 是 | P3、diff-only、K3、start125、EMA0.5、λrel=0.5 | 当前Rel候选 |
| `rn2` | 是 | 0 | Null | P2，其余匹配`rl2` | P2 Null |
| `rl2` | 是 | 否 | 是 | P2、diff-only、K3、start125、EMA0.5、λrel=0.5 | P2 Rel候选 |
| `h00_p1_k10` | 是 | 0 | 0 | P1、K10、共同路径但两个新增Loss均无梯度 | 组合严格Null |
| `h10_p1_k10` | 是 | 是 | 0 | P1、all-positive、λproto=1、start50 | Proto-only候选 |
| `h01_p1_k10` | 是 | 0 | 是 | P1、K10、same+diff、λrel=1、cosine | Rel-only消融 |
| `h11_p1_k10` | 是 | 是 | 是 | P1、K10、两个λ均1、start50、cosine | Proto+Rel候选 |

`num_positive=6`对这里使用的SupLoss路径不起作用；SupLoss使用同标签正样本集合。该参数只对KCL路径有意义。

## 5. 对比学习增强

所有增强都对同一clip的全部帧使用时间一致的空间参数，避免逐帧随机变换制造虚假运动。微调增强保持固定的旧版mild配置，以下策略只改变对比预训练，以隔离预训练增强效应。

| ID | RRC scale / ratio | flip | color jitter | gray | blur | 目的 |
|---|---|---|---|---:|---|---|
| `a0` | .85–1.0 / .9–1.1 | H=.5 | p=.2，.1/.1/.1/.02 | 0 | p=.1,k5,σ=.1–1 | 当前基线 |
| `a1_crop` | .65–1.0 / .8–1.25 | H=.5 | 同a0 | 0 | 同a0 | 空间尺度与位置变化 |
| `a2_photo` | 同a0 | H=.5 | p=.8，.4/.4/.2/.08 | .1 | 同a0 | 光照与颜色不变性 |
| `a3_blur` | 同a0 | H=.5 | 同a0 | 0 | p=.5,k7,σ=.1–2 | 模糊与成像质量鲁棒性 |
| `a4_combined` | .7–1.0 / .8–1.25 | H=.5 | p=.5，.3/.3/.2/.05 | .1 | p=.2,k7,σ=.1–1.5 | 中等组合增强 |
| `a5_stress` | .5–1.0 / .75–1.333 | H=.5 | p=.8，.5/.5/.3/.1 | .2 | p=.5,k7,σ=.1–2 | 强度上限/失败边界，不默认入选 |
| `a6_no_flip` | 同a0 | 关闭 | 同a0 | 0 | 同a0 | 判断水平翻转是否破坏方向信息 |

不使用vertical flip。`a5_stress`主要用于发现增强过强的性能崩溃点，不应只因单seed偶然最高而直接作为最终配置。

### 增强设计依据与本项目改动

- [SimCLR](https://arxiv.org/abs/2002.05709)系统实验表明，随机裁剪与颜色扰动的组合对对比表示十分关键，并使用翻转、颜色扰动和高斯模糊构造视图。因此本包把crop、photometric和blur拆开，再设置一个中等组合策略，而不是直接只比较“弱/强”两个不可解释的混合包。
- [CVRL](https://openaccess.thecvf.com/content/CVPR2021/html/Qian_Spatiotemporal_Contrastive_Video_Representation_Learning_CVPR_2021_paper.html)指出，视频空间增强应在时间维保持一致，避免对每帧施加不一致变换而破坏运动线索。本项目所有空间与颜色参数在一个clip内保持一致。
- 与上述论文不同，本数据集是细粒度工业动作且包含方向相反或阶段相反的动作。我们没有直接采用极强默认策略，而是保留`a6_no_flip`验证水平翻转是否错误抹除方向信息，并把`a5_stress`明确限定为失败边界实验。
- 本轮暂不改变两视图的时间采样，确保U3只回答空间/成像增强问题；时间视图重叠度应在增强策略锁定后作为独立阶段研究，不能与crop/颜色/blur同时改变。

## 6. 类别采样策略

| ID | 对比预训练 | 微调 | 目的 |
|---|---|---|---|
| `natural` | 原始频率 | 原始频率 | 与历史实验完全匹配 |
| `weighted_ft` | 原始频率 | sqrt-inverse WeightedRandomSampler | 单独检查微调不平衡 |
| `balanced_pre` | 每batch 16类×每类2样本 | 原始频率 | 防止take/put占64%并稳定SupLoss正样本 |
| `balanced_pre_weighted_ft` | 16×2均衡batch | sqrt-inverse weighted | 同时处理预训练和微调不平衡 |

17类但每batch只选16类是有意设计：batch仍为32，并保证入选类别各有2个样本。类别在不同batch中轮换。当前实现是单GPU、`no_ddp`，与H100运行方式一致。

## 7. 建议阶段与数量

| 阶段 | 预设/选择 | 单次标准规模 | 目的 |
|---|---|---:|---|
| U0 | prepare | 0训练 | 生成并审计t15/t17两套协议manifest |
| U1a | `confirm15_min_n_s1` | 5配置×N×seed1=5 | 先完成旧15类最小确认集合 |
| U1b | `confirm15_final` | 5配置×4人×2seed=40 | 预算允许时做多被试/seed确认 |
| U2 | `full17_entry_dev` | 18 | 比较Direct、SupLoss及不平衡处理；d0自动去除无效重复 |
| U3 | `aug17_s0_screen` | 7增强×3开发折=21 | 用s0筛选增强；采样策略用U2胜者 |
| U4 | `augmentation_loss_core`×A0+前2增强 | 4×3增强×3折=36 | 检查增强与Loss交互 |
| U5 | `augmentation_controls`×胜者增强 | 4×1×3折=12 | 严格Active–Null确认 |
| U6 | 手工锁定配置 | 配置数×seed数×最终测试对象数 | 无验证全量重训练并只测试epoch 50 |

不要同时盲目运行U1–U6。U2决定采样方式，U3决定增强候选，U4/U5决定Loss与增强组合，之后才创建U6最终manifest。

## 8. 汇总内容

`analyze_unified.py`输出：

- 每次运行的BA、Macro-F1、Accuracy；
- 配置跨被试/seed均值和标准差；
- 每类precision/recall/F1与support；
- 三种光照的分层结果；
- long-format混淆矩阵；
- 严格Active–Null配对差值；
- t17中原15类子集指标；
- take/put二类子集指标；
- manifest预期任务缺失审计。

主指标始终是BA，次指标是Macro-F1，Accuracy仅作辅助。N缺少`close`样本，因此报告同时保留测试集中实际出现类别数量和逐类support。
