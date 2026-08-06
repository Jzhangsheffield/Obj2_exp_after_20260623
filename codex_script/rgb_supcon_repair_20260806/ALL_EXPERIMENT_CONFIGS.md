# 全部实验配置、变量定义与判定规则

## A. 固定数据协议

- 任务：`tier1`，配置为 15 类，排除 take/put。
- 摄像头：`00143`；motion-crop 字段为 `rgb_cam_00143_motion_crop_m32`。
- 原始 train 与 val 合并后按受试者重建划分：M+J 为筛选训练集，MR 为完全未见受试者验证集。
- 同时生成 leave-one-subject-out 的 M、J、MR 三套清单，供最终两个候选补做稳健性检查。
- N 始终是锁定测试集，不能用于模型选择、超参数选择或 checkpoint 选择。
- Stage 0 按 `person/action/run` 审计录制级重叠；run 从 `original_key` 中删除 `_clip_...` 后得到。

为什么这样改：旧 train/val 同时含 M/J/MR，且约 23.76% 的 val 样本来自 train 已出现的 recording run。旧 val 既不是跨人验证，也不是严格的录制级独立验证，容易错误选择静态外观特征较强的模型。

## B. 所有预训练共同参数

| 参数 | 数值 | 说明 |
|---|---:|---|
| 主损失 | SupLoss-only | Stage 1–4 不启用 ProtoLoss/RelLoss，先修复 RGB 基础表征 |
| epochs | 200 | 所有筛选预训练一致 |
| optimizer | AdamW | 与原 RGB/IMU 对照一致 |
| learning rate | 1e-3 | warm-up 实验从较小值线性升到此值 |
| weight decay | 1e-4 | 固定 |
| projection dim | 128 | MoCo/SupLoss 投影空间 |
| queue | 1088 | 可被 batch 64 与 32 整除 |
| temperature | 0.07 | SupLoss 温度 |
| batch | 64 | 32 帧实验为 32，避免显存溢出 |
| temporal views | shared | 两个增强视图使用同一帧索引，避免动作阶段错配 |
| checkpoint | 每 50 epoch | 保存 0050/0100/0150/0200，自动续训 |
| image size | 224×224 | 固定 |

空间增强：RRC scale 0.85–1.0、ratio 0.9–1.1、水平翻转 0.5、垂直翻转 0、ColorJitter 概率 0.2 且 brightness/contrast/saturation/hue 为 0.1/0.1/0.1/0.02、灰度化 0、blur 概率 0.1、kernel 5、sigma 0.1–1.0。motion-crop 先按训练集均值 pad 成方形，再 resize/crop，并使用 Stage 0 实测 mean/std。

## C. Stage 1：容量与优化

| index | ID | depth | LR | warm-up | 辅助 CE | 唯一变化与目的 |
|---:|---|---:|---|---:|---:|---|
| 0 | s1_o0_r18_step | 18 | 50/100/150 各降 10× | 0 | 0 | 严格复现当前控制 |
| 1 | s1_o1_r10_step | 10 | 同上 | 0 | 0 | 只减小容量，检验 33M 模型是否对约 1k clip 过大 |
| 2 | s1_o2_r18_cos | 18 | cosine | 10 | 0 | 只修正“epoch 50 过早降到 1e-4” |
| 3 | s1_o3_r10_cos | 10 | cosine | 10 | 0 | 小模型与平滑优化组合 |
| 4 | s1_o4_r10_cos_ce02 | 10 | cosine | 10 | 0.2 | 在 512D backbone 上直接增加类别监督，防止仅投影头拟合 SupLoss |

辅助 CE 只在预训练期使用；最终推理不保留该分类头。它不是用 val 标签训练，仍只使用 M+J train 标签。

## D. Stage 2：运动输入

| index | ID | 输入 | backbone | 说明 |
|---:|---|---|---|---|
| 0 | s2_m0_rgb | 整帧 RGB | R3D-10 | Stage 2 matched control |
| 1 | s2_m1_crop | motion-crop RGB | R3D-10 | 放大局部操作区域，必须重新预训练，不能直接把 crop 喂给整帧 checkpoint |
| 2 | s2_m2_absdiff | 相邻标准化帧绝对差 | R3D-10 | 去除大部分静态外观，验证运动本身是否更可分 |
| 3 | s2_m3_dual | 整帧 RGB + 绝对帧差 | 双 R3D-10 | appearance/motion 独立编码，512D 融合；最高优先级候选 |
| 4 | s2_m4_crop_dual | crop RGB + crop 帧差 | 双 R3D-10 | 同时集中空间区域与显式运动 |

`absdiff` 定义为 `d[0]=0, d[t]=abs(x[t]-x[t-1])`。`rgb_absdiff` 形成 6 通道输入，但模型不会用一个 6 通道卷积混合；前 3 通道和后 3 通道分别进入 appearance/motion 分支，随后拼接并投影到 512D。

## E. Stage 3：时间范围与时间聚合

| index | ID | 帧数 | 时间结构 | batch | 目的 |
|---:|---|---:|---|---:|---|
| 0 | s3_t0_r10_f16 | 16 | 当前全局池化 | 64 | matched control |
| 1 | s3_t1_r10_f32 | 32 | 当前全局池化 | 32 | 只增加观察范围；若无收益说明问题不只是帧数不足 |
| 2 | s3_t2_attn_f16 | 16 | T3 + temporal attention | 64 | 保留约 8 个时间位置并学习加权聚合 |
| 3 | s3_t3_attn_f32 | 32 | T3 + temporal attention | 32 | 同时增加范围与显式顺序容量 |

T3 将 stem maxpool、layer3、layer4 的时间 stride 改为 1；空间 stride 不变。与旧 T3 不同，本阶段不立即对保留的时间位置做全局平均，而是在空间平均后用可学习 attention 聚合。因此它检验的是“保留时间 + 使用时间”，而不只是保留时间尺寸。

## F. Stage 4：冻结 IMU 教师

教师固定为严格 IMU SupLoss-only checkpoint。先对 M+J train 按 manifest 顺序缓存 512D 特征和 15 个类中心；教师全程 `eval` 且不更新。RGB 推理时不需要 IMU。

| index | ID | 实例余弦权重 | 关系 KL 权重 | 含义 |
|---:|---|---:|---:|---|
| 0 | s4_x0_rgb_control | 0 | 0 | 同训练设置的纯 RGB 控制 |
| 1 | s4_x1_instance | 0.1 | 0 | RGB 512D 经线性 adapter 后与同一样本 IMU 512D 对齐 |
| 2 | s4_x2_relation | 0 | 0.5 | 对齐样本到 15 个 IMU 类中心的相似度分布，温度 0.1 |
| 3 | s4_x3_both | 0.1 | 0.5 | 两种教师信号同时使用 |

实例损失为 `1-cos(adapter(f_rgb), f_imu)`。关系损失是教师/学生对 IMU 类中心相似度分布之间的 KL；学生和教师都先 L2 normalize。四组都保留 SupLoss 和 0.2 辅助 CE，确保 Null control 完全匹配。

## G. 每个 checkpoint 必做诊断

分别对 512D backbone 与 128D projection 输出：M+J train 和 MR val 的 cosine silhouette、有效秩、前 5 主成分解释方差、类内距离、类中心间距离、between/within、M+J 训练的 balanced logistic probe 在 MR 上的 BA/Macro-F1、1-NN BA。

MR 还做 reverse、shuffle、repeat-center 三种时间扰动，报告原始/扰动特征平均余弦相似度与 frozen BA 降幅。重点看 512D；PCA/UMAP 只能辅助，不能替代高维指标。

进入最终微调的最低门槛（相对 Stage 1 O0）：

- MR frozen linear BA 至少提高 5 个百分点；
- MR effective rank 明显高于旧 RGB 约 2 的水平，且 top-5 variance 不再接近 100%；
- reverse 或 shuffle 至少造成约 5 个百分点 BA 下降，或平均特征 cosine 明显低于当前约 0.99；
- train 提升但 MR 不升的配置判为过拟合，不进入 Stage 5。

## H. Stage 5：三种子确认

先在 `final_selection.json` 固定一个候选。每个 seed 运行：

1. `scratch_full_s*`：相同 backbone/input/split，从随机初始化监督训练 100 epoch；
2. `sup_head_s*`：加载所选 SupLoss checkpoint，只训练分类头 25 epoch；
3. `sup_full_s*`：加载 checkpoint，全模型微调 100 epoch，backbone LR 3e-4、head LR 1e-3。

微调 AdamW、weight decay 1e-4、周期 checkpoint 每 25 epoch；full 的 milestone 为 50/75，head 只有 25 epoch，因此不会触发 50/75 衰减。模型选择只看 MR validation balanced accuracy。最终 N 只运行一次完整的 9 个模型，并按三种模式分别报告均值、样本标准差和每类召回。

## I. 与 ProtoLoss V2 / RelLoss V2 的衔接

本包故意先不把 Proto/Rel 加进 Stage 1–4，因为旧证据显示基础 RGB 表征本身低秩且不含时间顺序；在坏表示上增加原型约束无法判断损失设计是否有效。Stage 5 证明 repaired SupLoss 至少超过 matched scratch 后，再把所选 backbone/input/split 移植到现有 V2 包，做 Sup、+ProtoV2、+RelV2、+Both 和 null-gradient 的同种子 factorial。若 repaired SupLoss 仍不超过 scratch，应继续修复视觉表征，而不是调大 λproto/λrel。
