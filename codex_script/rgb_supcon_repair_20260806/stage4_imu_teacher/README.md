# Stage 4：冻结 IMU 教师指导 RGB

先运行 `00_cache_teacher`，确认 HPC 同步了 IMU checkpoint 与 args.json；再运行 4 组 matched 实验。X1 是同一样本特征对齐，X2 是对齐到 IMU 类中心的关系分布，X3 合并二者。只有 X1/X2 单独至少一项在 MR 上稳定改善时，才把 X3 作为最终候选。该阶段推理仍为 RGB-only，但训练使用配对 IMU，论文中必须明确说明。
