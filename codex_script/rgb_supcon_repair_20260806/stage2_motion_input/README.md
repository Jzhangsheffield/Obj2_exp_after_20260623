# Stage 2：运动区域与显式帧差

5 组实验保持优化和损失一致，只改变输入表达与是否双流。motion-crop 统计来自 Stage 0。重点判断 raw descriptor 中约 38.8% BA 的帧差信号能否被 SupLoss 网络保留下来。双流运行成本约为单流两倍；若单独帧差已胜出而双流无增益，可停止 crop-dual 的多种子扩展。
