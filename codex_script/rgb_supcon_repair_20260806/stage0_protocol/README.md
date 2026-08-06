# Stage 0：评估协议与数据审计

运行 `00_validate` 后运行 `01_prepare_protocol`。输出新的 M+J train、MR val、三套 leave-one-subject-out manifests、录制 run 重叠统计、RGB/crop/IMU 路径完整性，以及 motion-crop mean/std。任何 GPU 训练前必须完成。本阶段不读取 N 标签做模型选择。
