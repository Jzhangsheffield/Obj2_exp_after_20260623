# Stage 1：容量和优化筛选

先运行 pretrain array（5 组），再运行 diagnose array，最后 summarize。重点比较 O0→O1 的容量效应、O0→O2 的学习率效应、O3→O4 的辅助 CE 效应。只看 MR 的 512D frozen BA、有效秩和时间扰动，不做 N test。若 O4 明显胜出，后续 stage 的默认 R3D-10+cosine+CE0.2 合理；否则把胜出的 Stage 1 设置同步到后续配置再运行。
