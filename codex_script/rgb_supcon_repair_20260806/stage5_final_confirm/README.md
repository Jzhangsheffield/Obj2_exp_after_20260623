# Stage 5：三种子最终确认与锁定 N 测试

Stage 1–4 完成后，建议先对最多两个候选运行 `00_crossval_selected`。通过 `CV_STAGE` 和 `CV_EXPERIMENT` 指定候选；默认示例是 `stage2/s2_m3_dual`。然后编辑 `config/final_selection.json` 并设置 `selection_ready=true`。运行 9 个微调任务：scratch full、Sup head-only、Sup full，各 seeds 1/2/3。只有 MR validation 结果完整且配置冻结后，才设置 `ALLOW_LOCKED_TEST=YES` 并运行编号 90 的测试脚本。N 结果不能再用于回头修改候选。
