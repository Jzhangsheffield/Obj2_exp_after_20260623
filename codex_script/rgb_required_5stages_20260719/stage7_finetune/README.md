# Stage 7：微调协议与最终测试

目的：避免一个固定 fine-tune LR 掩盖预训练表征的真实质量。比较 scratch、当前结构SupLoss、T3结构SupLoss、best proto、best rel、best proto+rel 的 head-only 与 full fine-tune。

提交前必须编辑 `config/selected_sources.json`，把 proto、rel、proto_rel 换成前面阶段依据验证集选出的 checkpoint。`01_pretrain_array.slurm` 不训练模型，只检查这些路径并写 SHA256 冻结清单。

运行：

```bash
bash slurm/submit_pipeline.sh
```

共22个微调任务：scratch 的 head/full，加五种预训练候选各自的 head-only 和 full backbone LR `1e-4/3e-4/1e-3`。T3候选在微调和测试时自动使用 `t3_lfb` backbone；其它候选使用原始 `current` backbone。

所有验证决策冻结后才能测试：

```bash
sbatch --export=ALL,ALLOW_LOCKED_TEST=YES slurm/03_test_array.slurm
```

测试完成后提交 `slurm/04_summarize.slurm`。正式结论以 `results/ft_rgb_req_s7_20260719/test/rgb_test_results_ranked.csv` 为准；不要根据测试结果返回修改 Stage 1/4/5 配置。
