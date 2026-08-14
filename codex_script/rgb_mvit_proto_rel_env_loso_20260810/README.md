# RGB MViT-v2-S + ProtoLoss/RelLoss（旧版）环境感知 LOSO 实验包

> 2026-08-14：新增 **Stage 8 四被试探索性 LOSO**。本阶段因需要直接复用原 MR 筛选协议和旧版 runner，配置、Slurm、`.bat` 与说明均放在当前父目录，**不放入 `confirmation_runner`**。Stage 8 是对 MR/N 差异的被试异质性探索，不是无偏最终测试。

> 2026-08-13：15/17类双任务、take/put、无验证最终重训练、50轮微调和增强筛选等统一确认协议仍位于[`confirmation_runner/run_unified.py`](./confirmation_runner/run_unified.py)。原Stage 1–7保持不变；Stage 8 是上述迁移说明的明确例外。

> 2026-08-12新增：跨对象、多seed锁定确认实验、灵活提交与配对统计工具位于 [`confirmation_runner`](./confirmation_runner/README.md)。原Stage 1–7及历史输出保持不变。

## 1. 实验目标

本实验包固定使用 **Kinetics-400 初始化的 MViT-v2-S**，验证原项目中已经实现、且早于 V2 方案的 ProtoLoss 和 RelLoss 能否在强时序骨干上进一步超过：

1. K400 MViT-v2-S 直接全参数微调（Direct FT）；
2. K400 MViT-v2-S + SupLoss-only 对比预训练；
3. SupLoss + ProtoLoss；
4. SupLoss + RelLoss；
5. SupLoss + ProtoLoss + RelLoss。

实验兼顾两个假设：RGB 中每类可能存在由三种光照 `left / normal / right` 形成的子结构，因此 Prototype 数量 P=2 与 P=3 成对、均衡探索，并保留少量 P=1 对照；同时借鉴原 N-as-test 中 EMG/IMU 的有效组合，但不假定传感器上的最优超参数可以直接迁移到 RGB。

> 重要：本包不会导入或使用 `rgb_proto_rel_v2_20260804`。ProtoLoss/RelLoss 来自项目原有的旧版训练脚本；MViT 适配来自已经通过 Stage 6 验证的运行时适配层。

## 2. 目录

```text
rgb_mvit_proto_rel_env_loso_20260810/
├─ README.md
├─ ALL_EXPERIMENT_CONFIGS.md
├─ run.py                         # 统一入口
├─ config/
│  ├─ experiment_plan.json       # 所有固定参数与实验表
│  └─ selection.json             # 阶段闸门与入选配置
├─ common/                        # 旧损失接入、MViT 接入、配置辅助
├─ tools/                         # LOSO 划分、光照诊断、结果汇总
└─ scripts/
   ├─ windows/                    # Windows .bat
   └─ slurm/                      # Stanage .slurm
```

默认结果目录：

- Windows：`D:\Junxi_data\Obj2_experiments_after_260623\results\rgb_mvit_pr_env_loso_20260810`
- HPC：`/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/results/rgb_mvit_pr_env_loso_20260810`

## 3. 实验阶段与数量

| 阶段 | 作用 | 配置数 | 实际训练次数 |
|---|---|---:|---:|
| Stage 0 | 校验环境并生成四人 LOSO + 内层验证划分 | 0 | 0 |
| Stage 1 | Direct FT 与 SupLoss-only 基线 | 2 | 2（筛选折 MR） |
| Stage 2A | Proto 主筛选；P2/P3 平衡，少量 P1 | 11 | 11 |
| Stage 2B | Proto 权重局部加密，条件运行 | 4 | 4 |
| Stage 3A | Rel 主筛选与 Null-rel 对照 | 6 | 6 |
| Stage 3B | Rel 权重、Top-K、启动点加密，条件运行 | 8 | 8 |
| Stage 4 | EMG/IMU 配置迁移与 RGB 成对改写 | 11 | 11 |
| Stage 5 | 入选 Proto × Rel 组合及 Null 对照 | 6 | 6 |
| Stage 6 | 6 个候选做四人 LOSO 外层测试 | 6 | 24（6×4 人） |
| Stage 7 | SupLoss 与最终胜者补 seed 2/3 | 4 个带 seed 配置 | 16（4×4 人） |
| Stage 8 | MR/N 差异的四被试探索：P1 主线 + 一组 P3 桥接 | 8 | 32（8×4 人） |

数量口径：

- 核心筛选：Stage 1 + 2A + 3A + 4 + 5，共 **36 次**。
- 完整筛选：再加入条件阶段 2B、3B，共 **48 次**。
- 核心方案加 Stage 6：36 + 24，共 **60 次**。
- 完整 Stage 1–6：48 + 24，共 **72 次**。
- 包含可选 Stage 7 的最完整方案：72 + 16，共 **88 次训练**。
- Stage 8 单独计数，共 **32 个 fold-config pipeline**，不并入旧 Stage 1–7 的 88 次历史口径。

这里“一次训练”表示一个完整 pipeline：需要预训练的配置会执行“对比预训练 → prototype/环境诊断 → 全参数微调”；Direct FT 不执行对比预训练。Stage 6/7 还会自动测试外层留出人。

## 4. 推荐运行顺序

### 4.1 Windows

先进入包含 PyTorch/torchvision 的 Conda 环境。在命令提示符中进入本实验包，然后依次运行；如果 `python` 不在 PATH，可先执行 `set PYTHON_BIN=你的环境中python.exe完整路径`：

```bat
scripts\windows\00_validate.bat
scripts\windows\01_prepare.bat
scripts\windows\02_stage1_screen.bat
scripts\windows\03_stage2a_screen.bat
scripts\windows\05_stage3a_screen.bat
scripts\windows\07_stage4_sensor_transfer.bat
```

根据内层验证结果决定是否运行两个局部加密阶段：

```bat
scripts\windows\04_stage2b_optional.bat
scripts\windows\06_stage3b_optional.bat
```

随后编辑 `config\selection.json`：填入 Stage 2/3 入选 ID，并把 `stage5_ready` 改为 `true`。再运行：

```bat
scripts\windows\08_stage5_selected_combinations.bat
```

根据 Stage 1–5 的内层验证结果填写 `best_p2`、`best_p3`、`best_overall`、`best_p1`，将 `stage6_ready` 改为 `true`，然后执行四人 LOSO：

```bat
scripts\windows\09_stage6_loso4.bat
scripts\windows\11_summarize.bat
```

如需最终多随机种子确认，设置 `stage7_best_overall`，将 `stage7_ready=true`：

```bat
scripts\windows\10_stage7_multiseed_optional.bat
scripts\windows\11_summarize.bat
```

### 4.2 Stanage / Slurm

先检查 `scripts/slurm/common_env.sh` 中的项目根目录、数据根目录和 Conda 环境名称。在当前已经可以正常使用 GPU/PyTorch 环境的节点上，手动执行校验和数据准备：

```bash
cd /mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/codex_script/rgb_mvit_proto_rel_env_loso_20260810
bash scripts/slurm/00_validate.slurm
bash scripts/slurm/01_prepare.slurm
```

确认 `results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json` 已生成后，提交仅包含 GPU 训练的核心依赖链：

```bash
bash scripts/slurm/submit_core_screen.sh
```

该脚本不会提交 `00_validate.slurm` 或 `01_prepare.slurm`；它会先检查 prepare 的审计文件，然后只提交 Stage 1、2A、3A、4。Stage 2B、3B 为条件实验，Stage 5 必须在写入 `selection.json` 后手工提交：

```bash
sbatch scripts/slurm/04_stage2b_optional.slurm
sbatch scripts/slurm/06_stage3b_optional.slurm
sbatch scripts/slurm/08_stage5_selected_combinations.slurm
sbatch scripts/slurm/09_stage6_loso4.slurm
sbatch scripts/slurm/10_stage7_multiseed_optional.slurm
sbatch scripts/slurm/11_summarize.slurm
```

不要在所有阶段开始前就提交 Stage 5–7，因为它们会检查选择闸门，未准备好时主动终止，避免跑错配置。

### 4.3 Stage 8 四被试探索

Stage 8 不依赖 `config/selection.json`，但要求 `01_prepare` 已生成四折划分和 `protocol_audit.json`。Stanage 推荐使用提交脚本，它会提交 32 项 GPU array，并在全部成功后自动汇总：

```bash
bash scripts/slurm/submit_stage8_loso4_explore.sh
```

也可只提交 array，不自动汇总：

```bash
sbatch scripts/slurm/13_stage8_loso4_explore.slurm
```

Windows 默认依次运行全部四折；也可传入一个留出人以分批运行：

```bat
scripts\windows\13_stage8_loso4_explore.bat
scripts\windows\13_stage8_loso4_explore.bat MR
```

已存在非空 `test_results.csv` 的 fold-config 会自动跳过，因而中断后可直接重提。Stage 8 的机器配置位于 `config/experiment_plan.json`，所有输出仍写入 `results/rgb_mvit_pr_env_loso_20260810` 的 `stage8` 子目录。

### 4.4 只跑一个实验或一个留出人

Windows 示例：只跑 Stage 2A index 3、MR 为测试人，但筛选阶段只使用 MR 的 inner-val，不访问 outer-test：

```bat
scripts\windows\run_one.bat stage2a 3 MR
```

通用入口：

```text
python run.py pipeline --stage <stage> --index <index> --fold <M|J|MR|N|all> [--outer-test]
```

只有旧协议的最终 Stage 6/7，以及明确标记为探索性外测的 Stage 8，才应使用 `--outer-test`。查看实验表或仅打印命令：

```text
python run.py list --stage stage2a
python run.py pipeline --stage stage2a --index 3 --fold MR --dry-run
```

## 5. 数据划分与防泄漏规则

- Stage 1–5 固定用 `MR` 作为筛选折：其余三人的样本再按“人 × 类别 × 光照 × run”分组切出 20% inner-val。
- 同一 run 不会同时进入 inner-train 与 inner-val。
- MR 的 outer-test 在筛选时不用于选超参数。
- Stage 6 对 M/J/MR/N 各留一人，共四折；此时报告的是四人 LOSO-CV，而不是“N 永久锁定测试集”。
- Stage 7 只补 SupLoss 和 Stage 6 胜者的 seed 2/3，不重新搜索参数。
- Stage 8 会查看四个 outer-test，因此只能用于刻画被试差异；若根据其结果选择方法，后续必须另做冻结配置、多 seed 的确认，不能把同一四折结果再次称为独立最终测试。

## 6. 选择闸门怎么填

`config/selection.json` 中填写的是实验 ID，不是数组 index。建议先用验证准确率/宏 F1 排序，再检查 prototype 是否健康：

```json
{
  "stage5_ready": true,
  "best_p2_proto": "ps2_l010",
  "best_p3_proto": "ps3_l010",
  "best_p2_rel": "rl2_k3_s125",
  "best_p3_rel": "rl3_k3_s125"
}
```

Stage 5 完成后再填写 `best_p2`、`best_p3`、`best_overall`、`best_p1` 并打开 Stage 6。具体选择规则见 `ALL_EXPERIMENT_CONFIGS.md`。

## 7. 输出和诊断

每个预训练实验保存：

- 训练日志与最终/周期权重；主权重间隔为 50 epoch；
- 每 10 epoch 单独保存 prototype 诊断状态；
- RelLoss 启动前后各 10 epoch 范围内的额外边界权重；
- prototype assignment 数量、dead prototype、prototype 相似度、soft entropy 等旧版脚本已有诊断；
- 本包追加 prototype assignment 与 `left/normal/right` 的 NMI、ARI、purity 和列联表。

微调保存周期为 25 epoch，并保留最佳验证权重；汇总脚本输出逐实验 CSV、LOSO 均值/标准差以及 Markdown 总结。

当前 H100 配置统一使用预训练、微调和测试 batch size 32。相对旧 Stage 6C 的 batch size 8，预训练/微调学习率采用平方根缩放到原来的 2 倍；queue=1088、positive 数量、epoch 数和损失参数不变。若显存不足，应统一降低所有实验的 batch size，并重新评估学习率，不能只降低某几个 Proto/Rel 配置。

## 8. 判断“损失有效”的最低标准

不要只比较单个 MR 折的最高准确率。建议同时满足：

1. Stage 6 四折平均准确率或 macro-F1 高于 SupLoss-only；
2. 改善不是由单一受试者贡献，至少多数折方向一致；
3. Stage 7 补 seed 后优势仍存在，均值增益大于随机波动；
4. Null-proto/Null-rel 不应得到同等改善，否则收益可能来自训练路径或额外计算而非损失梯度；
5. prototype 不应大面积死亡或全部高度相似；P3 若真正对应光照，assignment 与 lighting 应出现稳定但不能完全压倒动作类别的关联。

完整参数与每个实验的解释见 [ALL_EXPERIMENT_CONFIGS.md](ALL_EXPERIMENT_CONFIGS.md)。

## 9. 已完成 Stage 1/2A/3A/4 的 MR 外层测试

当前筛选模型均使用 fold_MR 的 M/J/N 开发数据训练，测试清单为从未参与训练或 inner-val 的 `runtime/splits/fold_MR/outer_test.jsonl`（MR，共 385 个样本）。不要改用原 N-as-test 测试清单，因为 fold_MR 训练中已经包含 N，会产生受试者泄漏。

Stanage 使用一个 GPU 作业依次测试全部 30 个最佳验证权重并自动汇总：

```bash
sbatch scripts/slurm/12_test_completed_screen_mr.slurm
```

Windows：

```bat
scripts\windows\12_test_completed_screen_mr.bat
```

结果仍写入 `results/rgb_mvit_pr_env_loso_20260810`：逐实验总表位于 `test/fold_MR/<stage>/<experiment>/test_results.csv`，逐样本预测和详细指标位于对应 classifier run 目录，最终排名位于 `summary/outer_test_ranking.csv` 和 `summary/SUMMARY.md`。

Slurm 脚本会先检查30个最佳验证权重，并跳过已经存在非空 `test_results.csv` 的实验，因此作业中断后可直接重新提交继续测试。若需要强制重测某个实验，先移走该实验对应的 `test_results.csv`。

本次一次性测试 30 个候选主要用于检查验证趋势能否迁移到 held-out MR。由于所有候选都被同时查看，不能再根据 MR 测试结果继续调参并把同一 MR 结果当作无偏最终性能；正式结论仍需冻结配置后做四人 LOSO/多 seed。

## 10. Stage 8：四被试异质性探索（2026-08-14）

### 10.1 为什么这样设计

MR 上 `rl3_k3_s125` 相对严格配对的 `rn3_k3_s125` 呈正向改善，而 N 上方向反转，说明单一留出人的结论不足。把原 30 配置完整复制到四个人会形成 120 个 pipeline，并会在外层测试集上继续做大规模选择；计算成本和统计偏差都过高。因此 Stage 8 只保留能回答当前问题的 8 个配置：

| idx / ID | P | 配置 | 主要问题 |
|---:|---:|---|---|
| 0 `x8_d0_direct` | — | K400 Direct FT | 每个被试上的强骨干下限 |
| 1 `x8_s0_sup` | 1 | SupLoss-only | 所有旧损失的主基线 |
| 2 `x8_h00_p1_k10` | 1 | λp=0，λr=0 | H2 路径严格 Null |
| 3 `x8_h10_p1_k10` | 1 | λp=1，λr=0 | Proto-only 主效应 |
| 4 `x8_h01_p1_k10` | 1 | λp=0，λr=1 | Rel-only 主效应 |
| 5 `x8_h11_p1_k10` | 1 | λp=1，λr=1 | Proto×Rel 联合效果 |
| 6 `x8_rn3_k3_s125` | 3 | λr=0，late diff-only 路径 | MR/N 反转的严格配对 Null |
| 7 `x8_rl3_k3_s125` | 3 | λr=0.5，start125，Top-K3 | MR/N 反转的 active 配置 |

P1 的四项使用完全相同的 H2 路径：`contrastive_proto_rel`、start=50、EMA=0.5、same+diff、Top-K10、cosine，只改变 λproto/λrel。这样可以用 2×2 因子设计区分 Proto、Rel 和组合效应。

### 10.2 Prototype 数量决策

**不建议把所有配置都改成 P=1。** P=1 时每类只有一个类中心，ProtoLoss 退化为类中心约束，也不存在可解释的类内多 prototype 结构；若完全删除 P2/P3，就无法判断原 MR/N 差异是否来自多 prototype 的 Rel 几何。

本阶段采用折中方案：

- P=1 作为主线，因为当前结果没有支持 P2/P3 对应光照子结构，且同类 prototype 高度重合；
- 暂停 P=2，因为它未提供相对 P1/P3 的独特证据，继续成对铺网格的边际价值较低；
- 仅保留一组 P=3 的严格 `rn3/rl3` 配对，因为它正是触发四被试探索的配置，删除它会使新实验无法回答 MR 与 N 为什么方向相反。

若四折中 P3 active-null 差异大多接近零或方向不一致，后续可正式把多 prototype 分支降级；若多数折同向且幅度稳定，再设计匹配参数的 P1/P2/P3 专门实验，而不是恢复整个旧网格。

### 10.3 解释边界

Stage 8 使用 seed 1、每折训练人数据中的 20% inner-val 选择 `best_val_balanced.pth`，随后查看该折 outer-test。报告时应给出四折逐折结果、mean±std，以及以下配对差值：`h10-h00`、`h01-h00`、`h11-h00`、`h11-s0`、`rl3-rn3`、`rl3-s0`。由于四个外层结果都会被查看，本阶段适合描述“效应的被试依赖性”，不适合直接产生无偏的最终泛化声明。
