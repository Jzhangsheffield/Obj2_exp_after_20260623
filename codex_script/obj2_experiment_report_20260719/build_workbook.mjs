import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const ROOT = "D:/Junxi_data/Obj2_experiments_after_260623";
const OUT = `${ROOT}/analysis/obj2_experiment_report_20260719`;
const TABLES = `${OUT}/tables`;
const QA = `${OUT}/qa_workbook`;
const OUTPUT = `${OUT}/Obj2_实验结果汇总_扩充版_20260720.xlsx`;

const SOURCES = [
  ["Family Summary", "family_summary.csv"],
  ["Top Models", "top_models.csv"],
  ["Selected Metrics", "selected_best_available.csv"],
  ["Strict Last", "strict_last_checkpoint.csv"],
  ["CL Configs", "cl_configs_deduplicated.csv"],
  ["FT Runs", "ft_training_runs.csv"],
  ["Class Recall", "per_class_recall.csv"],
  ["Issues", "quality_issues.csv"],
  ["Diagnostics", "model_diagnostics.csv"],
  ["Checkpoint Pairs", "checkpoint_pairs.csv"],
  ["Module Summary", "module_method_summary.csv"],
  ["Module Pairs", "module_matched_pairs.csv"],
  ["Selected Modules", "selected_module_models.csv"],
  ["Module Per Class", "selected_module_per_class.csv"],
  ["Class Effects", "class_effect_matrix.csv"],
  ["Effect Summary", "class_effect_summary.csv"],
  ["Bootstrap CIs", "paired_bootstrap_summary.csv"],
  ["Feature Diagnostics", "targeted_feature_diagnostics.csv"],
  ["Reliability", "reliability_bins_selected.csv"],
  ["Metric Definitions", "metric_definitions.csv"],
  ["Figure Guide", "figure_reading_guide.csv"],
];

const NAVY = "#17365D";
const BLUE = "#2F75B5";
const PALE_BLUE = "#D9EAF7";
const PALE_GREEN = "#E2F0D9";
const PALE_AMBER = "#FFF2CC";
const PALE_RED = "#FCE4D6";
const LIGHT = "#F3F6F9";
const WHITE = "#FFFFFF";
const TEXT = "#1F2937";

function colLetter(index0) {
  let n = index0 + 1;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function cleanSheetName(name) {
  return name.replace(/[\\/:*?\[\]]/g, "_").slice(0, 31);
}

function formatImportedSheet(sheet) {
  const used = sheet.getUsedRange();
  const values = used.values || [];
  const rows = values.length;
  const cols = rows ? values[0].length : 0;
  if (!rows || !cols) return;

  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Calibri", size: 10, color: TEXT };
  used.format.verticalAlignment = "top";
  used.format.wrapText = false;
  used.format.autofitColumns();
  used.format.autofitRows();

  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format.fill = NAVY;
  header.format.font = { name: "Calibri", size: 10, bold: true, color: WHITE };
  header.format.wrapText = true;
  header.format.rowHeight = 34;

  for (let c = 0; c < cols; c += 1) {
    const h = String(values[0][c] ?? "").toLowerCase();
    const col = sheet.getRangeByIndexes(0, c, rows, 1);
    if (/path|json|manifest|directory|run_dir|summary_csv|weight/.test(h)) {
      col.format.columnWidth = 28;
      col.format.wrapText = true;
    } else if (/run|source|issue|evidence|impact|recommended|label|config/.test(h)) {
      col.format.columnWidth = 22;
      col.format.wrapText = true;
    } else {
      col.format.columnWidth = Math.min(18, Math.max(10, String(values[0][c] ?? "").length + 2));
    }
    if (/acc|f1|ece|confidence|error_rate|delta|momentum|learning_rate|weight_decay|loss$/.test(h)) {
      sheet.getRangeByIndexes(1, c, Math.max(1, rows - 1), 1).setNumberFormat("0.0000");
    }
    if (/epoch|seed|rank|n_models|num_|queue|topk|prototype|target_len|batch_size/.test(h)) {
      sheet.getRangeByIndexes(1, c, Math.max(1, rows - 1), 1).setNumberFormat("0");
    }
  }

  if (rows > 1) {
    sheet.getRangeByIndexes(1, 0, rows - 1, cols).conditionalFormats.addCustom("=MOD(ROW(),2)=0", { fill: "#F8FAFC" });
  }
}

function rowIndexByHeaders(values, criteria) {
  const headers = values[0].map((v) => String(v));
  const idx = Object.fromEntries(headers.map((h, i) => [h, i]));
  for (let r = 1; r < values.length; r += 1) {
    let ok = true;
    for (const [key, expected] of Object.entries(criteria)) {
      const actual = values[r][idx[key]];
      if (String(actual) !== String(expected)) { ok = false; break; }
    }
    if (ok) return { excelRow: r + 1, headers, idx };
  }
  throw new Error(`No matching row for ${JSON.stringify(criteria)}`);
}

async function main() {
  await fs.mkdir(QA, { recursive: true });
  const [firstSheetName, firstFilename] = SOURCES[0];
  const firstCsv = await fs.readFile(path.join(TABLES, firstFilename), "utf8");
  const workbook = await Workbook.fromCSV(firstCsv, { sheetName: firstSheetName });
  for (const [sheetName, filename] of SOURCES.slice(1)) {
    const csv = await fs.readFile(path.join(TABLES, filename), "utf8");
    await workbook.fromCSV(csv, { sheetName });
  }
  for (const [sheetName] of SOURCES) formatImportedSheet(workbook.worksheets.getItem(sheetName));

  const overview = workbook.worksheets.add("Overview");

  const definitions = workbook.worksheets.add("Workbook Guide");
  definitions.getRange("A1:C10").values = [
    ["字段/规则", "定义", "使用建议"],
    ["balanced accuracy", "各类别 recall 的宏平均；本报告主指标", "类别不平衡时优先于普通 accuracy"],
    ["best available", "优先 best_val；缺失时回退 last", "用于覆盖全部实验，但需结合 checkpoint 字段"],
    ["strict last", "只比较 last.pth 测试", "用于 checkpoint 口径一致的稳健核对"],
    ["full", "加载预训练权重并更新 backbone 与分类头", "RGB 的主要有效模式"],
    ["head_only", "冻结 backbone，仅训练分类头", "IMU/EMG 可行；RGB 当前明显不足"],
    ["scratch", "不使用对比学习预训练", "同实验族的主要基线"],
    ["pp", "百分点；例如 0.60→0.65 为 +5 pp", "不是相对百分比"],
    ["model×sample", "跨多个模型视图累计的样本事件", "高置信错误计数不是唯一原始样本数"],
    ["统计限制", "当前 FT 非空 seed 仅为 1", "1–2 pp 差异不应视为确定收益"],
  ];
  formatImportedSheet(definitions);
  definitions.getRange("A2:C10").format.rowHeight = 38;
  definitions.getRange("A2:C10").format.wrapText = true;
  definitions.getRange("A:A").format.columnWidth = 18;
  definitions.getRange("B:C").format.columnWidth = 38;
  const metricDefinitions = workbook.worksheets.getItem("Metric Definitions");
  metricDefinitions.getRange("A2:D14").format.wrapText = true;
  metricDefinitions.getRange("A2:D14").format.rowHeight = 48;
  metricDefinitions.getRange("A:A").format.columnWidth = 24;
  metricDefinitions.getRange("B:B").format.columnWidth = 42;
  metricDefinitions.getRange("C:C").format.columnWidth = 38;
  metricDefinitions.getRange("D:D").format.columnWidth = 44;
  const figureGuide = workbook.worksheets.getItem("Figure Guide");
  figureGuide.getRange("A2:C9").format.wrapText = true;
  figureGuide.getRange("A2:C9").format.rowHeight = 60;
  figureGuide.getRange("A:A").format.columnWidth = 30;
  figureGuide.getRange("B:C").format.columnWidth = 52;
  const effectSummary = workbook.worksheets.getItem("Effect Summary");
  effectSummary.getRange("I2:J100").format.wrapText = true;
  effectSummary.getRange("I:J").format.columnWidth = 44;
  effectSummary.getRange("A2:J100").format.rowHeight = 42;
  const issuesSheet = workbook.worksheets.getItem("Issues");
  issuesSheet.getRange("A2:E7").format.rowHeight = 64;
  issuesSheet.getRange("A2:E7").format.wrapText = true;

  const top = workbook.worksheets.getItem("Top Models");
  const topValues = top.getUsedRange().values;
  const topHeaders = topValues[0].map(String);
  const balCol = topHeaders.indexOf("test_balanced_acc");
  const sourceCol = topHeaders.indexOf("source_key");
  const selectedModulesSheet = workbook.worksheets.getItem("Selected Modules");
  const selectedModulesValues = selectedModulesSheet.getUsedRange().values;
  const selectedModulesHeaders = selectedModulesValues[0].map(String);
  const selectedModulesBalCol = selectedModulesHeaders.indexOf("test_balanced_acc");

  overview.showGridLines = false;
  overview.getRange("A1:F2").merge();
  overview.getRange("A1").values = [["Obj2 多模态对比学习与微调实验汇总"]];
  overview.getRange("A1:F2").format.fill = NAVY;
  overview.getRange("A1:F2").format.font = { name: "Calibri", size: 20, bold: true, color: WHITE };
  overview.getRange("A1:F2").format.verticalAlignment = "center";
  overview.getRange("A1:F2").format.horizontalAlignment = "left";

  overview.getRange("A4:B9").values = [
    ["数据覆盖", "数量"],
    ["去重后 CL 配置", 234],
    ["FT 训练运行", 473],
    ["测试汇总记录", 667],
    ["最佳可用运行/相机视图", 482],
    ["可用逐样本 CSV", 482],
  ];
  overview.getRange("A4:B4").format.fill = BLUE;
  overview.getRange("A4:B4").format.font = { bold: true, color: WHITE };
  overview.getRange("A4:B9").format.borders = { color: "#CBD5E1", style: "continuous", weight: 1 };
  overview.getRange("A4:A9").format.columnWidth = 25;
  overview.getRange("B4:B9").format.columnWidth = 14;

  overview.getRange("D4:F9").values = [
    ["结论", "状态", "含义"],
    ["IMU", "明确收益", "except_take_put 最佳约 +7.9 pp；take_put 约 +4.5 pp"],
    ["EMG", "中等收益", "最佳约 +6 pp，但 ECE 高、绝对性能仍低"],
    ["RGB 基础", "未稳定胜出", "full 可局部提升；head-only 明显失败"],
    ["RGB Round-2", "最有希望", "弱/共享增强 + backbone LR 3e-4 达 0.624"],
    ["RGB 双相机", "当前失败", "最佳预训练均值 0.542，低于 scratch 0.624"],
  ];
  overview.getRange("D4:F4").format.fill = BLUE;
  overview.getRange("D4:F4").format.font = { bold: true, color: WHITE };
  overview.getRange("D4:F9").format.borders = { color: "#CBD5E1", style: "continuous", weight: 1 };
  overview.getRange("D4:D9").format.columnWidth = 20;
  overview.getRange("E4:E9").format.columnWidth = 16;
  overview.getRange("F4:F9").format.columnWidth = 42;
  overview.getRange("D5:F9").format.wrapText = true;
  overview.getRange("E5:E5").format.fill = PALE_GREEN;
  overview.getRange("E6:E6").format.fill = PALE_AMBER;
  overview.getRange("E7:E7").format.fill = PALE_RED;
  overview.getRange("E8:E8").format.fill = PALE_GREEN;
  overview.getRange("E9:E9").format.fill = PALE_RED;

  overview.getRange("A12:F12").merge();
  overview.getRange("A12").values = [["关键风险与解释边界"]];
  overview.getRange("A12:F12").format.fill = NAVY;
  overview.getRange("A12:F12").format.font = { bold: true, color: WHITE, size: 12 };
  overview.getRange("A13:F16").merge();
  overview.getRange("A13").values = [[
    "• 当前非空 seed 仅为 1，无法估计方差。\n" +
    "• RGB scratch 在不同实验族间出现约 6–7 pp 波动，应统一代码版本、确定性设置和共享 checkpoint。\n" +
    "• best_val 与 last 测试口径不一致；请同时查看 Selected Metrics 与 Strict Last。\n" +
    "• 本工作簿用于结果追溯；正式解释与图表见同目录 DOCX 报告。"
  ]];
  overview.getRange("A13:F16").format.fill = LIGHT;
  overview.getRange("A13:F16").format.wrapText = true;
  overview.getRange("A13:F16").format.verticalAlignment = "top";
  overview.getRange("A13:F16").format.font = { size: 11, color: TEXT };

  overview.getRange("H2:K2").values = [["模态", "Scratch", "Full", "Head-only"]];
  const modalities = ["rgb", "emg", "imu"];
  overview.getRange("H3:H5").values = modalities.map((m) => [m.toUpperCase()]);
  const formulaRows = [];
  for (const modality of modalities) {
    const refs = [];
    for (const mode of ["scratch", "full", "head_only"]) {
      const found = rowIndexByHeaders(topValues, { dataset_scope: "except_take_put", modality, finetune_mode: mode, rank: 1 });
      refs.push(`='Top Models'!${colLetter(balCol)}${found.excelRow}`);
    }
    formulaRows.push(refs);
  }
  overview.getRange("I3:K5").formulas = formulaRows;
  overview.getRange("H2:K2").format.fill = BLUE;
  overview.getRange("H2:K2").format.font = { bold: true, color: WHITE };
  overview.getRange("H2:K5").format.borders = { color: "#CBD5E1", style: "continuous", weight: 1 };
  overview.getRange("I3:K5").setNumberFormat("0.000");
  overview.getRange("H2:K5").format.columnWidth = 15;

  const chart = overview.charts.add("bar", overview.getRange("H2:K5"));
  chart.title = "except_take_put：各模态最佳 balanced accuracy";
  chart.hasLegend = true;
  chart.legend = { position: "bottom" };
  chart.xAxis = { axisType: "textAxis" };
  chart.yAxis = { numberFormatCode: "0.00", minimumScale: 0, maximumScale: 0.75 };
  chart.setPosition("H8", "P23");

  overview.getRange("A27:D27").values = [["模态", "Scratch", "SupCon", "最佳自研模块"]];
  overview.getRange("A28:A30").values = modalities.map((m) => [m.toUpperCase()]);
  const moduleFormulaRows = [];
  for (const modality of modalities) {
    const refs = [];
    for (const role of ["scratch", "supcon", "module"]) {
      const found = rowIndexByHeaders(selectedModulesValues, { modality, model_role: role });
      refs.push(`='Selected Modules'!${colLetter(selectedModulesBalCol)}${found.excelRow}`);
    }
    moduleFormulaRows.push(refs);
  }
  overview.getRange("B28:D30").formulas = moduleFormulaRows;
  overview.getRange("A27:D27").format.fill = BLUE;
  overview.getRange("A27:D27").format.font = { bold: true, color: WHITE };
  overview.getRange("A27:D30").format.borders = { color: "#CBD5E1", style: "continuous", weight: 1 };
  overview.getRange("B28:D30").setNumberFormat("0.000");
  overview.getRange("A27:D30").format.columnWidth = 18;
  const moduleChart = overview.charts.add("bar", overview.getRange("A27:D30"));
  moduleChart.title = "匹配基线与最佳自研模块";
  moduleChart.hasLegend = true;
  moduleChart.legend = { position: "bottom" };
  moduleChart.xAxis = { axisType: "textAxis" };
  moduleChart.yAxis = { numberFormatCode: "0.00", minimumScale: 0, maximumScale: 0.70 };
  moduleChart.setPosition("H26", "P42");

  overview.getRange("A18:F18").merge();
  overview.getRange("A18").values = [["工作表导航"]];
  overview.getRange("A18:F18").format.fill = NAVY;
  overview.getRange("A18:F18").format.font = { bold: true, color: WHITE };
  overview.getRange("A19:F24").values = [
    ["Family Summary", "实验族聚合与最佳配置", "Top Models", "各任务/模态/模式排名", "Selected Metrics", "best available 明细"],
    ["Strict Last", "严格 last checkpoint 对照", "CL Configs", "去重后的预训练配置", "FT Runs", "训练日志与配置汇总"],
    ["Class Recall", "类别 recall 与相对 scratch 变化", "Issues", "数据质量与实验风险", "Diagnostics", "校准、过拟合与置信度"],
    ["Checkpoint Pairs", "best_val 与 last 的逐运行差异", "Module Summary", "实验簇×方法总体表现", "Selected Modules", "每模态匹配三模型"],
    ["颜色提示", "绿色=较明确收益", "", "黄色=需谨慎", "", "红色=失败/高风险"],
    ["生成日期", "2026-07-20", "主指标", "balanced accuracy", "seed", "1"],
  ];
  overview.getRange("A19:F24").format.borders = { color: "#CBD5E1", style: "continuous", weight: 1 };
  overview.getRange("A19:F24").format.wrapText = true;
  overview.getRange("A19:F24").format.columnWidth = 19;
  overview.getRange("A19:F24").format.rowHeight = 34;

  overview.getRange("A1:P42").format.font = { name: "Calibri", color: TEXT };
  overview.getRange("A1:F2").format.font = { name: "Calibri", size: 20, bold: true, color: WHITE };
  overview.freezePanes.freezeRows(2);

  const overviewInspect = await workbook.inspect({ kind: "region", sheetId: "Overview", range: "A1:P42", maxChars: 9000 });
  await fs.writeFile(path.join(QA, "overview_inspect.json"), JSON.stringify(overviewInspect, null, 2), "utf8");
  const formulaInspect = await workbook.inspect({ kind: "formula", sheetId: "Overview", range: "A1:P42", maxChars: 5000, options: { maxResults: 100 } });
  await fs.writeFile(path.join(QA, "formula_inspect.json"), JSON.stringify(formulaInspect, null, 2), "utf8");
  const errorScan = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 200 }, summary: "final formula error scan" });
  await fs.writeFile(path.join(QA, "error_scan.json"), JSON.stringify(errorScan, null, 2), "utf8");

  const exportBlob = await SpreadsheetFile.exportXlsx(workbook);
  await exportBlob.save(OUTPUT);

  const renderRanges = {
    "Overview": "A1:P42",
    "Family Summary": "A1:R25",
    "Top Models": "A1:T18",
    "Selected Metrics": "A1:T18",
    "Strict Last": "A1:T18",
    "CL Configs": "A1:T18",
    "FT Runs": "A1:T18",
    "Class Recall": "A1:P24",
    "Issues": "A1:E12",
    "Diagnostics": "A1:T18",
    "Checkpoint Pairs": "A1:T18",
    "Module Summary": "A1:T18",
    "Module Pairs": "A1:T18",
    "Selected Modules": "A1:T18",
    "Module Per Class": "A1:Q24",
    "Class Effects": "A1:R24",
    "Effect Summary": "A1:J24",
    "Bootstrap CIs": "A1:M12",
    "Feature Diagnostics": "A1:K24",
    "Reliability": "A1:M24",
    "Metric Definitions": "A1:D18",
    "Figure Guide": "A1:C12",
    "Workbook Guide": "A1:C10",
  };
  for (const [sheetName, range] of Object.entries(renderRanges)) {
    const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    await fs.writeFile(path.join(QA, `${cleanSheetName(sheetName)}.png`), bytes);
  }

  console.log(OUTPUT);
}

await main();
