# 规则基线

SonoGPT V1 把规则系统与神经网络分开。规则负责抽取和质控；模型只做已训练的 `generate` 任务（规范 JSON → 中文报告）。这些规则用于学习 Demo 对照，不是临床诊断或 TI-RADS。

## 1. 模板生成基线

| 项目 | 值 |
| --- | --- |
| 模块 | `src/sonogpt/baselines/template_report.py` |
| 四个语序家族 | `location_first_v2`、`descriptor_first_v2`、`dimensions_first_v2`、`flow_first_v2` |
| 作用 | JSON → 确定性中文报告，不新增 Schema 中没有的信息 |

模板生成对规范 JSON 可以做到字段级零幻觉。模型在语义正确性上接近模板，但不会写出留出语序 `flow_first_v2`。

## 2. 规则抽取基线

| 项目 | 值 |
| --- | --- |
| 版本 | `rule_extract` 1.0.0 |
| 模块 | `src/sonogpt/baselines/rule_extract.py` |
| 解析器 | `src/sonogpt/evaluation/report_parser.py` |
| 方向 | 报告文本 → Schema v1 |
| 未提及 | 保持 `not_mentioned`，不猜测 |

V1 **没有**训练 extract 任务模型。`scripts/infer.py extract` 走的就是这条规则基线。

复现评估：

```powershell
uv run python scripts/evaluate_extract_baseline.py
```

报告：

- `reports/baselines/rule_extract_20260825.json`
- `reports/baselines/rule_extract_20260825.md`

2026-08-25 冻结集结果（明示字段准确率，不是整条 JSON 逐字段含 unknown 的 Exact Match）：

| 集合 | 样本 | 可解析 | 明示字段准确率 | 尺寸精确匹配 | 整条结构 Exact Match |
| --- | --- | --- | --- | --- | --- |
| 夹具 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.8000 |
| 已见模板 | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.6100 |
| 留出模板 | 500 | 1.0000 | 1.0000 | 1.0000 | 0.6100 |
| 模拟挑战集 | 30 | 1.0000 | 1.0000 | 1.0000 | 0.7667 |

整条结构 Exact Match 低于明示字段准确率，是因为金标准里的 `unknown` 被规则写成 `not_mentioned`。评估报告把这两种状态视为不同，不把它们混成“抽对了”。幻觉率为 0。

尺寸支持 `mm` 与 `cm`；仅出现 `cm` 时按 ×10 换成毫米。不允许用大容差掩盖单位错误，容差为 0.05 mm。

## 3. 质控规则

| 项目 | 值 |
| --- | --- |
| 版本 | `qc_rules` 1.0.0 |
| 模块 | `src/sonogpt/baselines/qc.py` |
| 输入 | 报告文本 + 可选结构 JSON |

规则 ID 与严重度：

| 规则 ID | 默认严重度 | 含义 |
| --- | --- | --- |
| `QC.EMPTY_REPORT` | error | 空报告 |
| `QC.UNPARSEABLE_REPORT` | error | 规则无法对齐 |
| `QC.DIAGNOSTIC_LANGUAGE` | error | 出现良恶性、癌、TI-RADS 等诊断用语 |
| `QC.TEXT_FIELD_CONTRADICTION` | error | 同一报告左右叶或形状自相矛盾 |
| `QC.TEXT_LOCATION_MISMATCH` | error | 报告侧别与结构不一致 |
| `QC.TEXT_DIMENSION_MISMATCH` | error | 报告尺寸与结构毫米值不一致 |
| `QC.TEXT_OMITS_STATED_LOCATION` | error | 结构有部位，报告没有 |
| `QC.TEXT_OMITS_STATED_COMPOSITION` | error | 结构有成分，报告没有 |
| `QC.TEXT_OMITS_STATED_ECHOGENICITY` | error | 结构有回声，报告没有 |
| `QC.TEXT_FIELD_VALUE_MISMATCH` | error | 报告写出了该字段但与结构不同 |
| `QC.TEXT_OMITS_STATED_SHAPE` 等 | warning | 结构有形状/边缘/强回声/血流/淋巴结，报告未写 |
| `QC.UNIT_MIXED_MM_CM` | warning | 同时出现 mm 与 cm |
| `QC.STRUCTURE_NEGATION_MISLABEL` | warning | 报告明示未见，结构却是 `not_mentioned` |
| `QC.STRUCTURE_INTERNAL_CONSISTENCY` | warning | 结构组合与工程一致性规则冲突 |

`passed` 只看 error，warning 不阻断。推理生成在 QC error 时回退到模板报告。

复现评估：

```powershell
uv run python scripts/evaluate_qc_baseline.py
```

报告：

- `reports/baselines/qc_rules_20260825.json`
- `reports/baselines/qc_rules_20260825.md`

2026-08-25 结果：

- 干净样本（夹具 / 已见 / 留出 / 挑战集）error 假阳性为 0，通过率 1.0。
- 夹具平均 warning 0.5，来自部分夹具结构触发工程一致性提示，不是临床结论。
- 11 类注入缺陷各 20 条，共 220 条，micro/macro recall 均为 1.0。

## 4. 与模型对照

| 能力 | 规则基线 | SonoGPT-16M（4900） |
| --- | --- | --- |
| JSON → 报告 | 模板，字段零幻觉 | 语义接近模板；留出语序 `flow_first_v2` 一次未出现 |
| 报告 → JSON | 明示字段准确率 1.0 | 未训练 extract |
| 质控 | 独立规则，可测 | 生成后调用同一套 QC |

不要用模型自己抽取自己的生成结果当作唯一成绩。Generate 评估已经用本规则解析器做交叉检查。
