# 评估总览

本文汇总已经落地的评估，避免把不同集合、不同任务混成一个准确率。所有数字均可由脚本重新生成。项目定位是学习 Demo，不是临床验证。

## 1. Generate（JSON → 报告）

脚本：

```powershell
uv run python scripts/evaluate_generate.py --device cuda
```

产物：

- `reports/evaluation/sonogpt_16m_m3_generate_20260825.json`
- `reports/evaluation/sonogpt_16m_m3_generate_20260825.md`

默认 Checkpoint `step_00004900.pt`：

| 集合 | 样本 | 逐字匹配 | 任一训练模板匹配 | 明示字段准确率 | 尺寸精确匹配 |
| --- | --- | --- | --- | --- | --- |
| 已见模板 | 1500 | 0.3320 | 0.9960 | 0.9997 | 1.0000 |
| 留出模板 | 500 | 0.0000 | 0.9960 | 0.9997 | 1.0000 |
| 模拟挑战集 | 30 | 0.0000 | 0.9667 | 1.0000 | 0.9630 |

字段指标来自独立规则解析器，不是模型自己抽自己。挑战集不与模板测试集平均。`flow_first_v2` 在生成中出现 0 次。继续默认使用 4900 而不是 5000。

## 2. Extract 规则基线（报告 → JSON）

脚本：

```powershell
uv run python scripts/evaluate_extract_baseline.py
```

产物：`reports/baselines/rule_extract_20260825.*`

明示字段准确率在夹具、已见、留出、挑战集上均为 1.0，幻觉率 0。整条结构 Exact Match 较低，是因为 `unknown` 被写成 `not_mentioned`。详见 `docs/baselines.md`。

## 3. 质控规则

脚本：

```powershell
uv run python scripts/evaluate_qc_baseline.py
```

产物：`reports/baselines/qc_rules_20260825.*`

干净样本 error 假阳性 0；11 类注入缺陷 recall 1.0。详见 `docs/baselines.md`。

## 4. 不要做的对照

- 不要把三集 generate 成绩平均后当作最终分数
- 不要把挑战集称为真实人类或临床集
- 不要用模型抽取模型生成的报告，再当作唯一质量证明
- 不要覆盖冻结数据、Tokenizer、挑战集或正式 Checkpoint
- 不要在公司电脑重跑 5000 Step 训练
- 本地网页 Demo 是单条展示，不能代替冻结集评估；见 `docs/web.md`
