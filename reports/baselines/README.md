# 规则基线评估留痕

日期：2026-08-25  
冻结：`synthetic_v1_5k_frozen_v1`  
挑战集：`simulated_human_challenge_v1`

这些报告由脚本生成，不要手改数字。解读见 `docs/baselines.md` 与 `docs/evaluation.md`。

| 文件 | 脚本 | 含义 |
| --- | --- | --- |
| `rule_extract_20260825.json` / `.md` | `scripts/evaluate_extract_baseline.py` | 报告 → 结构，规则抽取 1.0.0 |
| `qc_rules_20260825.json` / `.md` | `scripts/evaluate_qc_baseline.py` | 干净样本假阳性 + 11 类注入缺陷 |

摘要：

- 抽取：四集明示字段准确率均为 1.0，幻觉率 0；整条结构 Exact Match 较低是因为 `unknown` vs `not_mentioned`。
- 质控：干净样本 error 假阳性 0；注入 220 条 micro/macro recall 1.0。

不是临床验证。
