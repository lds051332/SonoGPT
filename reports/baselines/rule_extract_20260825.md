# SonoGPT 规则抽取基线评估

- Created (UTC): `2026-08-25T02:32:15.859521+00:00`
- Extractor: `rule_extract` `1.0.0`
- Freeze ID: `synthetic_v1_5k_frozen_v1`
- Challenge freeze ID: `simulated_human_challenge_v1`

这是 **extract** 方向（报告 → 结构）的规则基线，不使用神经网络。
已见模板、留出模板、挑战集和夹具分开报告，没有混合平均。
本结果用于学习 Demo 对照，不是临床验证。

## fixtures

- Samples: `50`
- Parseable: `1.0000`
- Exact structure match: `0.8000`
- Mentioned-field accuracy: `1.0000`
- Hallucination rate: `0.0000`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`

## test_seen_templates

- Samples: `1500`
- Parseable: `1.0000`
- Exact structure match: `0.6100`
- Mentioned-field accuracy: `1.0000`
- Hallucination rate: `0.0000`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`

## test_heldout_templates

- Samples: `500`
- Parseable: `1.0000`
- Exact structure match: `0.6100`
- Mentioned-field accuracy: `1.0000`
- Hallucination rate: `0.0000`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`

## simulated_human_challenge

- Samples: `30`
- Parseable: `1.0000`
- Exact structure match: `0.7667`
- Mentioned-field accuracy: `1.0000`
- Hallucination rate: `0.0000`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`

