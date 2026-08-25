# 文档索引

SonoGPT 是学习 Demo，不是临床系统。实现进度以仓库根目录的 `IMPLEMENTATION_PROGRESS.md` 为准。

**第一次读本仓库：** 从 [`from_zero_to_one.md`](from_zero_to_one.md) 开始。它按开发顺序把 Schema、数据、Tokenizer、模型、训练、评估和推理串成一条可跟练的路径。

| 文档 | 读什么 |
| --- | --- |
| `from_zero_to_one.md` | 从 0 到 1：大模型开发全流程学习手册 |
| `evaluation.md` | generate / 规则抽取 / 质控三套评估怎么读、报告路径、禁止混合平均 |
| `baselines.md` | 模板生成、规则抽取、质控规则版本、指标含义 |
| `inference.md` | 本地 CLI、generate/extract/qc 差异、模板回退 |
| `web.md` | 本地网页 Demo：表单输入、启动方式、HTTP 接口 |

环境：

- `README.md`：当前状态与常用命令
- Tokenizer、合成 JSONL、Checkpoint 不进 Git；缺权重时按 `from_zero_to_one.md` 阶段 0 的本地产物路径放置
