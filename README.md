# SonoGPT

从零训练的轻量级甲状腺超声文本 Transformer。当前版本把单结节结构化所见转换成中文报告，用于学习与求职演示，不用于临床诊断。

## 当前状态

正式训练已完成。下一步是在已见模板、留出模板和模拟人工挑战集上评估，而不是重新训练。

| 项目 | 值 |
| --- | --- |
| 参数量 | 15,003,648 |
| 词表 | 1,807-piece BPE |
| 冻结数据 | `synthetic_v1_5k_frozen_v1` |
| 正式训练 | 5,000 Step，RTX 2060 已完成 |
| 最终验证 Token Accuracy | 0.9723 |
| 最佳 Checkpoint | `step_00004900.pt` |
| 最终 Checkpoint | `step_00005000.pt` |

更完整的进度见 `IMPLEMENTATION_PROGRESS.md`。换电脑继续开发见 `COMPANY_HANDOFF.md`。

## 本地环境

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --extra dev
uv run pytest
```

Windows / Linux 会安装 `torch 2.12.0+cu126`。没有 NVIDIA 驱动时，评估和推理仍可使用 `--device cpu`。

## 不要重新生成的内容

以下内容已经冻结，不要覆盖：

- `data/processed/synthetic_v1_5k_candidate_v2/`
- `artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/`
- `data/releases/synthetic_v1_5k_frozen_v1.freeze.json`
- `data/challenges/simulated_human_challenge_v1.jsonl`

训练脚本默认会校验冻结记录。数据、Tokenizer 或切分被改动后，训练和评估会失败。

## 下一步

1. 在已见模板测试集、留出模板测试集和模拟挑战集上评估。
2. 增加规则抽取与质控基线。
3. 增加推理入口。
