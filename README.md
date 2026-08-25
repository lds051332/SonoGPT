# SonoGPT

从零训练的轻量级甲状腺超声文本 Transformer。当前版本把单结节结构化所见转换成中文报告，用于学习演示，不用于临床诊断。

## 当前状态

冻结三集 generate 评估、规则抽取/质控基线、本地推理 CLI、本地网页 Demo 已完成。不要重新训练，也不要覆盖冻结产物。

| 项目 | 值 |
| --- | --- |
| 参数量 | 15,003,648 |
| 词表 | 1,807-piece BPE |
| 冻结数据 | `synthetic_v1_5k_frozen_v1` |
| 正式训练 | 5,000 Step，RTX 2060 已完成 |
| 最终验证 Token Accuracy | 0.9723 |
| 最佳 Checkpoint | `step_00004900.pt` |
| 最终 Checkpoint | `step_00005000.pt` |
| 已见模板任意训练模板匹配 | 0.9960 |
| 留出模板明示字段准确率 | 0.9997 |
| 模拟挑战集明示字段准确率 | 1.0000 |
| 规则抽取明示字段准确率 | 1.0000（四集） |
| 质控注入缺陷 recall | 1.0000 |

更完整的进度见 `IMPLEMENTATION_PROGRESS.md`。换电脑继续开发见 `COMPANY_HANDOFF.md`。文档索引：

| 文档 | 内容 |
| --- | --- |
| `docs/evaluation.md` | generate / extract / QC 三套评估怎么读 |
| `docs/baselines.md` | 模板、规则抽取、质控版本与指标 |
| `docs/inference.md` | 本地 CLI 用法与冒烟结果 |
| `docs/web.md` | 本地网页 Demo |

## 本地环境

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --extra dev
uv run pytest
```

Windows / Linux 会安装 `torch 2.12.0+cu126`。没有 NVIDIA 驱动时，评估和推理仍可使用 `--device cpu`。`uv run` 不必先 activate 虚拟环境。

## 不要重新生成的内容

以下内容已经冻结，不要覆盖：

- `data/processed/synthetic_v1_5k_candidate_v2/`
- `artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/`
- `data/releases/synthetic_v1_5k_frozen_v1.freeze.json`
- `data/challenges/simulated_human_challenge_v1.jsonl`

训练脚本默认会校验冻结记录。数据、Tokenizer 或切分被改动后，训练和评估会失败。

## 评估

公共参数写在子命令后面。已见模板、留出模板和模拟挑战集分开统计，不混合平均。

```powershell
uv run python scripts/evaluate_generate.py --device cuda
uv run python scripts/evaluate_extract_baseline.py
uv run python scripts/evaluate_qc_baseline.py
```

报告写入：

- `reports/evaluation/`：模型 generate
- `reports/baselines/`：规则抽取与质控

V1 没有训练 extract 模型。抽取成绩是规则基线，不是神经网络。

## 推理

```powershell
uv run python scripts/infer.py info --device cuda
uv run python scripts/infer.py generate --device cuda --json-file reports/inference/generate_smoke_input.json --output reports/inference/generate_smoke_output.json
uv run python scripts/infer.py extract --text "甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm。"
uv run python scripts/infer.py qc --text-file report.txt --json-file exam.json
```

`generate` 在 QC error 时默认回退到确定性模板。`extract` 只用规则。冒烟记录见 `reports/inference/`。

## 本地网页

```powershell
uv run python scripts/serve_demo.py --device cuda
```

浏览器打开 http://127.0.0.1:8000/ 。左侧是单结节表单（不是聊天框），右侧是模型报告、质控和规则抽取对照。说明见 `docs/web.md`。默认只监听本机，不上云。

## 下一步（可选）

1. 学习曲线实验（不要覆盖现有冻结版本）。
2. 如需对外访问，再用你熟悉的 Django 做前后台并部署（不要覆盖本机 FastAPI Demo 的默认 4900 Checkpoint）。
3. 真人独立编写临床挑战集（禁止进入训练）。
