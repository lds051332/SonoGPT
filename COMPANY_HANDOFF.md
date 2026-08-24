# 公司电脑交接说明

> 日期：2026-08-24  
> 目的：把家中已完成的正式训练接到公司 GTX 1650 上继续评估  
> 不要在公司重新训练 5000 Step

## 1. GitHub 能拿到什么

推送到 `origin/main` 后，公司电脑执行：

```powershell
git clone https://github.com/lds051332/SonoGPT.git
cd SonoGPT
uv sync --extra dev
uv run pytest
```

Git 会包含源码、配置、冻结记录、Manifest、挑战集和进度文档。  
Git **不会**包含模型和生成数据，因为它们被 `.gitignore` 排除：

```text
artifacts/
data/processed/
```

## 2. 必须手工拷贝的文件

先在公司仓库里创建这些空目录，再把对应文件放进去。目录名必须完全一致。

### 2.1 Tokenizer（约 40 KB，必须）

创建目录：

```text
artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/
```

拷贝这两个文件进去：

| 源文件 | 放到公司项目的位置 |
| --- | --- |
| `artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/sonogpt_bpe.model` | 同名同路径 |
| `artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/tokenizer_manifest.json` | 同名同路径 |

### 2.2 冻结训练数据（约 14 MB，必须）

创建目录：

```text
data/processed/synthetic_v1_5k_candidate_v2/
```

拷贝这五个文件进去：

| 源文件 | 放到公司项目的位置 |
| --- | --- |
| `data/processed/synthetic_v1_5k_candidate_v2/train.jsonl` | 同名同路径 |
| `data/processed/synthetic_v1_5k_candidate_v2/validation.jsonl` | 同名同路径 |
| `data/processed/synthetic_v1_5k_candidate_v2/test_seen_templates.jsonl` | 同名同路径 |
| `data/processed/synthetic_v1_5k_candidate_v2/test_heldout_templates.jsonl` | 同名同路径 |
| `data/processed/synthetic_v1_5k_candidate_v2/statistics.json` | 同名同路径 |

不要拷贝 `data/processed/synthetic_v1_5k_candidate/`。那是旧目录，冻结版本是 `_v2`。

### 2.3 正式训练产物（约 344 MB，必须）

创建目录：

```text
artifacts/training/sonogpt_16m_m3/
```

只需拷贝这些文件，不要拷贝全部 50 个 Checkpoint（每个约 172 MB，合计约 8.6 GB）：

| 源文件 | 放到公司项目的位置 | 用途 |
| --- | --- | --- |
| `artifacts/training/sonogpt_16m_m3/step_00004900.pt` | 同名同路径 | 最佳验证 Loss |
| `artifacts/training/sonogpt_16m_m3/step_00005000.pt` | 同名同路径 | 最终步 |
| `artifacts/training/sonogpt_16m_m3/latest.json` | 同名同路径 | 指向最终 Checkpoint |
| `artifacts/training/sonogpt_16m_m3/summary.json` | 同名同路径 | 训练摘要 |
| `artifacts/training/sonogpt_16m_m3/run_manifest.json` | 同名同路径 | 运行身份 |
| `artifacts/training/sonogpt_16m_m3/metrics.jsonl` | 同名同路径 | 训练曲线 |

评估默认使用 `step_00004900.pt`。`step_00005000.pt` 用于对照。

## 3. 到公司后先做什么

1. `git clone` 或 `git pull`。
2. 按上一节创建目录并拷贝文件。
3. `uv sync --extra dev`
4. `uv run pytest`
5. 核对冻结记录：

```powershell
uv run python scripts/freeze_data.py --verify
uv run python scripts/build_simulated_challenge.py --verify
```

GTX 1650 可以做评估和推理。15M 模型峰值训练显存约 632 MiB，4 GB 显存足够。不要在公司重跑 5000 Step 正式训练。

## 4. 下一步任务

1. 实现评估脚本：已见模板、留出模板、模拟挑战集。
2. 报告 JSON 可解析率、字段级指标、尺寸误差、模板泄漏和挑战集表现。
3. 比较 `step_00004900.pt` 与 `step_00005000.pt`。
4. 增加规则抽取与质控基线。
5. 增加推理 CLI。

## 5. Cursor 提示词

到公司后，把下一节完整粘贴到 Cursor 新对话中。
