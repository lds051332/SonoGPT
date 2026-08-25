# 推理 CLI

第一阶段同时提供本地 CLI 和本机网页。网页见 `docs/web.md`。用途是学习 Demo，**不能用于临床诊断**。

## 约束

- 默认 Checkpoint：`artifacts/training/sonogpt_16m_m3/step_00004900.pt`
- 先校验冻结记录，再加载 Tokenizer / 模型
- V1 只训练了 `generate`；`extract` 使用规则基线
- 生成超过 `max_seq_len` 或未产生 EOS 会报错，不静默截断
- QC 出现 error 时，`generate` 默认回退到确定性模板报告

版本：`inference` 1.0.0，Schema 1.0.0，`rule_extract` 1.0.0，`qc_rules` 1.0.0。

## 命令

公共参数写在子命令后面：

```powershell
uv run python scripts/infer.py info --device cuda
uv run python scripts/infer.py generate --device cuda --json-file reports/inference/generate_smoke_input.json --output reports/inference/generate_smoke_output.json
uv run python scripts/infer.py extract --text "甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm。"
uv run python scripts/infer.py qc --text-file report.txt --json-file exam.json
```

| 子命令 | 作用 | 是否加载权重 |
| --- | --- | --- |
| `info` | 打印模型、Tokenizer、冻结、规则版本 | 是（统计参数量） |
| `generate` | 结构 JSON → 报告 + QC + 可选模板回退 | 是 |
| `extract` | 报告 → 结构 JSON（规则） | 否 |
| `qc` | 报告 + 可选结构 → 规则发现 | 否 |

`--json` / `--text` 可以是内联字符串；`-` 表示从 stdin 读取。`--no-fallback` 关闭模板回退。`--output path.json` 在打印到 stdout 的同时写入文件。

## generate 处理链

```text
结构 JSON
  → Schema 校验
  → 规范序列化
  → Greedy 生成
  → QC（报告 vs 输入结构）
  → 无 error 则返回模型报告
  → 有 error 或生成失败则模板回退，并记录 fallback_reason
```

返回字段包括：`report`、`used_fallback`、`qc`、`checkpoint_sha256`、`freeze_record_sha256`、版本信息。默认不把用户输入写入长期日志。

## extract 处理链

```text
报告文本
  → 规则抽取
  → 对抽取结构做 QC
  → 返回 exam JSON、parseable、qc
```

`model_used` 恒为 `false`，`extract_task_trained` 恒为 `false`。

## 公司机冒烟（2026-08-25）

设备：GTX 1650，CUDA，Checkpoint 4900。

输入：`reports/inference/generate_smoke_input.json`（右叶中部 8×6 mm 实性低回声）。

输出：

- `reports/inference/generate_smoke_output.json`
- `reports/inference/extract_smoke_output.json`
- `reports/inference/qc_smoke_output.json`

结果：

- 模型报告与 `location_first_v2` 模板一致
- `used_fallback=false`，QC `passed=true`
- 用同一报告做 extract / qc，均可解析且通过

这是单条冒烟，不是完整测试集。完整 generate 评估见 `reports/evaluation/`。
