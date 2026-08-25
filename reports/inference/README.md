# 推理冒烟留痕

日期：2026-08-25  
设备：NVIDIA GeForce GTX 1650 with Max-Q Design，驱动 610.88，`torch.cuda` 可用  
Checkpoint：`artifacts/training/sonogpt_16m_m3/step_00004900.pt`  
冻结：`synthetic_v1_5k_frozen_v1`

本目录是单条 CLI 冒烟记录，不能代替 `reports/evaluation/` 中的冻结集评估。

| 文件 | 含义 |
| --- | --- |
| `generate_smoke_input.json` | 右叶中部 8×6 mm 实性低回声结节 |
| `generate_smoke_output.json` | `infer.py generate --device cuda` |
| `extract_smoke_output.json` | 对模型报告做规则抽取 |
| `qc_smoke_output.json` | 对模型报告 + 输入结构做 QC |

命令：

```powershell
uv run python scripts/infer.py generate --device cuda --json-file reports/inference/generate_smoke_input.json --output reports/inference/generate_smoke_output.json
```

结果摘要：

- 生成报告：`甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm，呈非高于宽形，边缘光滑，内未见明显强回声，CDFI示结节内及周边未见明显血流信号。颈部未见明显可疑淋巴结。`
- Checkpoint SHA-256：`497ee0595eb95d04774d8a36f11f38a404c37e0376418502ae6eb3f6b06651b3`
- 冻结记录 SHA-256：`0947163809972d3e6d5db5ea0e8957c13f6f514e778743faf66c240a9f7bd58a`
- `used_fallback`: false
- QC: passed，0 error / 0 warning
- 生成耗时约 1.75 秒（含加载后的 greedy）
- 随后 extract / qc 均通过

未用于临床，也未保存患者数据。
