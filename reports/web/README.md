# 本地网页 Demo 留痕

日期：2026-08-25  
设备：NVIDIA GeForce GTX 1650，CUDA  
Checkpoint：`artifacts/training/sonogpt_16m_m3/step_00004900.pt`  
冻结：`synthetic_v1_5k_frozen_v1`  
页面版本：`web_demo` 1.0.0

本目录是 HTTP Demo 冒烟记录，不能代替 `reports/evaluation/` 中的冻结集评估。

| 文件 | 含义 |
| --- | --- |
| `demo_health_smoke_20260825.json` | `GET /api/health` |
| `demo_template_smoke_20260825.json` | `POST /api/template`，冒烟样本 |
| `demo_generate_smoke_20260825.json` | `POST /api/generate`，冒烟样本 |

启动：

```powershell
uv run python scripts/serve_demo.py --device cuda
```

浏览器：http://127.0.0.1:8000/

生成摘要：

- 报告：`甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm，呈非高于宽形，边缘光滑，内未见明显强回声，CDFI示结节内及周边未见明显血流信号。颈部未见明显可疑淋巴结。`
- `used_fallback`: false
- `matches_default_template`: true
- QC: passed
- 设备: cuda

未用于临床，也未保存患者数据。页面默认只监听 127.0.0.1。
