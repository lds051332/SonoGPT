# 本地网页 Demo

第一阶段只提供本机可打开的网页，默认绑定 `127.0.0.1`，不上云、不公网暴露。用途是学习展示。**不能用于临床诊断。**

Django 前后台可以以后再接到云上；本地 Demo 用 FastAPI，是为了让 15M 权重在进程里常驻，避免每个请求重新加载 Checkpoint。

## 启动

```powershell
uv sync --extra dev
uv run python scripts/serve_demo.py --device cuda
```

浏览器打开：http://127.0.0.1:8000/

没有 GPU 时用 `--device cpu`。Checkpoint 缺失时页面仍可点「只看模板」，「生成报告」会返回 503，并提示按 `COMPANY_HANDOFF.md` 拷贝 `step_00004900.pt`。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 只监听本机 |
| `--port` | `8000` | 端口 |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--checkpoint` | `artifacts/training/sonogpt_16m_m3/step_00004900.pt` | 与 CLI 相同 |
| `--preload` / `--no-preload` | 默认预加载 | 启动时把模型装进显存/内存 |

`--device` 写在脚本参数里即可，没有子命令。

## 页面上输入是什么

不是聊天框，也不是超声图。左侧是 Schema v1 的单结节表单：部位、上中下、毫米尺寸、成分、回声、形状、边缘、强回声、血流、淋巴结。提交时拼成与 CLI 相同的 JSON。

五个示例按钮来自夹具/冒烟样本，点一下就能生成：

- 冒烟样本（右叶中部 8×6 mm 实性低回声）
- 囊性无回声
- 峡部囊实性（segment 自动为 `not_applicable`）
- 高于宽 + 可疑淋巴结
- 部分未提及

## 页面上输出是什么

| 区域 | 来源 |
| --- | --- |
| 中文报告 | 15M 模型 greedy 生成；QC error 时可回退模板 |
| 徽章 | 是否用模型、QC、是否与默认模板逐字相同、设备 |
| 模板对照 | `render_report`，规则，不是神经网络 |
| 规则抽取 | `rule_extract` 1.0.0，**不是** extract 任务模型 |
| JSON | 提交给模型的结构，便于核对 |

## HTTP 接口

| 方法 | 路径 | 是否加载权重 |
| --- | --- | --- |
| GET | `/` | 否 |
| GET | `/api/health` | 否 |
| GET | `/api/meta` | 否（表单字段、示例、冻结信息） |
| POST | `/api/template` | 否 |
| POST | `/api/generate` | 是 |
| POST | `/api/extract` | 否 |
| GET | `/api/docs` | OpenAPI，调试用 |

生成请求体：

```json
{
  "exam": { "...ThyroidExam..." },
  "fallback_template": true
}
```

同一时刻只跑一个 generate（进程内锁）。默认不把用户输入写入长期日志。

## 冒烟（2026-08-25，公司机 CUDA）

产物：

- `reports/web/demo_generate_smoke_20260825.json`
- `reports/web/demo_template_smoke_20260825.json`
- `reports/web/demo_health_smoke_20260825.json`

冒烟样本生成结果：`used_fallback=false`，QC 通过，报告与默认 `location_first_v2` 模板逐字相同，设备 `cuda`，Checkpoint `step_00004900.pt`。

## 不要做的事

- 不要把本页说成临床系统或 TI-RADS
- 不要把规则抽取说成模型抽取
- 不要覆盖冻结数据、Tokenizer 或正式 Checkpoint
- 不要在公司机重训 5000 Step
- 不要默认把 `--host` 改成 `0.0.0.0` 再放到公网（上云是后续 Django/部署的事）
