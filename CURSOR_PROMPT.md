# 公司电脑 Cursor 提示词

把下面代码块原样粘贴到公司电脑的 Cursor 新对话中。

```text
你是 SonoGPT 项目的继续开发助手。请先只读阅读以下文件，不要修改代码，先理解当前状态再开始实现：

- README.md
- COMPANY_HANDOFF.md
- IMPLEMENTATION_PROGRESS.md
- docs/README.md
- docs/evaluation.md
- docs/baselines.md
- docs/inference.md
- docs/web.md
- configs/model/sonogpt_16m_candidate.json
- configs/training/sonogpt_16m_m3.json
- data/releases/synthetic_v1_5k_frozen_v1.freeze.json
- data/releases/simulated_human_challenge_v1.freeze.json
- reports/training/sonogpt_16m_m3_5000step_20260824.json
- reports/evaluation/sonogpt_16m_m3_generate_20260825.md
- reports/baselines/rule_extract_20260825.md
- reports/baselines/qc_rules_20260825.md
- reports/inference/README.md
- reports/web/README.md

项目定位：学习 Demo，甲状腺单结节文本任务，不做图像，不做临床诊断。

当前已完成：
1. Schema v1、合成数据生成器、4 套模板、防泄漏切分。
2. 数据已冻结：synthetic_v1_5k_frozen_v1。
3. Tokenizer 已冻结：1807-piece BPE。
4. 15,003,648 参数 Decoder-only Transformer。
5. 家中 RTX 2060 已完成 5000 Step 正式训练。
6. 最终验证 Loss 0.05746，Token Accuracy 0.9723。
7. 最佳验证 Loss 在 step_00004900.pt（0.05696）；评估默认用它，不要改成 5000。
8. 30 条 AI 模拟人工挑战集已独立冻结，禁止进入训练。
9. 冻结三集 generate 评估已完成（reports/evaluation/）。
10. 规则抽取基线 1.0.0、质控规则 1.0.0 已完成（reports/baselines/）。
11. 本地推理 CLI 已完成（scripts/infer.py）；V1 extract 只用规则，没有 extract 任务模型。
12. 本地网页 Demo 已完成（scripts/serve_demo.py，http://127.0.0.1:8000/）。

本机约束：
- 公司电脑是 GTX 1650，约 4GB 显存。
- 不要重新训练 5000 Step。
- 评估和推理可用 CUDA；没有 CUDA 则用 CPU。
- 不要覆盖冻结数据、Tokenizer、挑战集或正式 Checkpoint。
- 已见 / 留出 / 挑战集指标禁止混合平均。
- 挑战集不能称为真实人类或临床集。
- 网页默认只绑定 127.0.0.1，不要擅自改成公网服务。

开始前先检查这些路径是否存在：
- artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/sonogpt_bpe.model
- data/processed/synthetic_v1_5k_candidate_v2/train.jsonl
- artifacts/training/sonogpt_16m_m3/step_00004900.pt
- artifacts/training/sonogpt_16m_m3/step_00005000.pt

如果缺失，先停止并告诉我按 COMPANY_HANDOFF.md 拷贝哪些文件。不要重新生成数据。

Demo 闭环已经齐。下一步只做用户明确点名的可选工作，例如：
- 学习曲线实验（新冻结版本，不要覆盖 synthetic_v1_5k_frozen_v1）
- Django 对外前后台 / 上云（不要替换默认 4900 Checkpoint）
- 真人独立编写临床挑战集（禁止进入训练）

不要重做已经完成的 generate 评估、规则抽取、质控、推理 CLI 或本地网页 Demo，除非用户要求修复。
全程用简体中文回复。工作留痕，写好 md。不要虚构临床审核或真实患者数据。

先给出你读到的当前状态摘要，再等用户指定可选任务，或按其最新一句话开始。
```
