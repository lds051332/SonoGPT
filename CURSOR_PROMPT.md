# 公司电脑 Cursor 提示词

把下面代码块原样粘贴到公司电脑的 Cursor 新对话中。

```text
你是 SonoGPT 项目的继续开发助手。请先只读阅读以下文件，不要修改代码，先理解当前状态再开始实现：

- README.md
- COMPANY_HANDOFF.md
- IMPLEMENTATION_PROGRESS.md
- configs/model/sonogpt_16m_candidate.json
- configs/training/sonogpt_16m_m3.json
- data/releases/synthetic_v1_5k_frozen_v1.freeze.json
- data/releases/simulated_human_challenge_v1.freeze.json
- reports/training/sonogpt_16m_m3_5000step_20260824.json
- src/sonogpt/data/freeze.py
- src/sonogpt/evaluation/challenge.py
- src/sonogpt/training/trainer.py
- scripts/train_model.py

项目定位：学习与求职 Demo，甲状腺单结节文本任务，不做图像，不做临床诊断。

当前已完成：
1. Schema v1、合成数据生成器、4 套模板、防泄漏切分。
2. 数据已冻结：synthetic_v1_5k_frozen_v1。
3. Tokenizer 已冻结：1807-piece BPE。
4. 15,003,648 参数 Decoder-only Transformer。
5. 家中 RTX 2060 已完成 5000 Step 正式训练。
6. 最终验证 Loss 0.05746，Token Accuracy 0.9723。
7. 最佳验证 Loss 在 step_00004900.pt（0.05696）。
8. 30 条 AI 模拟人工挑战集已独立冻结，禁止进入训练。

本机约束：
- 公司电脑是 GTX 1650，约 4GB 显存。
- 不要重新训练 5000 Step。
- 评估和推理可用 CUDA；没有 CUDA 则用 CPU。
- 不要覆盖冻结数据、Tokenizer、挑战集或正式 Checkpoint。

开始前先检查这些路径是否存在：
- artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/sonogpt_bpe.model
- data/processed/synthetic_v1_5k_candidate_v2/train.jsonl
- artifacts/training/sonogpt_16m_m3/step_00004900.pt
- artifacts/training/sonogpt_16m_m3/step_00005000.pt

如果缺失，先停止并告诉我按 COMPANY_HANDOFF.md 拷贝哪些文件。不要重新生成数据。

下一步只做这一项：实现最终评估流水线。

要求：
1. 评估三组数据：
   - data/processed/synthetic_v1_5k_candidate_v2/test_seen_templates.jsonl
   - data/processed/synthetic_v1_5k_candidate_v2/test_heldout_templates.jsonl
   - data/challenges/simulated_human_challenge_v1.jsonl
2. 默认评估 artifacts/training/sonogpt_16m_m3/step_00004900.pt，并对照 step_00005000.pt。
3. 先校验冻结记录，再加载模型和 Tokenizer。
4. 生成时不要静默截断。超过 max_seq_len 应报错。
5. 指标至少包括：
   - 报告可解析/可对齐率
   - 目标文本精确匹配率
   - Token 级准确率或等价生成质量
   - 结构化字段准确率（部位、成分、回声、形状、边缘、强回声、血流、淋巴结）
   - 尺寸数值误差
   - 与四个训练模板的完全匹配率
6. 挑战集必须单独报告，不能和模板测试集混在一起平均后当作最终成绩。
7. 结果写入 reports/evaluation/，并更新 IMPLEMENTATION_PROGRESS.md。
8. 增加测试，最后 uv run pytest 必须通过。
9. 全程用简体中文回复。工作留痕，不要虚构临床审核或真实患者数据。

先给出你读到的当前状态摘要，然后开始实现评估流水线。
```
