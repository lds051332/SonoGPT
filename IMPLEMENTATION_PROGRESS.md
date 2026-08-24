# SonoGPT 实施进度

> 更新时间：2026-08-24  
> 当前阶段：正式 5000 Step 训练已完成 / 下一步为三集评估

## 本次已完成

### 1. 建立最小 Python 项目

- 使用 `src` 包布局；
- 项目包名统一为 `sonogpt`；
- 使用 Python 3.11+；
- 使用 Pydantic 2 执行运行时 Schema 校验；
- 使用 pytest 执行自动化测试；
- 使用 uv 创建环境并生成 `uv.lock`；
- 增加 `.gitignore`，排除虚拟环境、缓存、模型产物和生成数据。

关键文件：

```text
pyproject.toml
uv.lock
src/sonogpt/
tests/
```

### 2. 定义单结节 Schema v1

实现文件：

```text
src/sonogpt/schemas/domain.py
```

当前 Schema 固定：

- 器官为甲状腺；
- Schema 版本为 `1.0.0`；
- V1 必须且只能包含一个结节；
- 部位区分左右叶、峡部及上中下部；
- 尺寸统一保存为毫米；
- 尺寸必须包含 2～3 个有限正数；
- 成分、回声、形状、边界、强回声、血流和淋巴结均使用枚举；
- 区分 `not_mentioned`、`unknown`、`not_applicable`；
- 强回声与血流中的 `none` 表示原文明示未见，不等于未提及；
- 拒绝未定义的额外字段；
- 峡部必须使用 `segment="not_applicable"`。

当前 Schema 是工程数据契约，不包含诊断结论，也没有实现 TI-RADS。

### 3. 建立确定性模板报告基线

实现文件：

```text
src/sonogpt/baselines/template_report.py
```

基线可以把通过校验的单结节结构转换成稳定中文报告，支持：

- 部位；
- 2～3 维尺寸；
- 成分与回声；
- 形状；
- 边界；
- 强回声；
- 血流；
- 可疑淋巴结描述。

基线不使用 AI、不生成诊断结论。字段未提及或无法规范化时，模板不会自行补充内容。

### 4. 编写 50 条人工测试夹具

数据文件：

```text
data/fixtures/single_nodule_v1.jsonl
```

覆盖内容：

- 左叶、右叶、峡部；
- 上部、中部、下部及未明确分段；
- 2 维与 3 维尺寸；
- 整数与小数尺寸；
- 各类成分、回声、形状、边界、强回声及血流；
- 字段缺失、未知和不适用；
- 明确未见与未提及的差异；
- 淋巴结阴性、可疑及未提及。

这些夹具是工程测试数据，尚未经过超声专业人员审核，不能作为临床标准或医学 Gold Dataset。

### 5. 自动化验证

测试文件：

```text
tests/test_domain_schema.py
tests/test_template_report.py
```

已验证：

- 非法尺寸被拒绝；
- 多结节被 V1 Schema 拒绝；
- 非法峡部位置被拒绝；
- 未定义字段被拒绝；
- 未提及与明确未见保持不同语义；
- 50 条夹具全部通过 Schema；
- 模板输出与 50 条预期报告逐字一致；
- 模板报告不包含良恶性、癌症、确诊或治疗结论。

本次测试结果：

```text
29 passed in 1.15s
```

复现命令：

```powershell
uv sync --extra dev
uv run pytest
```

### 6. 实现可复现语义病例采样器

实现文件：

```text
src/sonogpt/data/constraints.py
src/sonogpt/data/semantic_generator.py
```

当前采样器：

- 只生成 Schema v1 的甲状腺单结节结构，不包含诊断结论；
- 直接构造并校验 `ThyroidExam`，没有另建字段体系；
- 使用局部伪随机数生成器，相同种子和数量会得到完全相同的病例；
- 按规范 JSON 的 SHA-256 生成完整、稳定的 `semantic_case_id`；
- 保证同一次采样中的语义病例 ID 唯一；
- 保留少量 `not_mentioned` 和 `unknown`，但不再为可观察的结节属性生成
  `not_applicable`；
- 尺寸顺序固定为横径、前后径、可选纵径，并与高于宽/非高于宽一致；
- 合成病例已报告的任一径线不小于 3 mm，最大/最小径线比不超过 3；
- 使用条件采样约束囊性、海绵状、无回声与血流、强回声、边缘和形状的组合；
- 约束只用于合成训练数据，不缩窄 Schema 对外部真实数据的接收范围。

生成器与合成约束版本均为 `1.1.0`，默认种子为 `20260824`。

### 7. 实现 4 种报告模板家族

实现文件：

```text
src/sonogpt/data/renderers.py
```

当前模板家族：

```text
location_first_v2
descriptor_first_v2
dimensions_first_v2
flow_first_v2
```

`location_first_v2` 与确定性模板报告完全一致。其余模板复用
`template_report.py` 生成的规范文本片段，只调整语序，不新增 Schema
中不存在的信息。所有样本均为 `task="generate"`，尚未生成 `extract`
反向任务。

渲染器版本为 `1.1.0`。其中 `smooth` 明确渲染为“边缘光滑”，
`vascularity=none` 明确渲染为“结节内及周边未见明显血流信号”，不再
把边缘光滑降级成边界清晰，也不再只描述内部无血流。

每条生成样本记录：

- `sample_id`；
- `semantic_case_id`；
- 规范 JSON 输入和报告目标；
- `template_family`；
- Schema、生成器和渲染器版本；
- 随机种子与合成数据来源。

### 8. 实现双层防泄漏切分

实现文件：

```text
src/sonogpt/data/split.py
```

当前默认切分：

- 先按 `semantic_case_id` 将病例分为 80% 训练、10% 验证、10% 测试；
- 训练、验证和测试的语义病例 ID 两两无交集；
- 训练、验证和普通测试只使用 3 个已见模板家族；
- `flow_first_v2` 只用于测试病例的模板留出视图，训练和验证均不可见；
- 普通测试与模板留出测试共享测试病例，目的是只改变表达方式进行配对比较；
- 所有测试病例都未进入训练，因此不存在测试语义泄漏。

### 9. 实现统计、数据写出与 SHA-256 manifest

实现文件：

```text
src/sonogpt/data/manifest.py
scripts/generate_data.py
```

统计内容包括：

- 样本数、语义病例数、任务和模板家族分布；
- 每个 Schema 枚举值的病例级计数；
- 缺失状态计数；
- 尺寸维数、范围和 mm 单位计数；
- 输入/目标字符长度分位数；
- 重复样本 ID、语义病例、输入和目标计数；
- 每个切分的样本数和语义病例数。

写出前会重新校验规范 JSON、Schema、合成约束、`semantic_case_id`、
`sample_id`、模板目标和版本。manifest 记录尺寸顺序、约束版本，以及每个
JSONL 和统计文件的字节数、记录数与 SHA-256，可通过 `verify_manifest`
重新计算验证。

已用以下命令生成 100 个语义病例的小型可复现快照：

```powershell
uv run python scripts/generate_data.py --count 100
```

结果为 310 条 `generate` 样本：

- 训练：240 条 / 80 个病例；
- 验证：30 条 / 10 个病例；
- 已见模板测试：30 条 / 10 个病例；
- 留出模板测试：10 条 / 同一组 10 个测试病例。

manifest：

```text
data/manifests/synthetic_v1_small.manifest.json
SHA-256: 9f9d1c6791a565483980561dd4c7a8ed54831e08142173f9cc3a34cff2cb095f
```

生成的 JSONL 和统计文件位于被 `.gitignore` 排除的 `data/processed/`，
不会提交大批派生数据；manifest 可以提交。

### 10. 增加数据管线测试与 Python 版本固定

新增测试覆盖：

- 同种子可复现、Schema 始终有效、ID 稳定且唯一；
- 4 个模板家族语序不同且保留所有明示字段；
- 生成任务记录稳定且不含诊断结论；
- 语义病例切分无交集；
- 留出模板不进入训练或验证；
- 统计按病例去重、样本 ID 无重复；
- manifest 可复算且能检测文件篡改。

新增 `.python-version`，固定本项目默认使用 Python 3.12。本次已实际执行：

```powershell
uv sync --extra dev
uv run pytest
```

结果为 `29 passed in 1.15s`。

### 11. 实现字符级与 SentencePiece BPE Tokenizer

实现文件：

```text
src/sonogpt/tokenizer/base.py
src/sonogpt/tokenizer/character.py
src/sonogpt/tokenizer/sentencepiece_bpe.py
src/sonogpt/tokenizer/validation.py
```

字符级基线：

- 固定 8 个控制 Token；
- 训练词表只读取训练集文本；
- 已见字符按字符编码；
- 未见 Unicode 使用 UTF-8 byte fallback，保持完全可逆；
- 词表保存为确定性 JSON，并计算内容及文件 SHA-256；
- 支持加载后词表和编码结果一致。

SentencePiece BPE：

- 使用 `model_type="bpe"`；
- 使用 identity normalization，不改写规范 JSON 或报告；
- 开启 byte fallback；
- 固定控制 Token；
- 禁止随机打乱语料并使用单线程训练，便于复现；
- 支持模型 SHA-256、往返编码和未知 Token 率验证。

最初已用 512 词表完成 BPE Smoke Test；后续候选词表实测见第 17 节。

### 12. 实现 Causal LM 数据编码与 Loss Mask

实现文件：

```text
src/sonogpt/data/dataset.py
```

生成任务编码格式：

```text
<bos><task_generate><input>{规范JSON}<target>{报告}<eos>
```

当前行为：

- Prompt、JSON 和 `<target>` 控制符对应标签均为 `-100`；
- 只对报告 Token 和 `<eos>` 计算损失；
- 批处理动态右侧 Padding；
- Padding 同时受 Attention Mask 和 Loss Mask 保护；
- 序列过长时明确抛出 `SequenceTooLongError`，禁止静默截断；
- 字符级 32 条过拟合集实测最大长度为 462，因此调试配置使用
  `max_seq_len=512`。

### 13. 实现极小 Decoder-only Transformer

实现文件：

```text
src/sonogpt/model/config.py
src/sonogpt/model/attention.py
src/sonogpt/model/block.py
src/sonogpt/model/gpt.py
```

已实现：

- Token Embedding 与可学习位置 Embedding；
- Pre-LayerNorm；
- 多头 Causal Self-Attention；
- Padding Mask；
- GELU 前馈网络；
- 输入输出 Embedding 权重共享；
- 仅目标区域的交叉熵；
- Greedy 生成和 EOS 停止；
- 参数量统计、配置保存加载及权重保存加载。

单元测试已验证未来 Token 不影响此前位置、Padding 不改变有效 Token
Logits、权重确实共享，模型保存加载后输出一致。

### 14. 完成 32 条样本 CPU 过拟合

实现文件：

```text
src/sonogpt/training/reproducibility.py
src/sonogpt/training/overfit.py
scripts/run_overfit.py
```

过拟合诊断使用 32 个不同 `semantic_case_id`，且每个病例只选择
`location_first_v2` 一个确定目标。这一点很重要：普通合成训练集允许同一
JSON 对应多个语义等价模板，但若在链路正确性测试中保留多个目标，就存在
相同 Prompt 对应不同下一 Token 的不可约损失，不能用于证明训练代码能接近
零损失。

复现命令：

```powershell
uv run python scripts/run_overfit.py
```

实际结果：

```text
设备：CPU
样本：32
参数量：156,096
最大序列：453 Token
训练步数：800
Batch Size：4
学习率：0.001
初始 Loss：5.947933
最终 Loss：0.021447
初始目标 Token 准确率：0.015441
最终目标 Token 准确率：0.995459
训练循环耗时：24.052 秒
```

模型、Tokenizer 和结果写入被 `.gitignore` 排除的
`artifacts/overfit/character_32/`，不会提交权重。

当前环境解析到 `torch 2.13.0+cpu`，`torch.version.cuda` 为 `None`，
因此本次只证明 CPU 训练链路正确，尚未验证 RTX 2060。进入 M3 前需先用
`nvidia-smi` 确认驱动，再安装匹配的 PyTorch CUDA 构建，不能把当前结果
写成 GPU 训练结果。

### 15. M2 自动化验证

新增测试覆盖：

- 字符级 Tokenizer 已见/未见 Unicode 往返编码；
- Tokenizer 保存加载与哈希；
- BPE byte fallback、数字、单位和 JSON 标点；
- Prompt/目标 Loss Mask；
- 超长序列拒绝和动态 Padding；
- 模型形状、Causal Mask、Padding Mask、权重共享；
- Greedy EOS、权重保存加载；
- 小集合训练损失确实下降。

重新执行：

```powershell
uv sync --extra dev
uv run pytest
```

结果为 `42 passed in 4.43s`。

### 16. 生成 5,000 病例修复版候选数据

生成命令：

```powershell
uv run python scripts/generate_data.py `
  --count 5000 `
  --output-directory data/processed/synthetic_v1_5k_candidate_v2 `
  --manifest-path data/manifests/synthetic_v1_5k_candidate_v2.manifest.json
```

结果：

- 5,000 个唯一语义病例；
- 训练：12,000 条 / 4,000 个病例；
- 验证：1,500 条 / 500 个病例；
- 已见模板测试：1,500 条 / 500 个病例；
- 留出模板测试：500 条 / 同一组 500 个测试病例；
- 合计 15,500 条 `generate` 样本；
- 样本 ID 重复数为 0，目标文本重复数为 0；
- 所有尺寸均为 mm，已报告径线范围为 3.0～40.0 mm；
- 所有 5,000 个病例均通过合成约束 v1.1。

manifest：

```text
data/manifests/synthetic_v1_5k_candidate_v2.manifest.json
SHA-256: fe7683dfff397c2c4f3cb6c6a27fda2386b7e6f03150b1e298913576c7f8bc29
```

已重新计算并验证 manifest 中所有文件的 SHA-256。JSONL 仍位于被忽略的
`data/processed/`。该数据尚未经过甲状腺超声专业人员审核，因此只能称为
“候选合成数据”，不能称为临床数据或最终冻结数据。

### 17. 训练可复现的 1,807-piece BPE 候选模型

新增脚本：

```text
scripts/train_tokenizer.py
```

训练器只读取 `train.jsonl` 的 12,000 条样本、24,000 段输入/目标文本。
验证集和测试集仅在训练结束后用于往返编码、未知 Token 和序列长度统计，
没有进入 SentencePiece 训练语料。

修复版受控语料最多形成 1,807 个有效 Piece；继续合并时已经没有有效符号。
强行保留 1,920 或补齐 4,096 会产生未使用 Embedding。因此候选词表根据
最终语料实测调整为 1,807。未来模板与自然语言多样性显著增加后再重新评估。

复现命令：

```powershell
uv run python scripts/train_tokenizer.py
```

验证结果：

- 请求词表：1,807；
- 实际词表：1,807；
- 全部切分往返编码成功；
- 全部切分 `<unk>` 率为 0；
- 训练完整序列最大 148 Token；
- 验证最大 143 Token；
- 已见模板测试最大 144 Token；
- 留出模板测试最大 148 Token；
- 所有 15,500 条序列均小于 `max_seq_len=384`。

为了让模型文件哈希可跨临时路径复现，保存前会从 SentencePiece protobuf
中移除训练语料和临时模型的绝对路径，并使用确定性序列化。连续两次训练
得到相同模型 SHA-256：

```text
743af34aa2d8bec332f82ce6756cd1458e9687244def23ece5c5b60ee8883ac2
```

候选 Tokenizer 及其 manifest 位于被忽略的：

```text
artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/
```

### 18. 落地约 15.00M 参数候选正式配置

配置与检查脚本：

```text
configs/model/sonogpt_16m_candidate.json
scripts/inspect_model.py
```

当前配置：

```text
vocab_size: 1807
max_seq_len: 384
n_layers: 8
n_heads: 6
d_model: 384
d_ff: 1536
dropout: 0.1
tie_embeddings: true
bias: false
```

实测结果：

```text
实际参数量：15,003,648
FP32 参数内存：57.234 MiB
FP16 参数内存：28.617 MiB
前向 Smoke Test：[2, 150] → [2, 150, 1807]
Logits：全部有限
```

`15.004M` 符合项目约 16M、10M～20M 的目标区间。参数量来自代码实际
统计，不沿用设计稿中的估算。检查脚本同时验证候选 Tokenizer 与模型
`vocab_size` 一致，并记录 Tokenizer SHA-256。

### 19. 候选数据与模型配置自动化验证

新增验证包括：

- 同一语料在不同临时路径训练得到相同 BPE 模型哈希；
- 候选模型配置可加载；
- 候选模型实际参数量固定为 15,003,648；
- 1,807 词表与 384 上下文均来自实际数据统计；
- 生成器输出全部通过合成约束，审核发现的反例由单元测试固定。

重新执行：

```powershell
uv sync --extra dev
uv run pytest
```

结果为 `56 passed in 5.22s`。

### 20. 根据 50 条审核报告修复合成数据

第一次抽查修复前的留出模板样本，结果为 9 条通过、27 条需复核、14 条建议
剔除。主要问题是成分与回声矛盾、尺寸比例极端且形状不一致、可观察属性误用
`not_applicable`，以及边缘和无血流措辞发生语义降级。

本轮修复：

- 新增版本化 `constraints.py`，把审核结论固化为可测试的合成数据质量规则；
- 囊性、海绵状和无回声模式使用条件分布，不再与内部/混合血流及不相容强
  回声随机组合；
- 实性和囊实性不再生成无回声；
- 形状先决定横径与前后径关系，限制极端比例并增加 3 mm 合成下限；
- 可观察结节属性不再采样 `not_applicable`；
- 修正 `smooth` 和 `vascularity=none` 的中文渲染，并将四个模板家族升级到
  v2；
- manifest v1.1 记录约束版本和尺寸顺序，写出时执行全量约束校验；
- 保留 Schema v1 的宽接收能力，以上限制不用于拒绝外部真实病例。

修复过程中又发现并补上“成分未提及的无回声结节 + 内部血流”遗漏规则。
最终对 `test_heldout_templates.jsonl` 再抽查 50 条，结果为：

```text
pass: 50
review: 0
reject: 0
```

该结果说明审核发现的工程性系统问题在本次样本中未再出现，但仍不能替代执业
超声专业人员的正式审核或临床签字。修复前数据保留作问题追溯，后续训练只使用
`synthetic_v1_5k_candidate_v2`。

### 21. 实现 M3 正式训练器与精确断点恢复

实现文件：

```text
src/sonogpt/training/config.py
src/sonogpt/training/scheduler.py
src/sonogpt/training/checkpoint.py
src/sonogpt/training/trainer.py
scripts/train_model.py
configs/training/sonogpt_16m_m3.json
tests/test_training.py
```

正式训练器包括：

- AdamW，并对矩阵权重与 Bias/LayerNorm 参数使用独立 Weight Decay 组；
- 线性 Warmup + Cosine Decay，调度器步数可序列化；
- CUDA FP16/BF16 Autocast，其中 FP16 使用 GradScaler；
- CPU 自动关闭 AMP，不把 CPU Smoke Test 写成 GPU 验证；
- 按目标 Token 数加权的梯度累积，使动态长度微批次的 Loss 权重正确；
- 梯度裁剪、非有限梯度检测及 AMP 跳步记录；
- 验证集目标 Token Loss 与 Teacher-forced Token Accuracy；
- 可序列化的确定性 Shuffle 流，恢复时继续同一 Epoch 和样本位置。

Checkpoint 版本为 `1.0.0`，采用临时文件 + 原子替换写入，包含：

- 模型、Optimizer、Scheduler 和 GradScaler 完整状态；
- Global Step、样本数、目标 Token 数、最佳验证 Loss；
- 数据批次 Epoch 与 Cursor；
- Python、NumPy、PyTorch CPU 和 CUDA RNG 状态；
- 模型/训练配置、数据 manifest、训练/验证文件与 Tokenizer SHA-256；
- 设备类型和实际 AMP 状态。

恢复时会拒绝数据、Tokenizer、配置或运行设备不一致的 Checkpoint。
`latest.json` 同时记录最新 Checkpoint 文件名和 SHA-256，加载前会复算验证。
单元测试已证明：连续训练 4 步，与“训练 2 步 → 保存 → 新 Trainer 恢复 →
继续 2 步”的模型参数、Optimizer、Scheduler、批次位置和计数完全一致。

候选训练配置：

```text
Micro Batch Size: 4
Gradient Accumulation: 8
Effective Batch Size: 32
Max Steps: 5,000
Learning Rate: 3e-4
Min Learning Rate: 3e-5
Warmup Steps: 200
Weight Decay: 0.1
Gradient Clip Norm: 1.0
AMP: float16（仅 CUDA 启用）
```

端到端 CPU Smoke Test：

```powershell
uv run python scripts/train_model.py --smoke-test
uv run python scripts/train_model.py --smoke-test --resume latest
```

实测使用最终 15,003,648 参数配置、修复版数据和 1,807-piece BPE：

```text
训练步数：2
训练样本视图：32
验证样本视图：16
初始训练 Loss：7.709571
第 2 步训练 Loss：6.539666
第 2 步验证 Loss：6.040632
Checkpoint 恢复后验证 Loss：6.040632
跳过 Optimizer Step：0
```

每次运行在 `artifacts/training/` 下留下 `run_manifest.json`、
`metrics.jsonl`、`summary.json`、`latest.json` 和分步 Checkpoint。该目录
被 `.gitignore` 排除，不会提交模型权重。

新增训练测试后，当时全量结果为 `62 passed in 5.64s`。

### 22. 完成 RTX 2060 Laptop CUDA 与 200 Step 基准

环境检测：

```text
GPU: NVIDIA GeForce RTX 2060 Laptop
VRAM: 6,144 MiB
Compute Capability: 7.5
Driver: 560.81
驱动支持的 CUDA 上限: 12.6
Python: 3.12.13
```

原环境为 `torch 2.13.0+cpu`，无法使用显卡。根据驱动能力和官方 PyTorch
索引，将项目锁定为：

```text
torch 2.12.0+cu126
CUDA Runtime 12.6
cuDNN 9.10.2
```

`pyproject.toml` 已增加显式 `pytorch-cu126` 官方索引和平台 Marker，
`uv.lock` 也已更新。以后执行普通 `uv sync --extra dev` 会继续安装锁定的
CUDA 构建，不需要另装系统 CUDA Toolkit。

新增 GPU 留痕与验证文件：

```text
src/sonogpt/training/telemetry.py
scripts/compare_checkpoints.py
reports/benchmarks/rtx2060_200step_20260824.json
```

CUDA 验证已通过：

- PyTorch 能识别 RTX 2060 和 6 GB 显存；
- 2,048×2,048 FP16 CUDA 矩阵乘法结果全部有限；
- 最终 15,003,648 参数模型完成 CUDA AMP 前向、反向、验证和 Checkpoint；
- GPU 训练时 GradScaler 没有跳过 Optimizer Step；
- 训练 CLI 用 `nvidia-smi` 每秒采样温度、利用率、显存和功耗；后续已在
  第 23 节升级为可暂停恢复的温控。

200 Step 基准使用 Micro Batch 4、梯度累积 8、有效 Batch 32：

```text
Optimizer Step: 200
总耗时（含两次验证和 Checkpoint）: 52.916 秒
纯训练 Step 耗时: 47.388 秒
训练吞吐: 135.05 样本/秒
目标 Token 吞吐: 3,354.85 Token/秒
PyTorch 峰值分配显存: 366.73 MiB
PyTorch 峰值保留显存: 386 MiB
nvidia-smi 系统级峰值显存: 1,378 MiB
最高温度: 73°C
平均温度: 66.94°C
最高功耗: 73.28 W
平均 GPU 利用率: 37.16%
AMP 跳步: 0
```

验证指标仅用于确认训练链路和趋势，不代表最终模型质量：

```text
Step 100: Loss 2.012309 / Token Accuracy 0.495294
Step 200: Loss 1.091569 / Token Accuracy 0.662879
```

实际 GPU 断点恢复也已验证：从原训练的 Step 100 Checkpoint 在新进程继续到
Step 200，训练 Loss、验证指标、模型、Optimizer、Scheduler、Scaler、批次
位置和全部 RNG 状态均与不中断训练完全一致。两个 `torch.save` 容器因文件
元数据具有不同文件哈希，但逐值语义哈希相同：

```text
semantic SHA-256:
e7a846a72eca03c5bd0d1acf28ed06606e896cc850a8d141ea62bdb167f34b10
```

CUDA 环境下重新执行 `uv run pytest`，最终结果为 `63 passed in 4.88s`。
完整机器可读报告已写入
`reports/benchmarks/rtx2060_200step_20260824.json`；模型权重仍只保存在被
忽略的 `artifacts/training/`。

### 23. 完成长任务进度心跳、温控恢复与 Batch A/B

训练 CLI 已新增独立于主训练循环的后台心跳。即使主线程正在初始化模型、
验证或序列化 Checkpoint，终端仍会定时输出：

```text
phase: 当前阶段
completed_steps / total_steps / percent: 当前进度
steps_per_second / eta_seconds: 速度和预计剩余时间
seconds_since_progress: 距上次实际推进的时间
status: running / warning_possible_stall / completed / failed
```

默认每 5 秒输出一次心跳；60 秒无实际推进时显示
`warning_possible_stall`，但不会误杀仍在工作的进程。数据加载与编码、
模型初始化、训练、验证、Checkpoint、温控冷却和最终汇总均有独立阶段。
可用以下参数调整：

```text
--progress-interval-seconds
--stall-warning-seconds
```

RTX 2060 Laptop 上完成了保持有效 Batch 32 的 100 Step A/B：

```text
Micro 4  / Accum 8: 2,587.82 Token/s，366.53 MiB，最高 72°C
Micro 8  / Accum 4: 4,831.70 Token/s，450.79 MiB，最高 74°C
Micro 16 / Accum 2: 9,148.81 Token/s，622.57 MiB，最高 78°C
```

短测表明 Micro 16 吞吐最高，且显存余量充足。但持续热测试发现不能只看
短测温度：

```text
Micro 16，500 Step: 9,321.35 Token/s，最高 84°C，最高 107.56 W
Micro 8: 连续运行至 Step 460 时达到旧版 85°C 安全线并主动停止
```

因此训练 CLI 从“达到温度直接中止”升级为温控暂停。默认 82°C 暂停，
降至 75°C 自动继续，最长等待 600 秒；期间心跳阶段为
`thermal_cooldown`。仍可通过 `--thermal-action abort` 改为立即中止。
强制低阈值实测成功触发两次冷却、累计等待 5.00 秒并完成训练。

候选正式配置已从 Micro 4 / Accum 8 调整为 Micro 16 / Accum 2，有效
Batch 仍为 32，并由默认温控提供笔记本安全余量。也可临时覆盖：

```text
--micro-batch-size
--gradient-accumulation-steps
--max-gpu-temperature
--thermal-resume-temperature
--max-cooldown-seconds
```

新增留痕：

```text
src/sonogpt/training/progress.py
scripts/benchmark_batch_sizes.py
reports/benchmarks/rtx2060_batch_ab_20260824.json
```

完整测试结果为 `65 passed in 5.31s`，相关文件无 IDE Lint 错误。

### 24. 完成专业审核留痕与数据冻结

项目所有者确认外部专业审核信息：

```text
审核结论：通过，未发现需要修改的问题
抽查数量：50 条
审核角色：执业超声医师
审核日期：2026-08-24
未解决问题：0
```

审核身份未在仓库中记录，冻结文件将证据来源明确标记为
`project_owner_reported_external_review`，不虚构审核者姓名或机构。

已创建正式冻结版本：

```text
Freeze ID: synthetic_v1_5k_frozen_v1
Freeze Record: data/releases/synthetic_v1_5k_frozen_v1.freeze.json
Freeze Payload SHA-256:
113522f93475a3f09502da0254a7b78107c8a265b6b998971b9e5824efd1a3c7
Data Manifest SHA-256:
fe7683dfff397c2c4f3cb6c6a27fda2386b7e6f03150b1e298913576c7f8bc29
Tokenizer Model SHA-256:
743af34aa2d8bec332f82ce6756cd1458e9687244def23ece5c5b60ee8883ac2
```

冻结记录绑定以下内容：

- 12,000 条训练样本、1,500 条验证样本；
- 1,500 条已见模板测试样本和 500 条留出模板测试样本；
- 数据统计、Manifest、生成器、约束、渲染器和 Schema 版本；
- 1,807-piece BPE 模型及其训练语料来源；
- 生成、渲染、切分与 Tokenizer 相关代码的来源哈希；
- 专业审核结论和审核范围。

冻结采用“原文件 + 不可变哈希记录”，不复制约 15 MB 的相同数据。原目录名
虽然仍含 `candidate_v2`，但正式发布身份由 Freeze ID 和全部 SHA-256
确定。任何数据、切分、Manifest、Tokenizer 或冻结记录修改都会导致校验失败。

新增：

```text
src/sonogpt/data/freeze.py
scripts/freeze_data.py
tests/test_freeze.py
data/releases/synthetic_v1_5k_frozen_v1.freeze.json
```

训练入口现在默认执行 `freeze_verification` 阶段，并要求命令行中的数据目录、
Manifest 和 Tokenizer 与冻结记录一致。Freeze ID 和冻结哈希也会写入每次
训练的 `run_manifest.json` 和 Checkpoint `run_identity`。冻结版本 CPU
Smoke Test 已通过；完整测试为 `68 passed in 5.42s`，无 IDE Lint 错误。

训练数据冻结记录明确写为 `challenge_set_included=false`。挑战集在下一节
作为独立评估发布物冻结，不反向修改已经冻结的训练发布记录。

### 25. 完成 AI 模拟人工挑战集并独立冻结

根据项目仅用于学习 Demo 的范围，创建了 30 条固定的“模拟人工挑战”样本。
这些样本由 AI 手工策划病例和参考措辞，不调用现有随机语义采样器，也不调用
四个报告模板。来源被明确标记为：

```text
authoring_method: ai_simulated_manual_independent_v1
source: simulated_nonclinical
human_authored: false
clinical_data: false
intended_use: learning_demo_evaluation_only
```

覆盖重点包括：

- 新语序和同义改写；
- 二维、三维和小数尺寸；
- 未提及、未知和检查条件受限；
- 峡部、囊性、海绵样、囊实性与实性结节；
- 高于宽、甲状腺外侵犯、点状强回声和周边钙化；
- 无血流、周边血流、内部血流和混合血流；
- 可疑与无可疑淋巴结表述；
- 小结节和较大结节边界案例。

自动完整性检查结果：

```text
样本数：30
唯一语义病例：30
与冻结训练/验证/测试语义重叠：0
与冻结报告文本重叠：0
与四个训练模板完全相同：0
最大 BPE 序列长度：216
超过模型 384 Token 上限：0
```

挑战集没有并入训练集，也没有用于训练 Tokenizer。冻结记录设置
`training_use_prohibited=true` 和
`tokenizer_training_use_prohibited=true`。

独立冻结身份：

```text
Freeze ID: simulated_human_challenge_v1
Challenge File:
data/challenges/simulated_human_challenge_v1.jsonl
Challenge SHA-256:
43acba266a8a2b8b902a646cdbd3db461becfa7df089292d8d23699eea1b9653
Freeze Record:
data/releases/simulated_human_challenge_v1.freeze.json
Freeze Payload SHA-256:
be9d4b2f0d71170f649ada7177534c04d4f408e38ed40d12e12327561afd0acd
```

挑战集冻结记录链接到 `synthetic_v1_5k_frozen_v1`，同时绑定挑战文件、
构建脚本和训练冻结记录的 SHA-256。任意一项被修改都会校验失败。

新增：

```text
src/sonogpt/evaluation/challenge.py
src/sonogpt/evaluation/__init__.py
scripts/build_simulated_challenge.py
tests/test_challenge.py
data/challenges/simulated_human_challenge_v1.jsonl
data/releases/simulated_human_challenge_v1.freeze.json
```

挑战集冻结复核通过；完整测试为 `71 passed in 6.10s`，无 IDE Lint 错误。
这满足学习 Demo 的独立评估需求，但不能称为真实人类或临床挑战集。

### 26. 完成 RTX 2060 正式 5000 Step 训练

家中 RTX 2060 Laptop 已跑完冻结数据上的正式训练。命令为：

```text
uv run python scripts/train_model.py --device cuda
```

输出目录：`artifacts/training/sonogpt_16m_m3/`。该目录被 `.gitignore` 排除，不会进入 GitHub。

结果：

```text
状态：completed
步数：5000 / 5000
参数量：15,003,648
冻结版本：synthetic_v1_5k_frozen_v1
Micro Batch：16
梯度累积：2
有效 Batch：32
跳过 Optimizer Step：0
总耗时：826.407 秒
纯训练 Step 耗时：385.388 秒
吞吐：415.17 样本/秒，10,288.60 目标 Token/秒
PyTorch 峰值分配显存：632.36 MiB
nvidia-smi 峰值显存：1,605 MiB
最高温度：83°C
温控暂停：96 次，累计 301.14 秒
最终验证 Loss：0.057462
最终 Token Accuracy：0.972282
最佳验证 Loss：0.056965（Step 4900）
```

建议评估时同时比较：

```text
artifacts/training/sonogpt_16m_m3/step_00004900.pt
artifacts/training/sonogpt_16m_m3/step_00005000.pt
```

机器可读摘要：`reports/training/sonogpt_16m_m3_5000step_20260824.json`。

换电脑继续开发时，GitHub 只包含源码和冻结记录。Tokenizer、生成数据和 Checkpoint 需要按
`COMPANY_HANDOFF.md` 手工拷贝。公司 GTX 1650 足够做评估和推理，不要重跑正式训练。

### 27. 补齐交接文档

新增：

```text
README.md
COMPANY_HANDOFF.md
CURSOR_PROMPT.md
reports/training/sonogpt_16m_m3_5000step_20260824.json
```

## 当前尚未完成

- 已见模板、留出模板和模拟挑战集上的最终评估；
- 规则抽取与质控基线；
- 推理 CLI / 服务；
- 50,000～100,000 病例完整数据是否有收益仍需学习曲线证明。

## 下一步建议

下一次优先实现：

1. 在已见模板、留出模板和模拟人工挑战集上完成最终评估；
2. 比较 Step 4900 与 Step 5000 Checkpoint；
3. 增加规则抽取与质控基线，用于与模型结果对照；
4. 增加推理入口；
5. 如项目未来超出演示范围，再由真人独立编写临床挑战集。

## 需要人工确认的领域问题

本轮 50 条外部专业抽查已经通过。进入更大规模数据生成或扩展任务范围前，
仍需重新确认：

- 字段枚举是否完整；
- 中文术语是否符合目标报告规范；
- “未知”“未提及”“不适用”的边界；
- 成分、回声和强回声组合是否存在不合理表达；
- 模板报告的语序与医院实际习惯；
- 后续是否采用某一明确版本的 TI-RADS。

本次审核支持冻结当前工程候选版本，但不等同于监管批准，也不应把当前模板
称为通用临床标准。
