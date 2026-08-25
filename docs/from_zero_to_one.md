# 从 0 到 1：用 SonoGPT 学完一次大模型开发

这份文档按**真实开发顺序**讲解：从把问题写成可训练任务，到数据、Tokenizer、Transformer、训练、评估、推理。SonoGPT 是一个可以在笔记本上从零训完的完整闭环，用来学流程，而不是学「怎样做出 ChatGPT」。

**定位提醒：** 这是学习与工程演示。训练数据是合成报告，不是真实病历。模型不能用于临床诊断、筛查或治疗决策。

配套文档：

| 想查什么 | 去哪 |
| --- | --- |
| 当前实现进度与实测数字 | 仓库根目录 [`IMPLEMENTATION_PROGRESS.md`](../IMPLEMENTATION_PROGRESS.md) |
| 当初为什么这样设计 | [`SONOGPT_ARCHITECTURE_DESIGN.md`](../SONOGPT_ARCHITECTURE_DESIGN.md) |
| 评估数字怎么读 | [`evaluation.md`](evaluation.md) |
| 规则 / 模板 / 质控 | [`baselines.md`](baselines.md) |
| CLI / 网页 | [`inference.md`](inference.md)、[`web.md`](web.md) |

---

## 0. 怎么读

建议分三遍：

1. **只看地图（第 1 节）**，建立「工业界 LLM 流水线 ↔ 本仓库模块」的对应。
2. **按阶段跟代码（第 2～11 节）**，每一节都有：要解决的问题、本仓库怎么做、该读的文件、可执行的命令。
3. **对照结果与边界（第 12～14 节）**，学会读指标，并分清「这个小模型证明了什么」和「它没有证明什么」。

不需要一次跑完正式 5000 步训练。CPU 上跑测试、造小数据、过拟合 32 条，就已经覆盖开发里最容易写错的部分。GPU 只负责把已经验证正确的训练任务跑完。

---

## 1. 先建立正确的地图

### 1.1 你实际在做一个什么系统

医院报告系统通常要两件事：把结构化所见写成规范句子，以及把句子还原成可校验的结构。SonoGPT 把它们拆开，并且**只训练其中一个方向**：

| 方向 | 输入 | 输出 | 谁来做 |
| --- | --- | --- | --- |
| Generate | Schema v1 JSON | 中文所见草稿 | 从零训练的 15M Decoder-only Transformer |
| Extract | 中文报告 | Schema v1 JSON | 规则基线（V1 **没有**训练抽取模型） |
| QC | 报告 ± JSON | 一致性发现 | 版本化规则 |

这是本项目最重要的产品决策，也是学大模型时最容易忽略的工程决策：

- **模型处理模糊语言**（同一结构可以有多种合法语序）。
- **规则处理确定约束**（单位、缺失 vs 明示未见、报告与结构是否矛盾）。
- **不要用模型自己抽自己的生成结果当唯一成绩**。

当前范围固定为：中文、甲状腺、单个结节、所见描述。没有超声图像、多结节关系、良恶性判断、TI-RADS。

### 1.2 工业界「训一个语言模型」通常包含什么

把公开资料里的大模型研发流水线压成一条链，本仓库几乎逐步都有对应物，只是规模小几个数量级：

```text
问题定义 / 数据契约
        │
        ▼
采集或生成数据  →  清洗、约束、切分、冻结
        │
        ▼
Tokenizer（词表只看训练集）
        │
        ▼
模型结构（这里是从零写的 GPT 风格 Decoder）
        │
        ▼
把样本编成 next-token 预测  →  先过拟合证明链路
        │
        ▼
正式训练（优化器、调度、AMP、梯度累积、断点）
        │
        ▼
用冻结测试集评估（独立解析器 + 基线对照）
        │
        ▼
推理服务（校验、解码、质控、回退）
```

对应到仓库：

```text
schemas/domain.py          数据契约
data/semantic_generator.py 先采样「语义病例」
data/renderers.py          再渲染成多种中文表达
data/split.py              按病例切分，并留出未见语序
data/freeze.py             哈希冻结，防止事后改测试集
tokenizer/                 字符级基线 + SentencePiece BPE
model/                     8 层 Decoder-only
data/dataset.py            Causal LM 编码与 Loss Mask
training/overfit.py        32 条过拟合
training/trainer.py        正式训练循环
evaluation/                三套集合分开报告
inference/ + web/          CLI 与本机 Demo
baselines/                 模板 / 抽取 / 质控对照
```

### 1.3 为什么「小」反而更适合学

从零训练 7B/70B 会把学习重点变成「怎么调用现成框架、怎么凑显存」。本仓库刻意选择：

- 参数量 **15,003,648**（代码统计，不是估算），RTX 2060 6GB 笔记本可训完。
- 词表 **1,807**，来自受控合成语料的实际 BPE 饱和点，而不是先写死 4096 再补空洞 Embedding。
- 任务单一：`JSON → 报告` 的条件语言建模。
- 全部核心模块自己实现，不依赖 Hugging Face `Trainer` / `transformers.GPT2LMHeadModel`。

学完之后，你应能回答：

1. 一个 Token 是怎么从中文变成整数、再变成向量的。
2. Causal Mask 为什么能让模型做自回归。
3. 为什么只对 `<target>` 之后计损失。
4. 验证 Loss 最低的 Checkpoint 不一定是「逐字最像金标准」的 Checkpoint。
5. 高准确率可能只是学会了模板；要用留出语序和独立解析器拆开「记句式」和「映射字段」。

### 1.4 和 ChatGPT 类系统差在哪（避免学完产生错觉）

| | SonoGPT V1 | 典型对话大模型 |
| --- | --- | --- |
| 数据 | 5,000 个合成语义病例，约 15,500 条 generate 样本 | 万亿 Token 级网页/书籍/代码，再加指令与偏好数据 |
| 目标 | 受控 Schema 到受控句式 | 开放世界问答、工具调用、多轮对话 |
| 对齐 | 无 RLHF / DPO | 监督微调 + 偏好优化 + 安全策略 |
| 解码 | 默认 Greedy | 温度、top-p，常加约束解码 |
| 评价 | 字段准确率、尺寸匹配、模板泄漏 | 混合基准、人工评测、在线指标 |
| 部署 | 本机 PyTorch 权重 | 推理引擎、量化、分布式、限流 |

本仓库证明的是：**在冻结 Schema 和合成分布上，小 Transformer 能学会结构到内容的映射。** 它不证明医学理解、开放中文能力，或「再堆数据就能当临床系统」。

---

## 2. 阶段 0：环境与工程骨架

### 要先解决什么

还没写 Attention 之前，就要能：

- 同一套依赖在 Windows / Linux 复现；
- 测试能在无 GPU 时跑通；
- 大文件（权重、JSONL）不进 Git。

### 本仓库怎么做

- Python **3.12**（`.python-version`），包布局是 `src/sonogpt/`，避免脚本从仓库根目录误导入。
- 依赖用 [uv](https://docs.astral.sh/uv/) 锁定：`pyproject.toml` + `uv.lock`。
- Windows / Linux 安装 `torch 2.12.0+cu126`；没有 NVIDIA 驱动时，规则抽取、质控和测试仍可在 CPU 上跑。
- `.gitignore` 排除 `artifacts/`、`data/processed/`、虚拟环境和缓存。Git 里保留源码、Schema、manifest、冻结记录和评估报告。

### 跟练

```powershell
git clone https://github.com/lds051332/SonoGPT.git
cd SonoGPT
uv sync --extra dev
uv run pytest
```

测试通过只说明**代码契约**成立，不说明你已经有 Tokenizer 和 Checkpoint。缺权重时：模板渲染、规则抽取、大部分单测仍然可用；模型生成需要按 [`COMPANY_HANDOFF.md`](../COMPANY_HANDOFF.md) 放置本地产物。

### 该读的文件

- `pyproject.toml`
- `tests/`（先看文件名，建立「每个模块都有对应测试」的习惯）

---

## 3. 阶段 1：把问题定义成可训练任务

大模型项目失败，经常不是 Attention 写错，而是**输入输出没有唯一含义**。

### 3.1 先冻结 Schema，再写任何生成器

所有模块必须共用同一份领域契约：数据生成、训练标签、推理入参、评估金标准。禁止各写一套字段名。

本仓库的契约是 `ThyroidExam`：器官固定甲状腺、V1 必须且只能一个结节、尺寸统一为毫米、拒绝未定义的额外字段。

关键空值语义（学 NLP 标注时会反复遇到）：

| 值 | 含义 | 不能当成 |
| --- | --- | --- |
| `not_mentioned` | 原文没说 | 「没有钙化」 |
| `none` | 原文明示未见（强回声 / 血流） | 「没提到」 |
| `unknown` | 原文有信息但无法归一 | 缺失 |
| `not_applicable` | 当前情况不适用（如峡部没有上中下） | 随便填一个部位 |

如果「没提到钙化」和「明确未见钙化」共用一个空值，质控和抽取都无法判断对错。

### 3.2 任务格式：用控制符把 seq2seq 塞进 GPT

设计稿里双向任务都预留了 Token，V1 只训练 generate：

```text
<bos><task_generate><input>{规范 JSON}<target>{中文报告}<eos>
```

规范 JSON 使用固定键顺序、无多余空格（`canonical_exam_json`），降低模型把精力浪费在空格和字段顺序上。

`extract` 的对称格式在 Tokenizer 里已经留了 `<task_extract>`，但 **V1 没有生成 extract 训练样本，也没有训练这个方向**。这是刻意的范围控制：先证明 generate 全链路，再考虑多任务。

### 3.3 成功标准要能拆开写

「准确率 96%」没有分母时没有意义。本项目从设计阶段就要求分开报告：

- JSON / Schema 是否合法；
- 明示字段是否抽对；
- 尺寸数值与单位；
- 遗漏 vs 无依据新增（幻觉）；
- 已见语序 / 留出语序 / 挑战集，**禁止三集平均**。

### 该读的文件

- `src/sonogpt/schemas/domain.py`
- `tests/test_domain_schema.py`
- 夹具：`data/fixtures/single_nodule_v1.jsonl`（50 条人工构造的边界样本）

---

## 4. 阶段 2：数据 —— 先有语义，再有句子

这是合成数据项目最容易走错的一步。错误做法是：写几十个字符串模板，拼接出「十万条」，再从字符串反推标签。结果是模型背模板，测试集只是换了一点措辞的同一批句子。

### 4.1 四层数据，不要揉成一层

```text
1. 本体 / Schema     字段、枚举、单位、空值
2. 语义病例          先采样结构，赋予稳定 semantic_case_id
3. 文本实现          同一病例用不同语序家族渲染
4. 扰动 / 缺陷       错别字、矛盾、注入错误；用于质控，不与干净训练样本混用
```

`semantic_case_id` 是规范 JSON 的 SHA-256，而不是自增整数。同一结构在不同运行中 ID 相同，切分才能稳定复现。

### 4.2 字段不能独立乱采样

完全独立随机会得到「囊性结节却写内部丰富血流」这类工程上不合理的组合。本仓库把审核结论写成可测试的 `constraints.py`：

- 尺寸顺序固定为横径、前后径、可选纵径，并与高于宽 / 非高于宽一致；
- 合成病例已报告径线不小于 3 mm，最大/最小径比不超过 3；
- 囊性 / 海绵状 / 无回声与不相容的血流、强回声做条件采样。

这些约束**只约束合成训练数据**，不拿来拒绝外部真实病例。Schema 仍然接受更宽的输入。这是「训练分布」和「接口契约」分离。

### 4.3 防泄漏切分（比模型结构更重要）

禁止：把所有文本行 `shuffle` 后按 8:1:1 切开。同一语义病例的不同改写一旦分到训练和测试，测试成绩会虚高。

本仓库默认：

1. 先按 `semantic_case_id` 分成 80% 训练 / 10% 验证 / 10% 测试，三集病例 ID 无交集。
2. 训练、验证、普通测试只用 3 个已见模板家族：
   - `location_first_v2`
   - `descriptor_first_v2`
   - `dimensions_first_v2`
3. `flow_first_v2` **只出现在测试病例的留出视图**。训练和验证从未见过这种语序。
4. 已见模板测试和留出模板测试**共享同一批测试病例**，只改表达方式，以便配对比较。

因此正式 5,000 病例会得到：

| 切分 | 样本 | 病例 | 模板 |
| --- | --- | --- | --- |
| 训练 | 12,000 | 4,000 | 3 个已见家族 |
| 验证 | 1,500 | 500 | 同上 |
| 已见模板测试 | 1,500 | 500 | 同上 |
| 留出模板测试 | 500 | 同一 500 | 仅 `flow_first_v2` |

另有 30 条 **AI 模拟挑战集**：不走随机采样器、不走四个模板，禁止进入训练和 Tokenizer。它用来看「换一种说法」时字段还在不在，**不是**真实人类或临床挑战集。

### 4.4 冻结：测试集不是调参旋钮

数据、Tokenizer、代码哈希写入 `data/releases/*.freeze.json`。训练入口默认先校验冻结记录。改动任一 JSONL、manifest 或 Tokenizer，校验失败，实验身份作废。

这对应工业界的：**eval 集锁定、artifact 溯源、禁止用测试集挑模型**。本项目用验证 Loss 选 Checkpoint，测试集只在最终评估跑一次（实现上评估脚本可重复跑，但数字应能复现，而不是每次改切分）。

### 跟练（小数据，不必 5k）

```powershell
uv run python scripts/generate_data.py --count 100
```

默认写出 `data/processed/synthetic_v1_small/`（被 git 忽略）和可提交的 manifest。

不要为了「练手」覆盖已冻结的 `synthetic_v1_5k_candidate_v2`。学习曲线或新数据应使用新目录和新 Freeze ID。

### 该读的文件

- `src/sonogpt/data/semantic_generator.py`
- `src/sonogpt/data/constraints.py`
- `src/sonogpt/data/renderers.py`
- `src/sonogpt/data/split.py`
- `src/sonogpt/data/manifest.py`
- `src/sonogpt/data/freeze.py`
- `scripts/generate_data.py`
- `tests/test_data_split.py`、`tests/test_semantic_generator.py`

---

## 5. 阶段 3：先做规则基线，再写神经网络

如果确定性模板已经能把规范 JSON 写成零幻觉报告，你必须先承认这一点，再解释「为什么还要模型」。

本项目的回答是：

- **模板**：给定 JSON，输出稳定、可审计、字段级零幻觉。产品上它可以当默认或回退。
- **模型**：同一 JSON 对应多种合法语序；模型要学的是结构到内容，而不是背某一个句式。评估时「逐字匹配低、字段准确率高」是预期现象。
- **规则抽取**：V1 开放句式上的反向任务仍靠规则；没有假装 15M 模型已经会抽取。
- **质控**：缺失、单位、左右叶冲突等确定问题不交给生成模型「自己觉得有没有错」。

工业界对应物：检索基线、正则/词典 NER、模板 NLG，必须出现在模型卡片里。没有基线的「SOTA」无法解释增量来自算法还是来自数据泄漏。

### 该读的文件

- `src/sonogpt/baselines/template_report.py`
- `src/sonogpt/baselines/rule_extract.py`
- `src/sonogpt/baselines/qc.py`
- [`baselines.md`](baselines.md)

规则不需要 GPU：

```powershell
uv run python scripts/evaluate_extract_baseline.py
uv run python scripts/evaluate_qc_baseline.py
```

（完整冻结 JSONL 齐备时才能复现已发表数字。）

---

## 6. 阶段 4：Tokenizer —— 把文本变成整数

神经网络不读汉字，只读 Token ID。Tokenizer 是数据和模型之间的契约，必须**先于模型冻结**，并且**只用训练集**学习词表。

### 6.1 为什么先做字符级

中文领域、短报告、大量数字和 JSON 符号，字符级的价值是：

- 词表可打印、可检查；
- `8.0`、`×`、`mm` 不会被错误预分词拆丢；
- 过拟合测试时，问题出在模型还是 BPE 可以立刻分开；
- 未见 Unicode 用 UTF-8 byte fallback，往返可逆，而不是变成一个万能 `<unk>`。

实现：`src/sonogpt/tokenizer/character.py`。控制符固定在词表开头：

```text
<pad> <bos> <eos> <unk> <task_extract> <task_generate> <input> <target>
```

### 6.2 正式训练用 SentencePiece BPE

BPE 把高频子词合成更短序列，降低 15M 模型的上下文压力。本仓库：

- `model_type="bpe"`，identity normalization（不改写规范 JSON）；
- byte fallback；
- 单线程、禁止随机打乱语料，保证哈希可复现；
- 保存前去掉 protobuf 里的绝对路径，避免「同一语料不同临时目录得到不同文件哈希」。

设计稿曾写词表 4096。受控合成语料实际最多形成 **1,807** 个有效 piece，再合并没有新符号。强行补到 4096 只会留下从未更新的 Embedding 行。**词表大小由数据决定，不由论文里的整数决定。**

验收必须包括：`decode(encode(text))` 可逆、数字与单位不丢、测试集 `<unk>` 率接近 0、序列长度不超过 `max_seq_len`。本数据上全部 15,500 条完整序列（prompt+目标）都小于 384。

### 6.3 和预训练模型的关系

自定义 1807 词表**只**与从零训练的 SonoGPT checkpoint 配套。若以后做 Qwen/Llama 的 LoRA 对照，必须用基座原词表，不能把这套 ID 塞进预训练权重还声称兼容。

### 跟练

字符级随过拟合脚本训练，不必单独命令。BPE（需要 5k 数据）：

```powershell
uv run python scripts/train_tokenizer.py
```

不要覆盖已冻结的 `artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/`。

### 该读的文件

- `src/sonogpt/tokenizer/base.py`
- `src/sonogpt/tokenizer/character.py`
- `src/sonogpt/tokenizer/sentencepiece_bpe.py`
- `src/sonogpt/tokenizer/validation.py`
- `tests/test_tokenizers.py`

---

## 7. 阶段 5：从零实现 Decoder-only Transformer

### 7.1 为什么是 GPT 而不是 Encoder-Decoder

字段转换用 Encoder-Decoder 或 Encoder+分类头往往更省。本项目主目标是**把 GPT 风格训练走通**，所以用单个 Causal LM，用任务 Token 区分方向（V1 实际只用 generate）。这是教学选择，不是生产上「最省算力」的声明。

### 7.2 信息在网络里怎么走

对一个 batch `[B, T]` 的 `input_ids`：

1. **Token Embedding**：每个 ID → `d_model` 维向量。
2. **位置 Embedding**：可学习的绝对位置，加到 Token 向量上（不是 RoPE；首版求可读）。
3. **Dropout**。
4. 重复 `n_layers` 次 **Pre-LN Block**：
   - LayerNorm → Causal Self-Attention → 残差；
   - LayerNorm → GELU FFN（先扩到 `d_ff` 再缩回）→ 残差。
5. 最终 LayerNorm。
6. **LM Head**：`d_model → vocab_size` 的线性层，得到每个位置上下一个 Token 的 logits。

Pre-LN（先归一化再算子模块）比 Post-LN 更容易把小模型训稳。`tie_embeddings=true` 让 LM Head 与 Token Embedding **共享同一块权重**，减少参数并强制「读和写使用同一套向量空间」。

Attention 内部：把隐状态线性映射成 Q、K、V，拆成 `n_heads` 个头，每个头维度 `d_model / n_heads = 64`，再 `scaled_dot_product_attention`。因果掩码保证位置 `t` 只能看见 `≤ t`；Padding 位置不能被当成可见上下文，输出侧也会被 mask 掉。

### 7.3 正式配置与参数量

`configs/model/sonogpt_16m_candidate.json`：

| 超参 | 值 | 直观含义 |
| --- | --- | --- |
| `vocab_size` | 1807 | 与冻结 BPE 一致 |
| `max_seq_len` | 384 | 超过则训练/生成报错，不截断 |
| `n_layers` | 8 | 残差块层数 |
| `n_heads` | 6 | 每层头数 |
| `d_model` | 384 | 隐状态宽度 |
| `d_ff` | 1536 | FFN 中间宽度（通常 4×d_model） |
| `dropout` | 0.1 | 嵌入、注意力残差、FFN |
| `bias` | false | Linear / LayerNorm 无偏置，略减参数 |
| `tie_embeddings` | true | 输入输出共享 Embedding |

实测 **15,003,648** 参数。可用下面的分解验算（无 bias、权重共享）：

```text
Token Embedding     1807 × 384           =    693,888
位置 Embedding       384 × 384           =    147,456
每层 Attention+FFN+LN                     ≈  1,770,240
× 8 层                                     = 14,161,920
最终 LayerNorm       384                 =        384
────────────────────────────────────────────────────
合计                                        15,003,648
```

LM Head 不另计，因为与 Token Embedding 是同一存储。

过拟合诊断用更小的 2 层模型（约 15 万参数）+ 字符级词表，目的是**几分钟内把损失打到接近 0**，不是追求生成质量。

### 7.4 生成：Greedy + 必须遇到 EOS

`SonoGPT.generate` 每一步取 `argmax`，遇到 `<eos>` 停止。超过 `max_seq_len` 或步数用尽仍无 EOS，抛 `GenerationLimitError`，**禁止静默截断后当成功**。结构化任务不要高温采样：随机性会破坏数字和枚举。

### 该读的文件

- `src/sonogpt/model/config.py`
- `src/sonogpt/model/attention.py`
- `src/sonogpt/model/block.py`
- `src/sonogpt/model/gpt.py`
- `tests/test_model.py`（因果泄漏、Padding、权重共享、保存加载）
- `scripts/inspect_model.py`

---

## 8. 阶段 6：把样本编成「预测下一个 Token」

GPT 训练的本质就一句：**给定前面的 Token，预测下一个 Token。** 条件生成靠 Loss Mask 实现「只在答案上算对错」。

### 8.1 一条样本长什么样

对报告部分计损失，对 JSON 和控制符不计损失：

```text
input_ids:  <bos> <task_generate> <input> {JSON...} <target> 甲 状 腺 ... <eos>
labels:     -100  -100            -100    -100...   -100     甲 状 腺 ... <eos>
```

`-100` 是 PyTorch `cross_entropy` 的 `ignore_index`。实现见 `encode_generate_sample`。

`forward` 里还有一次标准移位：用位置 `t` 的 logits 去预测 `labels[t+1]`。因此模型在读完 `<target>` 之后，才开始为报告的第一个字负责。

### 8.2 动态 Padding

一个 batch 里句子长短不同，右侧补 `<pad>`。Padding 位置：

- `attention_mask = False`，注意力看不到它们；
- `labels = -100`，损失忽略它们。

序列超过 `max_seq_len` 抛 `SequenceTooLongError`。静默截断会让模型以为「没写完的报告也是合法答案」。

### 该读的文件

- `src/sonogpt/data/dataset.py`
- `tests/test_dataset_encoding.py`

---

## 9. 阶段 7：过拟合测试 —— 证明训练代码没写错

正式堆数据之前，必须能在 **32～128 条样本上把训练损失打到接近 0**。如果这都做不到，5,000 步只会稳定地训练一个错误实现。

本仓库额外强调一点：过拟合集里，**每个语义病例只保留一种模板**（`location_first_v2`）。普通训练集允许同一 JSON 对应三种语序；若过拟合时也保留多种合法下一 Token，存在不可约损失，不能用来证明「优化器能记住这批数据」。

实测（CPU，字符级，2 层，800 步）：

| | |
| --- | --- |
| 样本 | 32 |
| 参数 | 156,096 |
| 初始 Loss | 5.95 |
| 最终 Loss | 0.021 |
| 目标 Token 准确率 | 0.015 → 0.995 |

### 跟练（无需 GPU、无需冻结 5k 数据）

```powershell
uv run python scripts/run_overfit.py
```

### 该读的文件

- `src/sonogpt/training/overfit.py`
- `src/sonogpt/training/reproducibility.py`
- `tests/test_overfit.py`

---

## 10. 阶段 8：正式训练

过拟合通过后，才进入「像那么回事」的训练器。本阶段对应工业界的 pretrain/finetune 循环，只是数据是领域合成样本、步数是 5,000 而不是百万。

### 10.1 优化器与学习率

- **AdamW**，矩阵权重与 Bias/LayerNorm 分参数组，Weight Decay 主要打在矩阵上。
- **线性 Warmup 200 步** 到 `3e-4`，再 **Cosine** 降到 `3e-5`。小模型不要直接抄 7B 的 `2e-5`。
- **梯度裁剪** 范数 1.0；非有限梯度跳过该次 optimizer step 并记账。
- **FP16 AMP** 仅 CUDA；CPU 自动关闭，避免把 CPU smoke test 写成 GPU 验证。RTX 2060（Turing）默认 FP16，不要假设 BF16。

### 10.2 有效 Batch 与梯度累积

笔记本显存装不下大 batch 时，用微批次多次反传再更新一次权重：

```text
有效 batch = micro_batch_size × gradient_accumulation_steps
         = 16 × 2 = 32
```

动态长度下，简单把微批次 loss 平均再累加，会让短句和长句权重失真。本训练器按**目标 Token 数加权**累积，使一次 optimizer step 近似「32 条样本的目标 Token 总损失」。

A/B 短测曾显示 Micro 16 吞吐最高。笔记本还要管温度：超过阈值暂停，降到恢复温度再继续，而不是直接杀进程。正式 5000 步在 RTX 2060 Laptop 上温控暂停 96 次、累计约 5 分钟，这是真实硬件约束，不是算法失败。

### 10.3 Checkpoint 必须能精确恢复

一次中断后继续训练，若 Shuffle 顺序、RNG、Scaler、Scheduler 对不上，就不是同一条实验。本仓库 checkpoint 含：

- 模型 / 优化器 / 调度器 / GradScaler；
- 全局步、样本数、最佳验证 Loss；
- 可序列化的 epoch + cursor（确定性 Shuffle 流）；
- Python / NumPy / PyTorch CPU 与 CUDA RNG；
- 数据 manifest、Tokenizer、配置、冻结记录的哈希。

恢复时发现数据、Tokenizer、配置或设备不一致会直接拒绝。单元测试验证：连续训练 4 步，与「2 步 → 保存 → 新进程再 2 步」的权重与计数一致。

### 10.4 用哪一个 Checkpoint

正式训练 5000 步后：

| | Step 4900 | Step 5000 |
| --- | --- | --- |
| 角色 | 验证 Loss 最佳（0.056965） | 最后一步（0.057462） |
| 评估默认 | **使用这个** | 仅作对照 |

不要默认「训到最后一步最好」。过拟合验证集或学习率已经很小之后，最后几步可能略差。本项目评估显示 4900 的已见模板「任一训练模板匹配」略高于 5000。

### 跟练

CPU 只跑 2 步，验证入口和恢复，不代替正式训练：

```powershell
uv run python scripts/train_model.py --smoke-test
```

正式 GPU 训练（会校验冻结数据与 Tokenizer，耗时约十几分钟级，含温控）：

```powershell
uv run python scripts/train_model.py --device cuda
```

**不要覆盖** `artifacts/training/sonogpt_16m_m3/` 里已发表的 4900/5000，除非你明确在做新实验并换输出目录。

### 该读的文件

- `src/sonogpt/training/trainer.py`
- `src/sonogpt/training/config.py`
- `src/sonogpt/training/scheduler.py`
- `src/sonogpt/training/checkpoint.py`
- `configs/training/sonogpt_16m_m3.json`
- `scripts/train_model.py`
- `tests/test_training.py`
- `reports/training/sonogpt_16m_m3_5000step_20260824.json`

---

## 11. 阶段 9：评估 —— 数字必须能被拆穿

### 11.1 评生成，不要只评 BLEU

同一结构有三种已见合法写法，逐字 BLEU/精确匹配会惩罚「写对了但换了语序」的模型。本项目主指标是：

1. 生成报告；
2. 用**独立规则解析器**抽回字段（不是同一个 Transformer 再做 extract）；
3. 与金标准 JSON 比明示字段、尺寸、幻觉。

辅以：是否恰好等于某个训练模板、Teacher-forced Token 准确率（留出语序上会很难看，这是预期）。

### 11.2 三套集合分开读

默认 Checkpoint `step_00004900.pt`（数字来自冻结评估，可复现）：

| 测试集 | 样本 | 逐字匹配 | 任一训练模板匹配 | 明示字段准确率 | 尺寸精确匹配 |
| --- | --- | --- | --- | --- | --- |
| 已见模板语序 | 1,500 | 0.3320 | 0.9960 | 0.9997 | 1.0000 |
| 留出模板语序 | 500 | 0.0000 | 0.9960 | 0.9997 | 1.0000 |
| 模拟改写挑战集 | 30 | 0.0000 | 0.9667 | 1.0000 | 0.9630 |

怎么读：

- **已见 33% 逐字 vs 99.6% 任一模板**：模型常写出「另一个合法家族」，语义仍对齐。这不是失败。
- **留出逐字 0、字段仍几乎全对、且 0 次写出 `flow_first_v2`**：模型没有发明未见句式，但把结构映射到已见句式里的内容。这是「内容泛化」，不是「文风泛化」。
- **挑战集 Teacher-forced 很难、字段仍对**：参考句是改写，模型仍渲染成训练模板。不能写成「通过了真实医生挑战」。

**禁止**把三行平均成一个「总准确率」。分布不同、难度不同、声称也不同。

### 11.3 规则基线对照告诉你什么

抽取规则在受控模板和挑战改写上明示字段准确率 1.0、幻觉率 0；整条 JSON Exact Match 较低，是因为金标准 `unknown` 被规则写成 `not_mentioned`。评估把二者视为不同——**诚实暴露规则边界**，而不是把近似当成 Exact。

质控：干净样本 error 假阳性 0；11 类注入缺陷 recall 1.0。这证明规则在**自己定义的缺陷类型**上可测，不证明临床指南。

### 跟练

```powershell
uv run python scripts/evaluate_generate.py --device cuda
```

### 该读的文件

- `src/sonogpt/evaluation/report_parser.py`
- `src/sonogpt/evaluation/metrics.py`
- `src/sonogpt/evaluation/generation.py`
- `src/sonogpt/evaluation/pipeline.py`
- [`evaluation.md`](evaluation.md)
- `reports/evaluation/sonogpt_16m_m3_generate_20260825.md`

---

## 12. 阶段 10：推理与展示

训练好的权重如果不能走完「校验 → 生成 → 质控 → 失败可见」，就还不是一个系统。

### 12.1 generate 处理链

```text
结构 JSON
  → Schema 校验（非法直接失败）
  → 规范序列化（与训练时同一套 canonical JSON）
  → Greedy 生成直到 EOS
  → 规则 QC（报告 vs 输入结构）
  → 无 error：返回模型报告
  → 有 error 或生成失败：确定性模板回退，并记录 used_fallback
```

产品含义：模型是草稿生成器，模板是安全网。学习项目里这叫诚实；上线系统里这叫降级策略。

`extract` / `qc` **不加载 15M 权重**。页面和 CLI 都标明抽取不是任务模型。

### 12.2 Demo 为什么是表单不是聊天框

开放聊天会暗示「通用医疗助手」。本 Demo 的输入就是 Schema 字段，输出是所见草稿 + 模板对照 + 规则抽取。默认只绑定 `127.0.0.1`。

```powershell
uv run python scripts/infer.py generate --device cuda --json-file reports/inference/generate_smoke_input.json
uv run python scripts/serve_demo.py --device cuda
```

无 GPU 用 `--device cpu`。缺 Checkpoint 时网页仍可「只看模板」。

### 该读的文件

- `src/sonogpt/inference/engine.py`
- `scripts/infer.py`
- `src/sonogpt/web/app.py`
- [`inference.md`](inference.md)、[`web.md`](web.md)

---

## 13. 建议的学习顺序（读代码 + 动手）

下面按「认知负担」排序。每一行做完，你应能向别人讲清楚**为什么要有这一步**。

| 步 | 做什么 | 通过标准 |
| --- | --- | --- |
| 1 | `uv sync --extra dev` 与 `uv run pytest` | 测试全绿 |
| 2 | 读 `schemas/domain.py`，打开一条夹具 JSON | 能解释 `none` vs `not_mentioned` |
| 3 | 读 `template_report.py`，对照夹具预期文本 | 知道模板零幻觉从何而来 |
| 4 | 读 `semantic_generator.py` + `split.py` | 能画出「病例切分 + 留出语序」 |
| 5 | `generate_data.py --count 100` | 本地有 JSONL 与 manifest |
| 6 | 读 `dataset.py` 与 `model/*.py` | 能在纸上写出一条样本的 labels |
| 7 | `run_overfit.py` | Loss 明显降到接近 0 |
| 8 | 读 `trainer.py` 的累积损失与 checkpoint | 能说明有效 batch 和恢复字段 |
| 9 | 读评估报告，对照第 11 节 | 能解释 33% 逐字 vs 99.6% 模板匹配 |
| 10 | 有权重则 `infer.py generate` 或网页 | 能指出 QC 与模板回退发生在哪 |

有冻结 5k 数据与 Tokenizer 后，可加：`train_tokenizer.py`（新输出目录）、`train_model.py --smoke-test`、`evaluate_*.py`。

脚本地图：

```text
scripts/generate_data.py              语义采样 + 切分 + manifest
scripts/train_tokenizer.py            仅用 train.jsonl 训 BPE
scripts/inspect_model.py              打印参数量与词表一致性
scripts/run_overfit.py                字符级 32 条过拟合
scripts/train_model.py                正式训练 / CPU smoke
scripts/freeze_data.py                冻结与校验
scripts/build_simulated_challenge.py  挑战集构建与校验
scripts/evaluate_generate.py          三集 generate
scripts/evaluate_extract_baseline.py  规则抽取
scripts/evaluate_qc_baseline.py       质控
scripts/infer.py                      本地 CLI
scripts/serve_demo.py                 本机网页
```

---

## 14. 这个闭环教到了什么，下一步学什么

### 已经覆盖的「大模型开发」核心

- 任务与数据契约先于网络结构。
- 合成数据的正确分层与防泄漏切分。
- Tokenizer 与模型版本绑定、只用训练集。
- 标准 GPT 模块：Embedding、因果注意力、FFN、权重共享、自回归解码。
- Causal LM 的 Loss Mask、移位、Padding。
- 用过拟合区分「代码错」和「数据不够」。
- AdamW、Warmup+Cosine、AMP、梯度累积、可恢复 checkpoint。
- 独立评估器、基线、冻结测试集、禁止拍脑袋平均。
- 推理侧 Schema 校验、失败显式化、规则与模型分工。

### 本仓库刻意没做、需要另开项目才能学到的

| 主题 | 为什么 V1 没有 |
| --- | --- |
| 海量预训练 | 目标是从零实现，不是复现互联网语料训练 |
| 指令微调 / RLHF / DPO | 没有开放对话与偏好数据 |
| 多任务 extract 模型 | 先把 generate 评估做严；抽取用规则当对照 |
| 量化、KV cache、推理引擎 | 15M greedy 在笔记本上不是瓶颈 |
| Encoder-Decoder / 条件分类头 | 教学主线是 GPT |
| 多模态超声图 | 图像质控、配对与授权是另一条数据问题 |
| 真实病历 | 涉及授权、脱敏、与合成数据分开统计 |

若继续深造，更有迁移价值的顺序通常是：换一个仍有 Schema 的文本任务（检验你是否只会「甲状腺模板」）→ 在同一套评估纪律下接入小型预训练模型 LoRA 对照 → 再碰开放生成与对齐。不要一上来用本仓库的 99.97% 字段准确率去类比 ChatGPT。

### 常见错觉（项目里专门防过的）

1. **条数多 = 数据好。** 几十个模板重复一百万次，模型只是在背拼接规则。
2. **Loss 下降 = 任务完成。** 必须看字段、尺寸、留出语序和幻觉。
3. **Attention 热图 = 模型在思考。** 本仓库甚至不把可视化写成「可解释诊断」。
4. **用生成模型做质控。** 确定约束留给规则，模型负责语言映射。
5. **测试集上反复调参。** 验证集选模型，测试集冻结。
6. **静默截断 / 静默修 JSON。** 失败必须可见，否则指标在说谎。

---

## 15. 一页流程总图

```text
冻结 Schema v1
    → 语义病例采样 + 合成约束
    → 多模板渲染
    → 按 semantic_case_id 切分 + 留出 flow_first_v2
    → 规则模板 / 抽取 / 质控基线
    → 冻结数据与挑战集（SHA-256）
    → 字符级 Tokenizer + 32 条过拟合
    → 训练集上的 BPE（1807）冻结
    → 15M Decoder-only 训练（有效 batch 32，5000 step）
    → 验证 Loss 选 4900 Checkpoint
    → 独立解析器评估三套集合
    → CLI / 本机网页：生成 + QC + 模板回退
```

把这一页讲顺，你就已经走完一遍「从 0 到 1」的语言模型开发。其余只是把每个框里的规模、数据和对齐算法换大。
