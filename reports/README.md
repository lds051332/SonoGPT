# 报告目录

本目录保存可复现的评估与冒烟记录。数字以脚本产物为准，解读文档在 `docs/`。

| 子目录 | 内容 |
| --- | --- |
| `training/` | RTX 2060 正式 5000 Step 训练摘要 |
| `evaluation/` | 冻结三集 generate 评估（模型） |
| `baselines/` | 规则抽取与质控基线 |
| `inference/` | 公司机 CUDA 单条推理冒烟 |
| `web/` | 本地网页 Demo 的 HTTP 冒烟 |

不要把已见 / 留出 / 挑战集成绩平均。不要把挑战集称为真实人类或临床集。不要覆盖冻结数据或正式 Checkpoint。
