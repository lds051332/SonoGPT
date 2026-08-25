# 无 GPU 的 Linux 部署（CPU 推理）

面向没有 NVIDIA GPU 的 Linux 机器：把网页 Demo 跑起来，用 CPU 生成报告。**不是临床系统，不能用于诊断。**

仓库默认 `uv sync` 会在 Linux 上安装 `torch 2.12.0+cu126`（大约 2～3 GB）。无 GPU 时既占磁盘，也更容易把小内存机器撑爆。请按本文安装 **CPU 版 PyTorch**。

## 机器要什么

| | 最低 | 更稳妥 |
| --- | --- | --- |
| CPU | 2 核 | 2～4 核 |
| 内存 | 2 GB（很紧，建议加 swap） | **4 GB 及以上** |
| 磁盘 | 约 5 GB 空闲（系统盘） | 10 GB+ |
| Python | 3.11 或 3.12 | 系统已有 3.12 最好 |
| GPU | 不需要 | — |

15M 模型权重文件约 **172 MB**；参数本身约 57 MiB。启动时会把完整 Checkpoint（含优化器状态）读进内存，所以不要和「参数量很小」划等号。CPU 上生成一条报告通常是数秒，比本机 CUDA 慢。

已有 GitLab / Nginx 的机器：不要去抢 80/443。本 Demo 默认 **8000**。

## 要上传的文件

Git 里没有 Tokenizer、合成 JSONL 和 Checkpoint。clone 之后还要放这些路径（名称必须一致）：

```text
artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/sonogpt_bpe.model
artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/tokenizer_manifest.json
data/processed/synthetic_v1_5k_candidate_v2/train.jsonl
data/processed/synthetic_v1_5k_candidate_v2/validation.jsonl
data/processed/synthetic_v1_5k_candidate_v2/test_seen_templates.jsonl
data/processed/synthetic_v1_5k_candidate_v2/test_heldout_templates.jsonl
data/processed/synthetic_v1_5k_candidate_v2/statistics.json
artifacts/training/sonogpt_16m_m3/step_00004900.pt
```

合计大约 **186 MB**。网页生成只用 4900 这一份权重。启动时会校验冻结数据哈希，JSONL **不能省**。

从 Windows 本机拷（先在服务器建好空目录）：

```powershell
scp artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/sonogpt_bpe.model `
    artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/tokenizer_manifest.json `
    user@HOST:~/SonoGPT/artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2/

scp data/processed/synthetic_v1_5k_candidate_v2/* `
    user@HOST:~/SonoGPT/data/processed/synthetic_v1_5k_candidate_v2/

scp artifacts/training/sonogpt_16m_m3/step_00004900.pt `
    user@HOST:~/SonoGPT/artifacts/training/sonogpt_16m_m3/
```

## 1. 源码

```bash
sudo apt-get update
sudo apt-get install -y git curl
git clone https://github.com/lds051332/SonoGPT.git
cd SonoGPT
```

没有 sudo 时，用发行版已有的 `git`/`curl` 即可。

## 2. 安装 uv（用户态，不必装系统 Python）

系统已是 Python 3.12 也可以装 uv，用来按锁文件装依赖。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

若系统 Python 低于 3.11：

```bash
uv python install 3.12
```

## 3. 放产物

把上一节的 Tokenizer、JSONL、Checkpoint 拷到对应目录。然后：

```bash
mkdir -p \
  artifacts/tokenizers/sonogpt_bpe_1807_candidate_v2 \
  artifacts/training/sonogpt_16m_m3 \
  data/processed/synthetic_v1_5k_candidate_v2
ls -lh artifacts/training/sonogpt_16m_m3/step_00004900.pt
```

`step_00004900.pt` 应约为 172 MB。

## 4. 安装 CPU 版依赖

不要在这台机上直接 `uv sync`。使用：

```bash
bash scripts/sync_cpu.sh
```

等价于：先同步除 torch 以外的依赖，再从 CPU 索引重装 `torch==2.12.0`。期望输出类似：

```text
torch 2.12.0+cpu
cuda False
```

若看到 `2.12.0+cu126` 或 `cuda True`，说明仍是 CUDA 轮子，不要继续，重新执行脚本。

以后若要更新其它包：

```bash
uv sync --extra dev --no-install-package torch --inexact
```

不要省略 `--no-install-package torch`，否则会改回 cu126。

## 5. 冒烟

```bash
uv run pytest
uv run python scripts/infer.py generate --device cpu --json-file reports/inference/generate_smoke_input.json
```

`pytest` 不加载正式 Checkpoint。`infer.py generate` 会走 15M 模型，第一次可能要十几秒。

## 6. 启动网页

默认只绑本机，适合 SSH 隧道，避免把 Demo 直接挂到公网。

服务器：

```bash
uv run python scripts/serve_demo.py --device cpu --host 127.0.0.1 --port 8000
```

你的电脑：

```powershell
ssh -L 8000:127.0.0.1:8000 user@HOST
```

浏览器打开 http://127.0.0.1:8000/ 。缺权重时仍可「只看模板」；生成需要 Checkpoint。

后台常驻可以用 `tmux` / `systemd --user`，工作目录必须是仓库根目录，命令与上面相同。

## 7. 可选：本机反代（仍不要默认 `0.0.0.0`）

需要局域网或域名访问时，让 Nginx/Caddy 反代到 `127.0.0.1:8000`，并自己管 TLS 与访问控制。不要把 `scripts/serve_demo.py --host 0.0.0.0` 当成默认上线方式。

示例（需 sudo，端口按机器上已占用情况改）：

```nginx
server {
    listen 8088;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }
}
```

页面上有 GitLab（例如 8099）时，给 Demo 另开端口，不要覆盖 GitLab。

## 不要做的事

- 不要在无 GPU 的 Linux 上直接 `uv sync`（会拉 CUDA 轮子）
- 不要覆盖冻结 Tokenizer、JSONL 或 `step_00004900.pt`
- 不要在这台云主机上重训 5000 Step
- 不要把页面说成临床系统或 TI-RADS
- 不要把规则抽取说成模型抽取

相关说明：[`web.md`](web.md)、[`inference.md`](inference.md)。
