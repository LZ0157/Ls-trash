# Ls-trash — 智能体对话不一致检测

> 这是一个**仍在迭代中**的毕设项目。原作者用 prompt + GLM-4-Flash 做了一个端到端 baseline，
> 后续的人重新设计了架构，目前 Track A (硬冲突检测) 已经有一个可运行的 NLI baseline。
> Track B (软偏移检测) 尚未开始。
>
> **接手前请先读 [HANDOFF.md](HANDOFF.md)**——它包含完整的进度盘点、已淘汰路线、推荐下一步、以及环境/约束注意事项。

---

## 项目目标 (原作者表述)

> 毕设核心目标是检测智能体多轮对话中的"不一致"问题。经过前期调研和中期答辩的梳理，
> 我们将"不一致"拆解为两个需要分别处理的层级：
>
> - **硬冲突 (逻辑矛盾)**：事实错误、角色混淆、历史陈述自相矛盾等。（对应第一阶段）
> - **软偏移 (主题漂移)**：没有明显逻辑错误，但话题逐渐偏离用户初始意图。（对应任务书核心要求及第二阶段）

详见 [`重设计方案.md`](重设计方案.md) — 当前架构的设计原则与论证。

## 当前状态 (2026-04-07)

| 模块 | 状态 | 当前最佳指标 |
|---|---|---|
| **Track A** (硬冲突检测) | NLI baseline 已完成 + 阶段 1 ablation 已完成 | binary F1 = **0.7456**（200 平衡测试集）|
| **Track B** (软偏移检测) | **未开始** | — |

Track A 的核心方法：CDConv 数据 → 中文句切 → Erlangshen NLI 模型逐句对判别 → 后处理分类。
**全程 CPU 推理**，不依赖任何 LLM API。

## 快速开始

### 1. 环境

```bash
conda create -n track_a python=3.11 -y
conda activate track_a
pip install -r requirements.txt
```

### 2. 下载 NLI 模型 (~1.3GB)

需要 HuggingFace token (放在 `~/.cache/huggingface/token`，本仓库**不提供**任何 token):

```bash
mkdir -p /path/to/your/hf_cache/local-erlangshen-330m-nli
cd /path/to/your/hf_cache/local-erlangshen-330m-nli

TOKEN=$(cat ~/.cache/huggingface/token)
for f in config.json vocab.txt pytorch_model.bin; do
  curl -L -H "Authorization: Bearer $TOKEN" \
    -o "$f" \
    "https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-NLI/resolve/main/$f"
done
```

下载若卡顿（xet 后端常见），见 HANDOFF.md 的"模型下载排错"节。

### 3. 修改模型路径

`track_a/nli.py` 顶部:

```python
DEFAULT_MODEL = "/path/to/your/hf_cache/local-erlangshen-330m-nli"
```

### 4. 跑 smoke test

```bash
python -m track_a.nli                                    # NLI 模型自检
python -m track_a.evaluate --n_per_class 5 --fine_split  # 20 样本评测
```

### 5. 跑正式评测

```bash
python -m track_a.evaluate \
  --n_per_class 50 \
  --fine_split \
  --output results/my_eval.json
```

## 当前最佳数字 (200 条平衡测试集)

| 指标 | 值 |
|---|---|
| Binary precision | 0.781 |
| Binary recall | 0.713 |
| **Binary F1** | **0.7456** |
| 4-class accuracy | 0.30 |
| 4-class macro F1 | 0.23 |
| 推理速度 | ~9.6 样本/秒 (CPU) |

完整指标表与每条样本的预测细节见 `results/`。

## 文档导览

| 文件 | 内容 |
|---|---|
| **[HANDOFF.md](HANDOFF.md)** | **接手指南**——必读 |
| [重设计方案.md](重设计方案.md) | 架构设计文档（双 pipeline 思路） |
| [results/track_a_baseline_report.md](results/track_a_baseline_report.md) | Baseline 评测主报告 |
| [results/track_a_stage1_report.md](results/track_a_stage1_report.md) | 阶段 1 ablation 实验记录（含失败的尝试） |

## 代码结构

```
track_a/
├── data.py        — CDConv loader + 分层抽样
├── segmenter.py   — 中文按句号/逗号切句 + claim 过滤 (零依赖)
├── nli.py         — Erlangshen NLI 封装 (CPU 强制)
├── pipeline.py    — 句对生成 → NLI → 后处理分类
└── evaluate.py    — CLI 入口 + 指标计算 + 详细结果导出
```

每个模块独立可替换。比如想换 NLI 模型，只改 `nli.py` 的 `DEFAULT_MODEL`。

## 数据来源与引用

数据集 `cdconv.txt` 来自 CDConv (Zheng et al. 2022, EMNLP)，用于学术研究。
原仓库：<https://github.com/THU-coai/CDConv>

## 已知约束

- **无 GPU**：本项目所有评测都在 CPU 上跑，是出于环境约束。fine-tune NLI 这类训练任务需要接手人自己解决 GPU 问题（见 HANDOFF.md "下一步建议"）。
- **数据没有 train/test 标准切分**：`cdconv.txt` 是混合数据，需要自切或下载官方 split。
- **类 2 (角色混淆) 当前无法检测**：CDConv 数据不带 persona 字段，本基线对类 2 的 F1 = 0.00。

## License

代码部分：暂未指定，建议接手后视用途选择（MIT / Apache-2.0 / CC-BY-NC-4.0 等）。
数据部分：遵循 CDConv 原始 license。
