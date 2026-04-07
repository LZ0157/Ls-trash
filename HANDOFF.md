# 转手指南 (HANDOFF)

> 这份文档是给**下一个接手人**的——不是公开 README，是项目内部沟通。
> 写这份文档的人 (上一手) 不是原作者，而是中间接管过项目的人。
> **读完这份再动代码**，能避免重复踩 5-6 个已经踩过的坑。

---

## TL;DR (5 句话)

1. **项目目标**：检测智能体多轮对话中的"硬冲突"和"软偏移"。详见 `重设计方案.md`。
2. **当前进度**：Track A (硬冲突) 已有 NLI baseline，binary F1 = 0.7456。Track B (软偏移) **未开始**。
3. **当前最优方案**：Erlangshen-Roberta-330M-NLI + 句切 + 简单后处理。已验证 7 个改进变体**全部**比 baseline 差，原因是 NLI 模型本身存在 PR 曲线天花板。
4. **强烈推荐的下一步**：**fine-tune NLI** (LoRA on CDConv)。这是唯一被分析认为能突破当前天花板的路。
5. **核心约束**：CPU only (开发环境无 GPU)。fine-tune 需要接手人自己解决 GPU 问题，或接受 CPU 训练 4 小时一个 epoch 的代价。

---

## 一、当前进度盘点

| 模块 | 状态 | 文件位置 |
|---|---|---|
| 重设计方案 (架构文档) | ✅ 已完成 | `重设计方案.md` |
| Track A — Baseline | ✅ 已完成并评测 | `track_a/`, `results/track_a_baseline_report.md` |
| Track A — 阶段 1 改进尝试 | ✅ 已完成 (全部失败但有价值) | `results/track_a_stage1_report.md` |
| Track A — 阶段 2 (entity gating, class 2 解锁) | ⏸ 未开始 | — |
| Track A — fine-tune NLI | ⏸ 未开始 (但有方案) | 见本文档 §六 |
| Track B (软偏移检测) | ⏸ **完全未开始** | — |

---

## 二、关键文件导览

```
Ls-trash/
├── README.md                       # 公开面向，给 GitHub 上来的人看
├── HANDOFF.md                      # 你正在读的这个
├── 重设计方案.md                   # 整个项目的设计原则与论证 (必读)
├── requirements.txt                # 依赖
├── .gitignore
│
├── cdconv.txt                      # CDConv 数据集 (11660 条 JSONL)
│
├── track_a/                        # Track A 的全部代码
│   ├── data.py                     # CDConv loader + 分层抽样
│   ├── segmenter.py                # 中文切句 + claim 过滤 (零依赖)
│   ├── nli.py                      # Erlangshen NLI 封装 (CPU 强制)
│   ├── pipeline.py                 # 句对生成 → NLI → 后处理
│   └── evaluate.py                 # CLI 入口 + 指标
│
└── results/
    ├── track_a_baseline_report.md  # Baseline 评测主报告 (必读)
    ├── track_a_stage1_report.md    # 阶段 1 ablation 实验记录 (必读)
    ├── track_a_eval_200.json       # baseline 详细结果 (含 200 条预测+conflict)
    ├── track_a_v_filter*.json      # filter 变体结果 (失败)
    ├── track_a_v_bidir*.json       # 双向 NLI 变体结果 (失败)
    ├── track_a_v_doc*.json         # document mode 结果 (失败)
    └── track_a_t{0.5,...}.json     # threshold 扫描结果
```

**阅读顺序建议**: `重设计方案.md` → `results/track_a_baseline_report.md` → `results/track_a_stage1_report.md` → 本文档

---

## 三、环境配置

### 3.1 Python / 包

```bash
conda create -n track_a python=3.11 -y
conda activate track_a
pip install -r requirements.txt
```

实测在 `gemma4` env (上一手用的) 跑通：
- torch 2.10.0+cu128 (CUDA 部分用不到，纯 CPU 推理)
- transformers 5.5.0
- scikit-learn 1.8.0
- huggingface_hub 1.9.0

### 3.2 NLI 模型 (1.3 GB)

模型在 HuggingFace 上是 `IDEA-CCNL/Erlangshen-Roberta-330M-NLI`。

**坑**: HF 把这个模型迁到了新的 xet 后端 (`cas-bridge.xethub.hf.co`)，**未认证用户限速极严**，单线程 curl 会卡死或被限流。

下载需要：
1. 一个 HF token (read 权限即可，免费注册)
2. 写到 `~/.cache/huggingface/token` (mode 600)
3. 用并行 range 下载 (单线程会卡)

上一手用的下载脚本（可直接复用，注意改 `OUT` 路径）：

```bash
#!/bin/bash
TOKEN=$(cat ~/.cache/huggingface/token)
URL="https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-NLI/resolve/main/pytorch_model.bin"
OUT="/path/to/local-erlangshen-330m-nli/pytorch_model.bin"
TOTAL=1302263725
PARTS=8
PART_SIZE=$(( (TOTAL + PARTS - 1) / PARTS ))
mkdir -p "$(dirname "$OUT")"

for i in $(seq 0 $((PARTS - 1))); do
    START=$((i * PART_SIZE))
    END=$(( (i + 1) * PART_SIZE - 1 ))
    [ "$END" -ge "$TOTAL" ] && END=$((TOTAL - 1))
    (curl -sL --connect-timeout 30 --max-time 1500 \
       --retry 5 --retry-delay 5 --retry-all-errors \
       -H "Authorization: Bearer $TOKEN" \
       -H "Range: bytes=${START}-${END}" \
       -o "${OUT}.part${i}" "$URL") &
done
wait
cat "${OUT}.part"* > "$OUT" && rm "${OUT}.part"*

# config 和 vocab 是小文件，单 curl 即可
cd "$(dirname "$OUT")"
for f in config.json vocab.txt; do
  curl -sL -H "Authorization: Bearer $TOKEN" \
    -o "$f" \
    "https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-NLI/resolve/main/$f"
done
```

下载完后改 `track_a/nli.py` 顶部:
```python
DEFAULT_MODEL = "/path/to/your/local-erlangshen-330m-nli"
```

### 3.3 验证

```bash
python -m track_a.nli   # 应该输出 3 个测试句对的 NLI 概率
```

正常输出包括 `Label map: {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}` 和测试 case 的概率。

---

## 四、如何运行评测

### Smoke test (20 样本，~5 秒)
```bash
python -m track_a.evaluate --n_per_class 5 --fine_split --output results/_smoke.json
```

### 正式评测 (200 样本，~21 秒)
```bash
python -m track_a.evaluate --n_per_class 50 --fine_split --output results/eval_200.json
```

### 全集评测 (11660 样本，估计 ~20 分钟)
```bash
python -m track_a.evaluate --n_per_class 99999 --fine_split --output results/eval_full.json
```

### 已支持的全部 flag
| flag | 意义 | 推荐值 |
|---|---|---|
| `--n_per_class` | 每类抽样数 | 50 (200 总) 或 99999 (全集) |
| `--threshold` | NLI contradict 概率阈值 | 0.5 (默认最优) |
| `--fine_split` | 按逗号细切 | **必开** |
| `--filter_claims` | 过滤问句/填充词 | 不开 (已验证伤 recall) |
| `--bidirectional` | 双向 NLI 校验 | 不开 (已验证伤 recall) |
| `--mode {sentence,document}` | 切句对或整段对 | sentence |
| `--seed` | 抽样随机种子 | 42 |

---

## 五、当前最佳数字 (200 平衡测试集，seed=42)

```
Predict +    TP    FP    FN    Precision  Recall    F1
   137      107   30    43     0.781      0.713    0.7456   ← 当前 baseline
```

各类别细分:
| 类别 | precision | recall | f1 |
|---|---|---|---|
| 0 无矛盾 | 0.32 | 0.40 | 0.35 |
| 1 句内 | 0.30 | 0.12 | 0.17 |
| 2 角色混淆 | **0.00** | **0.00** | **0.00** |
| 3 历史 | 0.29 | 0.68 | 0.41 |

类 2 = 0.00 是因为 CDConv 数据不带 persona 字段，本基线无法识别角色。**这是已知的、结构性的问题**。

---

## 六、推荐的下一步 (按优先级)

### 🥇 优先级 1: Fine-tune NLI on CDConv

**为什么是首选**：阶段 1 实验证明，NLI 模型本身存在 PR 曲线天花板 (precision ~0.78, recall ~0.71)。任何输入侧/输出侧的过滤都只能在这条线上滑动，不能突破。要突破，必须改 NLI 模型本身的判别能力。

**方案 (CPU 友好)**:

```
1. 下载 CDConv 官方 train/test split (避免和 baseline test set 污染)
2. pip install peft  # LoRA 支持
3. 在 train split 上做 LoRA fine-tune Erlangshen
   - rank=16, alpha=32, lr=3e-4
   - target modules: query, value
   - 3-class NLI loss (entail/neutral/contradict)
   - batch_size=16, epochs=3-5
   - early stop on dev macro F1
4. 在 test split 上评测，对比 baseline F1=0.7456
```

**时间预算 (76 线程 CPU)**:
- LoRA fine-tune: 单 epoch ~15 分钟 → 总 ~45 分钟
- 全参数 fine-tune: 单 epoch ~75 分钟 → 总 ~4 小时

**预期增益**: F1 +0.07 ~ +0.15 (拍脑袋估计)

**注意事项**:
- 必须用**官方 CDConv split**，不能在自测试集上 fine-tune (会污染评测)
- 类不平衡严重 (7309:530:765:3056)，需要加权 loss 或重采样
- 监控**dev set** 上的 F1，避免在 test 上过拟合超参

### 🥈 优先级 2: 解锁 Class 2 (角色混淆) 检测

当前 class 2 的 F1 = 0.00。任何让它 > 0 的做法都是论文里的可写 contribution。

**方案**:
1. 用 pattern 识别 b1 中的"身份/状态断言":
   - `我是X` / `我会X` / `我不会X` / `我学过X` / `我有X` / `我没有X`
   - `我从XX毕业` / `我在XX工作`
2. 如果 b2 与上述 pattern 中的某条 NLI 矛盾 → 输出 label **2** 而不是 label 3
3. 这是个**规则-based hack**，不是真正的 persona 模型，但论文里能写

**时间预算**: 1-2 小时实现 + 评测

### 🥉 优先级 3: 实词重叠门控

阶段 1 留下来的唯一可能突破 PR 曲线的方向（其他都验证失败了）。

**思路**: 在 NLI 之前，先用 jieba 分词检查两个句子是否共享实词。完全无重叠的句对 (像 `你是哪个学校` × `我回家了`) 直接判 neutral，不送 NLI。

**预期**: 砍掉"话题切换"类 FP（约 12 个），且**不太可能伤 recall** (真矛盾几乎都共享话题)。

**时间预算**: 30-60 分钟

### 优先级 4: Track B 起步 (软偏移检测)

整个 Track B 还没开始。设计文档在 `重设计方案.md` 第三节。核心思路:
- 用 LLM 抽取用户初始 goal
- 每轮用 LLM judge 打 alignment 分
- **不要用 embedding 余弦相似度**（设计文档解释了为什么这是错的）

需要的资源:
- 一个能用的 LLM (本地或 API)
- 自己标 100-500 条带漂移标签的中文长对话 (这是论文 contribution)

---

## 七、已淘汰的方向 (避免重蹈覆辙)

阶段 1 已经测试过的方向，结论都是"比 baseline 差"。**别再尝试这些**:

| 方向 | 实测 F1 | 失败原因 |
|---|---|---|
| Bidirectional NLI (`NLI(p,h)` AND `NLI(h,p)`) | 0.625 | Erlangshen 在反方向上对真矛盾不敏感 (位置偏差) |
| Bidirectional + threshold=0.3 | 0.656 | 同上, AND 约束太强 |
| Document mode (整段对整段) | 0.565 | NLI 训练数据是短句对，长输入分散注意力 |
| Doc + bidir | 0.400 | 复合错误 |
| Aggressive filter (整段丢 ?-句) | 0.625 | 把 "我24了, 你呢?" 整段丢, 损失 32 个真 TP |
| Refined filter (per-clause) | 0.695 | 仍然伤 recall |
| Filter + bidir | 0.539 | 复合错误 |

**所有这些都在同一条 PR 曲线上滑动**——precision 涨得不够补偿 recall 的跌。详细分析见 `results/track_a_stage1_report.md`。

---

## 八、已知约束

### 8.1 GPU 不可用 (上一手的环境约束)

上一手的开发环境**严格禁止使用任何 GPU**——共享集群上 8 张 A100 都被别的进程占用，且不允许"叠加进程占用同一张 GPU"。

接手人如果有自己的 GPU，**优先级 1 (fine-tune NLI) 立刻可做**。如果没有 GPU，只能在 CPU 上跑（时间预算见 §六）。

### 8.2 数据集没有标准 train/test split

`cdconv.txt` 是 11660 条混合数据，没有标记 train/test。当前评测都是从全集 stratified sample 200 条做的。

**问题**: 这意味着如果做 fine-tune，必须**先解决 split 问题**，否则会污染评测。两条路:
- A. 自切 80/20 (固定 seed)
- B. 联网下载 CDConv 官方 split (推荐)

### 8.3 Class 2 (角色混淆) 当前无法检测

CDConv `cdconv.txt` 数据**不带 persona 字段**，无法判断"角色矛盾"。当前 baseline 把所有跨轮矛盾都映射到 label 3，导致 class 2 的 F1 = 0.00。

详见 §六优先级 2 的解锁方案。

### 8.4 NLI 模型有"位置偏差"

通过阶段 1 的 bidirectional 实验确认: Erlangshen-Roberta-330M-NLI 对 `(premise, hypothesis)` 和 `(hypothesis, premise)` 的判断不对称。这意味着任何依赖"对称性"的设计都会失效。

---

## 九、安全注意事项

### 9.1 ⚠️ 原作者的 API key 在 git 历史里

`.git/` 历史的早期 commit 里有 6 个 .py 文件硬编码了**原作者的智谱 (ZhipuAI) API key**。
这个 key 在 git history 里没有被清理——上一手不是原作者，没有权限代他处理 key 泄露。

接手后想看具体是哪个 key，可以 `git log --all -p -- '*.py' | grep -i 'API_KEY'`。

**接手人需要做的**:
1. 联系原作者 (`Ls-trash` 是原作者的项目目录) 让他到 [智谱平台](https://www.bigmodel.cn/) 把这个 key **吊销**
2. 然后再决定是否 push 到公开仓库
3. 或者用 `git filter-repo` 把 key 从历史里抹掉 (破坏性，会改 commit hash)

### 9.2 ⚠️ HuggingFace token 不在仓库里

上一手用的 HF token 写在 `~/.cache/huggingface/token` (本机文件，**不在 git 里**)。
接手人需要**自己注册一个 HF 账号**，自己生成 token，自己配置。

不要把任何 HF token 写进代码或 commit。

### 9.3 ⚠️ NLI 模型不在仓库里

`local-erlangshen-330m-nli/` 目录是上一手下载到 `/data/ziwen/hf_cache/` 下的，**不在仓库里** (1.3 GB 不适合 git)。
`.gitignore` 已经排除了 `*.bin` / `*.safetensors`。
接手人需要按 §3.2 的步骤自己下载。

---

## 十、上一手的决策记录 (为什么是这样)

记录这些是为了让接手人理解"为什么不是另一种做法"，避免不必要的回退/争论。

### 决策 1: 抛弃原作者的 prompt + GLM-4-Flash 方案

**理由**: 原方案的根本错误是**问题建模错了**——把"硬冲突"和"软偏移"挤进同一个 LLM 端到端分类器，且 glm-4-flash 这个量级根本不能稳定做语义对比。详见 `重设计方案.md` 第二节。

### 决策 2: Track A 用 NLI 而不是 LLM judge

**理由**: NLI 是 NLP 二十年的成熟问题，有专门的判别模型和数据集。判别模型比生成式 LLM:
- 便宜两个数量级
- 不"偷懒"
- 决策可解释 (有 entail/neutral/contradict 三个明确概率)

### 决策 3: 用 Erlangshen-Roberta-330M-NLI 而不是其他 NLI 模型

**理由**:
- 中文原生 (vs 多语言模型)
- IDEA-CCNL 系列稳定，工业界用得多
- 110M 太小，可能力不够；3B 太大，CPU 跑不动

### 决策 4: 阶段 1 全部失败后接受 baseline 是最优

**理由**: 7 个变体证明了 NLI 模型存在 PR 曲线天花板。继续在输入/输出做文章是无效的。下一步必须改 NLI 模型本身 → fine-tune。

### 决策 5: 不删原作者的 git 历史

**理由**: 上一手不是原作者，无权代他处理 API key 泄露。这是接手人和原作者之间需要协调的事。

---

## 十一、给接手人的建议

1. **先读完 `重设计方案.md` 和两份 results report**，再写代码。这能省你 1-2 天的时间。
2. **不要尝试在 baseline 上做"输入侧过滤"或"输出侧验证"**——已经证明这条路死了。
3. **优先解决 GPU 问题**。即使是 1 张 V100 / 一卡时的 colab pro，也能让 fine-tune 从 4 小时缩到 20 分钟，质量也更好。
4. **fine-tune 之前先解决数据 split 问题**。否则你会在自污染的 test set 上得到一个看起来很高、实际上不可信的 F1。
5. **Track B 是论文的核心**——任务书要求的是"主题一致性"，不是"逻辑矛盾"。Track A 是基础设施，Track B 才是真正要写论文的地方。如果时间紧，先做 Track B，Track A 拿当前 baseline 的数字凑个章节就行。
6. **遇到不确定的设计决定时，回去读 `重设计方案.md`**。那份文档不是装饰，它解释了为什么很多看似"自然"的方案是错的。

---

最后，祝接手顺利。

— 上一手 (2026-04-07)
