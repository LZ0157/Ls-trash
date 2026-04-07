# Track A 阶段 1 改进报告

> 在 NLI baseline (binary F1 = 0.7456) 之上做三类改进尝试: 候选句对过滤、双向 NLI 校验、document-level NLI baseline。
> **结论先行: 所有变体的 F1 都低于 baseline**。但分析揭示了清晰的"precision/recall 折衷曲线"，对下一步方向有指导意义。

## 一、所有变体的完整对比 (200 条 50/类)

| # | 变体 | 预测正 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0 | **Baseline** (fine_split only) | 137 | 107 | 30 | 43 | **0.781** | **0.713** | **0.7456** |
| 1 | Filter v1 (整段丢弃 ?-句) | 90 | 75 | 15 | 75 | 0.833 | 0.500 | 0.625 |
| 2 | **Filter v2** (refined: per-clause) | 109 | 90 | 19 | 60 | 0.826 | **0.600** | **0.695** |
| 3 | Bidirectional (threshold=0.5) | 90 | 75 | 15 | 75 | 0.833 | 0.500 | 0.625 |
| 4 | Bidirectional (threshold=0.3) | 103 | 83 | 20 | 67 | 0.806 | 0.553 | 0.656 |
| 5 | Filter v2 + Bidir | 69 | 59 | 10 | 91 | **0.855** | 0.393 | 0.539 |
| 6 | Document mode | 80 | 65 | 15 | 85 | 0.813 | 0.433 | 0.565 |
| 7 | Document + Bidir | 45 | 39 | 6 | 111 | 0.867 | 0.260 | 0.400 |

**裸眼可见的规律**:
- 所有改进都让 precision **+0.03 到 +0.09**
- 但都让 recall **−0.11 到 −0.45**
- F1 一律下降

## 二、为什么 filter 第一版 (v1) 失败

第一版 filter 把整段以 `?` 结尾的句子整体丢掉。但 CDConv 的对话有大量这种模式:

```
b1: "我今年24了，你呢？"          ← 真陈述 + 礼貌反问
b2: "我今年25了，你猜我多大？"     ← 真陈述 + 礼貌反问
```

第一版 filter 把 b1 整体丢掉 → 0 个候选句对 → 漏报 32 个 TP。

## 三、Filter v2 (refined) 的修复

按逗号细切**之后**才判断每个子句是否是问句：
- "你呢" / "几岁了" / "是吗" 这些**短问句子句**单独丢
- "我今年24了" 这种陈述子句**保留**

实现要点 (`segmenter.py`):
1. 切句时记录每段的结束标点
2. 每段按逗号细切
3. 只有**最后一个子句**继承原句的 `?` 标记
4. `is_claim()` 还检查"短句包含疑问粒子" (吗/呢/什么/哪/多少/几/谁)，长度 < 8 时视为问句

效果: F1 从 v1 的 0.625 → v2 的 0.695，recall 从 0.50 → 0.60。**有改善但仍未追上 baseline**。

## 四、为什么 bidirectional 失败

理论上"NLI(p, h) 和 NLI(h, p) 都判矛盾"应该能砍掉词面对立的伪矛盾，因为对称的对立才是真对立。

但实测：bidirectional 砍掉了 32 个真 TP，多数是 Erlangshen 在反方向上对真矛盾不敏感的 case。

我又试了 **threshold=0.3** 的"软"双向 (变体 4)，让两个方向都只要 0.3 就算命中——recall 从 0.50 回到 0.55，仍然不如 baseline。

**结论**: Erlangshen 在 CMNLI 训练数据上有**位置偏差**。对真矛盾的判断不对称。bidirectional 这条路在这个 NLI 模型上行不通，**除非换更大、训练更对称的 NLI 模型**。

## 五、为什么 document mode 失败

把整段 b1 和整段 b2 直接送 NLI:
- F1 = 0.565，比 baseline 低 0.18
- recall 仅 0.43

原因: NLI 模型训练数据是**短句对**（CMNLI/SNLI 都是单句），长输入分散注意力，对局部矛盾不敏感。
另外: `max_length=256` 截断也吞掉了部分 b2 的尾部内容。

**这条路死了**——除非用专门的 long-form NLI 模型 (没有现成中文版本)。

## 六、Precision-Recall 折衷曲线

把所有变体画在 PR 平面上：

```
recall ▲
  0.71 ┤●  baseline (best F1)
  0.70 ┤
  0.65 ┤
  0.60 ┤  ●  filter v2
  0.55 ┤  ●  bidir t=0.3
  0.50 ┤  ●  filter v1 / bidir
  0.45 ┤
  0.43 ┤  ●  document
  0.39 ┤  ●  filter+bidir
  0.30 ┤
  0.26 ┤  ●  doc+bidir
       └──┬────┬────┬────┬────┬─→
        0.78  0.81  0.83  0.85  0.87
                              precision
```

**所有变体都在一条同向曲线上滑动**——任何一个尝试都在用 recall 换 precision，且换得不划算 (相同 precision 处 baseline 的 recall 高出 0.10+)。

## 七、根本原因诊断

阶段 1 的所有改进都是**输入侧过滤**或**输出侧验证**。它们都假设"NLI 模型是对的，错的是给它喂的句对"。

但 PR 曲线说的是另一回事: **NLI 模型本身的判别能力就在 0.78 precision / 0.71 recall 这条曲线上**。无论你怎么过滤输入或验证输出，都跳不出这条线——因为模型对"模糊矛盾"的判断本来就不准。

要继续往上推 F1，必须改的是 NLI 模型对"接近矛盾"的 cases 的判别能力本身：
1. **fine-tune Erlangshen on CDConv** — 学会区分"话题切换 vs 真矛盾" → **需要 GPU，被禁**
2. **换更大的 NLI 模型** (e.g. mDeBERTa-large-xnli) — 更大模型 PR 曲线会整体外推
3. **集成多个 NLI 模型** 投票 — 统计上能提一点，但需要下载多个模型
4. **加 LLM 做二次判别** (用本地 0.5B Qwen 给 NLI 不确定的 case 复审) — CPU 慢，但可能有效

## 八、阶段 1 的副产品

虽然 F1 没提升，但拿到了几个有用的产物:

1. **`filter_claims=True` 是 precision-priority 模式的最佳选择** —— 当用例需要"被标的都是真矛盾，宁可漏几个"，应该用 filter v2 (precision 0.83)
2. **PR 曲线的存在** —— 给后续工作提供了一个明确的"baseline 优于 stage 1 所有改动"的对照基准
3. **Document mode 已被排除** —— 不用再尝试这条路
4. **Bidirectional 已被排除** —— 在 Erlangshen 上不可行

## 九、产物清单

| 文件 | 描述 |
|---|---|
| `results/track_a_eval_200.json` | Baseline (F1=0.7456) |
| `results/track_a_v_filter.json` | Filter v1 (aggressive) |
| `results/track_a_v_filter2.json` | **Filter v2 (refined, F1=0.695)** |
| `results/track_a_v_bidir.json` | Bidirectional t=0.5 |
| `results/track_a_v_bidir_t03.json` | Bidirectional t=0.3 |
| `results/track_a_v_filter_bidir.json` | Filter v1 + Bidir |
| `results/track_a_v_filter_bidir2.json` | Filter v2 + Bidir |
| `results/track_a_v_doc.json` | Document mode |
| `results/track_a_v_doc_bidir.json` | Document + Bidir |

## 十、阶段 2 的可选方向

按"是否能突破 PR 曲线"分类:

### 🟢 可能突破曲线（值得做）
- **A. 实词重叠门控** (entity overlap) — 这是**架构性**改动，不只是 filter；它把"完全无关的话题对"剔除在 NLI 输入之外，应该能改善 baseline 的 30 个 FP 中的话题切换类 (~12 个)，**且不太可能伤 recall** (因为真矛盾几乎都共享话题)
- **B. b1 隐式 persona 解锁 class 2** — 4 分类 macro F1 从 0.23 拉高的唯一可行路径
- **C. LLM 二次判别** — CPU 上 Qwen2.5-0.5B 给 NLI 不确定的 case (contradict 在 0.3-0.7) 复审，可能有效但慢

### 🟡 在曲线上滑动（不值得做）
- 各种 filter / threshold 组合 — 已穷尽，都在同一条曲线上

### 🔴 阶段 1 已淘汰
- Bidirectional NLI
- Document mode
- Aggressive filter v1

## 十一、给你的两个问题

1. **是否同意阶段 1 的负面结论**？也就是: 接受 baseline 是当前架构下的最优解，不再纠结 filter / bidir / doc 这些路。
2. **阶段 2 走 A (实词重叠门控) 还是 B (class 2 解锁) 还是 C (LLM 二次判别)**？
   - 我推荐先做 A：架构改动小，最有可能突破 PR 曲线，能帮你拿到第一个真的更好的 F1。
   - 如果想要**论文亮点**，B 是必做的——把 class 2 从 0.00 拉到任何 > 0 的数字都是一个可写的贡献。
   - C 比较实验性，做完后即使没提升，也能告诉我们"NLI 是否就是上限"。
