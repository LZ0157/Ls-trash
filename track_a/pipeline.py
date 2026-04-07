"""Track A pipeline: 句子切分 → 配对 → NLI → 分类后处理。

支持的开关:
    fine_split      : 是否按逗号细切
    filter_claims   : 是否丢掉疑问句/填充词
    bidirectional   : NLI 双向校验 — 必须 NLI(p,h) 和 NLI(h,p) 都判矛盾才算
    mode            : "sentence" (按句切+配对) 或 "document" (整段对整段)

CDConv 标签:
    0 = 无矛盾
    1 = 句内矛盾  (b2 内部自相矛盾)
    2 = 角色混淆  (无 persona 字段, 本基线无法可靠区分 2 vs 3)
    3 = 历史矛盾  (b2 与 b1 已建立的事实冲突)
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Literal

from .nli import NLIJudge
from .segmenter import split_sentences


PipelineMode = Literal["sentence", "document"]


class TrackAPipeline:
    """基于 NLI 的硬冲突检测器。"""

    def __init__(
        self,
        nli: NLIJudge,
        contradict_threshold: float = 0.5,
        fine_split: bool = False,
        filter_claims: bool = False,
        bidirectional: bool = False,
        mode: PipelineMode = "sentence",
        batch_size: int = 8,
    ) -> None:
        self.nli = nli
        self.threshold = contradict_threshold
        self.fine_split = fine_split
        self.filter_claims = filter_claims
        self.bidirectional = bidirectional
        self.mode = mode
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def detect(self, dialogue: Dict) -> Dict:
        if self.mode == "document":
            return self._detect_document(dialogue)
        return self._detect_sentence(dialogue)

    # ------------------------------------------------------------------
    # 模式 1: 整段对整段
    # ------------------------------------------------------------------
    def _detect_document(self, dialogue: Dict) -> Dict:
        b1 = (dialogue.get("b1", "") or "").strip()
        b2 = (dialogue.get("b2", "") or "").strip()
        if not b1 or not b2:
            return {"predicted_label": 0, "conflicts": [], "n_pairs_checked": 0}

        if self.bidirectional:
            results = self.nli.judge_batch([(b1, b2), (b2, b1)], batch_size=2)
            fwd, rev = results[0], results[1]
            is_contradict = (
                fwd["contradict"] >= self.threshold
                and rev["contradict"] >= self.threshold
            )
            p_contradict = min(fwd["contradict"], rev["contradict"])
            n_pairs = 2
        else:
            fwd = self.nli.judge(b1, b2)
            is_contradict = fwd["contradict"] >= self.threshold
            p_contradict = fwd["contradict"]
            rev = None
            n_pairs = 1

        if is_contradict:
            conflict = {
                "premise": b1,
                "hypothesis": b2,
                "source": "doc_b1_b2",
                "p_contradict": p_contradict,
                "p_entail_fwd": fwd["entail"],
                "p_neutral_fwd": fwd["neutral"],
            }
            if rev is not None:
                conflict["p_contradict_rev"] = rev["contradict"]
            return {
                "predicted_label": 3,  # doc 模式无法区分 1 vs 3
                "conflicts": [conflict],
                "n_pairs_checked": n_pairs,
            }
        return {"predicted_label": 0, "conflicts": [], "n_pairs_checked": n_pairs}

    # ------------------------------------------------------------------
    # 模式 2: 按句切 + 配对
    # ------------------------------------------------------------------
    def _detect_sentence(self, dialogue: Dict) -> Dict:
        b1 = dialogue.get("b1", "")
        b2 = dialogue.get("b2", "")

        b1_sents = split_sentences(b1, fine=self.fine_split, filter_claims=self.filter_claims)
        b2_sents = split_sentences(b2, fine=self.fine_split, filter_claims=self.filter_claims)

        # 1. 准备句对
        intra_pairs: List[Tuple[str, str]] = []
        cross_pairs: List[Tuple[str, str]] = []
        for i in range(len(b2_sents)):
            for j in range(i + 1, len(b2_sents)):
                intra_pairs.append((b2_sents[i], b2_sents[j]))
        for s1 in b1_sents:
            for s2 in b2_sents:
                cross_pairs.append((s1, s2))

        forward_pairs = intra_pairs + cross_pairs
        sources = ["intra_b2"] * len(intra_pairs) + ["cross_b1_b2"] * len(cross_pairs)

        if not forward_pairs:
            return {"predicted_label": 0, "conflicts": [], "n_pairs_checked": 0}

        # 2. NLI 推理 (单向 / 双向)
        fwd_results = self.nli.judge_batch(forward_pairs, batch_size=self.batch_size)

        if self.bidirectional:
            reverse_pairs = [(h, p) for p, h in forward_pairs]
            rev_results = self.nli.judge_batch(reverse_pairs, batch_size=self.batch_size)
            n_pairs_checked = len(forward_pairs) * 2
        else:
            rev_results = [None] * len(forward_pairs)  # type: ignore[list-item]
            n_pairs_checked = len(forward_pairs)

        # 3. 收集冲突
        conflicts: List[Dict] = []
        for (premise, hypothesis), src, fr, rr in zip(forward_pairs, sources, fwd_results, rev_results):
            fwd_ok = fr["contradict"] >= self.threshold
            if not fwd_ok:
                continue
            if self.bidirectional:
                if rr["contradict"] < self.threshold:  # type: ignore[index]
                    continue
                p_contradict = min(fr["contradict"], rr["contradict"])  # type: ignore[index]
                conflict = {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "source": src,
                    "p_contradict": p_contradict,
                    "p_contradict_fwd": fr["contradict"],
                    "p_contradict_rev": rr["contradict"],  # type: ignore[index]
                    "p_entail_fwd": fr["entail"],
                    "p_neutral_fwd": fr["neutral"],
                }
            else:
                conflict = {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "source": src,
                    "p_contradict": fr["contradict"],
                    "p_entail": fr["entail"],
                    "p_neutral": fr["neutral"],
                }
            conflicts.append(conflict)

        # 4. 后处理: 没冲突 → 0; 有冲突 → 按最强一个的来源决定
        if not conflicts:
            return {
                "predicted_label": 0,
                "conflicts": [],
                "n_pairs_checked": n_pairs_checked,
            }

        conflicts.sort(key=lambda c: -c["p_contradict"])
        top = conflicts[0]
        predicted = 1 if top["source"] == "intra_b2" else 3

        return {
            "predicted_label": predicted,
            "conflicts": conflicts,
            "n_pairs_checked": n_pairs_checked,
        }
