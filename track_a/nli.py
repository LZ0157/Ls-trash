"""NLI 判别器封装 (CPU only)。

载入中文 NLI 模型（默认 IDEA-CCNL/Erlangshen-Roberta-330M-NLI），对
(premise, hypothesis) 对输出 entail / neutral / contradiction 概率。

强制 CPU 推理: 在 import torch 之前清空 CUDA_VISIBLE_DEVICES。
"""

from __future__ import annotations

# === 严格 CPU 强制 ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# =====================

from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_MODEL = "/data/ziwen/hf_cache/local-erlangshen-330m-nli"
DEFAULT_CACHE = "/data/ziwen/hf_cache"  # 仅在 model_id 是 hub repo 时使用


class NLIJudge:
    """中文 NLI 判别器。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        cache_dir: str = DEFAULT_CACHE,
        max_length: int = 256,
    ) -> None:
        # 再次确认 CUDA 不可见
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", \
            "CUDA_VISIBLE_DEVICES must be empty before loading the model"
        assert not torch.cuda.is_available(), \
            "CUDA must not be available — refusing to load model"

        self.device = torch.device("cpu")
        self.max_length = max_length
        self.model_id = model_id

        print(f"[nli] Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"[nli] Loading model on CPU ...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()

        # 解析 label 映射
        id2label = self.model.config.id2label
        self.id2label: Dict[int, str] = {int(k): v for k, v in id2label.items()}
        self.contradiction_idx = self._find_label_idx(["contradict", "矛盾", "contradiction"])
        self.entail_idx = self._find_label_idx(["entail", "蕴含", "entailment"])
        self.neutral_idx = self._find_label_idx(["neutral", "中立", "中性"])
        print(f"[nli] Label map: {self.id2label}")
        print(f"[nli] entail={self.entail_idx} neutral={self.neutral_idx} contradict={self.contradiction_idx}")

    def _find_label_idx(self, keywords: List[str]) -> int:
        """根据关键词在 id2label 里找对应的下标，找不到时回退顺序索引。"""
        for idx, label in self.id2label.items():
            l = str(label).lower()
            for kw in keywords:
                if kw.lower() in l:
                    return idx
        # Fallback: assume convention 0=entail, 1=neutral, 2=contradiction
        # but we need to know what kw was queried; just return -1 to error out loud
        raise ValueError(f"Could not find any of {keywords} in label map: {self.id2label}")

    @torch.no_grad()
    def judge(self, premise: str, hypothesis: str) -> Dict:
        """单对推理。"""
        return self.judge_batch([(premise, hypothesis)], batch_size=1)[0]

    @torch.no_grad()
    def judge_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """批量推理。每条返回 {entail, neutral, contradict, argmax_label}。"""
        results: List[Dict] = []
        if not pairs:
            return results
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            premises = [p[0] if p[0] else " " for p in chunk]
            hypotheses = [p[1] if p[1] else " " for p in chunk]
            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            argmax = torch.argmax(logits, dim=-1).cpu().tolist()
            for prob, am in zip(probs, argmax):
                results.append({
                    "entail": prob[self.entail_idx],
                    "neutral": prob[self.neutral_idx],
                    "contradict": prob[self.contradiction_idx],
                    "argmax_label": self.id2label[int(am)],
                })
        return results


if __name__ == "__main__":
    judge = NLIJudge()
    pairs = [
        ("我每天都坚持跑步锻炼。", "我最讨厌运动了，从来不去跑步。"),  # contradict
        ("我是北京人，从小在胡同里长大。", "我喜欢吃北京烤鸭。"),       # neutral / entail
        ("天气很好。", "今天阳光明媚。"),                                 # entail
    ]
    out = judge.judge_batch(pairs)
    for (p, h), r in zip(pairs, out):
        print(f"P: {p}")
        print(f"H: {h}")
        print(f"  → entail={r['entail']:.3f} neutral={r['neutral']:.3f} contradict={r['contradict']:.3f}  ({r['argmax_label']})")
        print()
