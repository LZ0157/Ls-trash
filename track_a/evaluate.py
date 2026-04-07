"""Track A 评测脚本。

用法:
    cd Ls-trash
    python -m track_a.evaluate --n_per_class 5    # smoke test
    python -m track_a.evaluate --n_per_class 50   # 200 条正式评测
"""

from __future__ import annotations

# === 严格 CPU 强制 ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# =====================

import argparse
import json
import time
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, classification_report

from .data import load_cdconv, label_distribution, stratified_sample
from .nli import NLIJudge, DEFAULT_MODEL, DEFAULT_CACHE
from .pipeline import TrackAPipeline


def run(
    data_path: str,
    n_per_class: int,
    output_path: str,
    threshold: float = 0.5,
    fine_split: bool = False,
    filter_claims: bool = False,
    bidirectional: bool = False,
    mode: str = "sentence",
    batch_size: int = 8,
    model_id: str = DEFAULT_MODEL,
    cache_dir: str = DEFAULT_CACHE,
    seed: int = 42,
) -> None:
    print(f"=" * 60)
    print(f"Track A — CDConv 评测")
    print(f"=" * 60)

    # 1. 加载并抽样
    print(f"\n[1/4] Loading data: {data_path}")
    data = load_cdconv(data_path)
    print(f"      total: {len(data)}")
    print(f"      原始分布: {dict(sorted(label_distribution(data).items()))}")

    sampled = stratified_sample(data, n_per_class=n_per_class, seed=seed)
    print(f"      抽样: {len(sampled)} (每类目标 {n_per_class})")
    print(f"      抽样分布: {dict(sorted(label_distribution(sampled).items()))}")

    # 2. 加载 NLI 模型
    print(f"\n[2/4] Loading NLI model")
    nli = NLIJudge(model_id=model_id, cache_dir=cache_dir)

    # 3. 跑 pipeline
    print(
        f"\n[3/4] Running pipeline "
        f"(mode={mode}, threshold={threshold}, fine_split={fine_split}, "
        f"filter_claims={filter_claims}, bidirectional={bidirectional})"
    )
    pipeline = TrackAPipeline(
        nli=nli,
        contradict_threshold=threshold,
        fine_split=fine_split,
        filter_claims=filter_claims,
        bidirectional=bidirectional,
        mode=mode,
        batch_size=batch_size,
    )

    true_labels: list = []
    pred_labels: list = []
    details: list = []
    t0 = time.time()
    total_pairs = 0

    for i, item in enumerate(sampled):
        result = pipeline.detect(item)
        true_labels.append(item["label"])
        pred_labels.append(result["predicted_label"])
        total_pairs += result["n_pairs_checked"]
        details.append({
            "id": item["id"],
            "u1": item["u1"],
            "b1": item["b1"],
            "u2": item["u2"],
            "b2": item["b2"],
            "true": item["label"],
            "pred": result["predicted_label"],
            "n_pairs": result["n_pairs_checked"],
            "conflicts": result["conflicts"][:3],  # 只存前 3 个最强冲突
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(sampled):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"      [{i + 1:>3}/{len(sampled)}]  elapsed {elapsed:.0f}s  rate {rate:.2f} sample/s")

    elapsed = time.time() - t0
    print(f"      done in {elapsed:.0f}s ({total_pairs} NLI pairs)")

    # 4. 计算指标
    print(f"\n[4/4] Metrics")

    # 用于二分类的合并标签 (任何 1/2/3 都视为有矛盾)
    true_bin = [0 if l == 0 else 1 for l in true_labels]
    pred_bin = [0 if l == 0 else 1 for l in pred_labels]

    acc_bin = accuracy_score(true_bin, pred_bin)
    f1_bin = f1_score(true_bin, pred_bin, average="binary")
    print(f"\n  【二分类: 0=无矛盾 vs 1=有矛盾】")
    print(f"    accuracy: {acc_bin:.4f}")
    print(f"    F1:       {f1_bin:.4f}")

    # 四分类指标 (注意 pipeline 永远不会输出 2，会导致 2 类 0 分)
    acc = accuracy_score(true_labels, pred_labels)
    f1m = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    print(f"\n  【四分类: 0/1/2/3】")
    print(f"    accuracy:  {acc:.4f}")
    print(f"    macro F1:  {f1m:.4f}")

    print(f"\n  分类报告:")
    print(classification_report(
        true_labels, pred_labels,
        labels=[0, 1, 2, 3],
        target_names=["无矛盾(0)", "句内(1)", "角色混淆(2)", "历史(3)"],
        zero_division=0,
    ))

    pred_dist = dict(sorted(Counter(pred_labels).items()))
    print(f"  预测分布: {pred_dist}")

    # 把 2 和 3 在真实标签里合并成"跨轮矛盾"看待 → 3 类评测
    true_3cls = [3 if l == 2 else l for l in true_labels]
    acc3 = accuracy_score(true_3cls, pred_labels)
    f1m3 = f1_score(true_3cls, pred_labels, average="macro", zero_division=0)
    print(f"\n  【3 类合并 (label 2 → 3)】")
    print(f"    accuracy:  {acc3:.4f}")
    print(f"    macro F1:  {f1m3:.4f}")

    # 5. 保存
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model_id": model_id,
            "mode": mode,
            "threshold": threshold,
            "fine_split": fine_split,
            "filter_claims": filter_claims,
            "bidirectional": bidirectional,
            "batch_size": batch_size,
            "n_per_class": n_per_class,
            "seed": seed,
            "n_samples": len(sampled),
            "elapsed_seconds": round(elapsed, 1),
            "total_nli_pairs": total_pairs,
        },
        "metrics": {
            "binary": {"accuracy": acc_bin, "f1": f1_bin},
            "four_class": {"accuracy": acc, "macro_f1": f1m},
            "three_class_merged": {"accuracy": acc3, "macro_f1": f1m3},
            "pred_distribution": pred_dist,
        },
        "details": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  → 详细结果保存至 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A NLI baseline on CDConv")
    parser.add_argument("--data", default="cdconv.txt", help="CDConv jsonl path")
    parser.add_argument("--n_per_class", type=int, default=50, help="每类抽样数")
    parser.add_argument("--output", default="results/track_a_eval.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fine_split", action="store_true", help="按逗号细切")
    parser.add_argument("--filter_claims", action="store_true", help="过滤疑问句和填充词")
    parser.add_argument("--bidirectional", action="store_true", help="双向 NLI 校验")
    parser.add_argument("--mode", choices=["sentence", "document"], default="sentence")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        data_path=args.data,
        n_per_class=args.n_per_class,
        output_path=args.output,
        threshold=args.threshold,
        fine_split=args.fine_split,
        filter_claims=args.filter_claims,
        bidirectional=args.bidirectional,
        mode=args.mode,
        batch_size=args.batch_size,
        model_id=args.model,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
