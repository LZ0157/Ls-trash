"""CDConv 数据加载与分层抽样。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict


def load_cdconv(path: str | Path) -> List[Dict]:
    """加载 CDConv JSONL 文件，返回标准化对话列表。

    每条对话字段:
        id     : str   — file_method_model 拼接的标识
        u1/b1  : str   — 第 1 轮用户/机器人
        u2/b2  : str   — 第 2 轮用户/机器人
        label  : int   — 0=无矛盾, 1=句内, 2=角色混淆, 3=历史矛盾
    """
    data: List[Dict] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            data.append({
                "id": f"{item.get('file', '?')}_{item.get('method', '?')}_{item.get('model', '?')}",
                "u1": item.get("u1", ""),
                "b1": item.get("b1", ""),
                "u2": item.get("u2", ""),
                "b2": item.get("b2", ""),
                "label": int(item.get("label", 0)),
            })
    if skipped:
        print(f"[data] 跳过 {skipped} 行无法解析的样本")
    return data


def label_distribution(data: List[Dict]) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for item in data:
        dist[item["label"]] = dist.get(item["label"], 0) + 1
    return dist


def stratified_sample(
    data: List[Dict],
    n_per_class: int,
    seed: int = 42,
) -> List[Dict]:
    """按 label 分组等量抽样，返回打乱后的列表。"""
    rng = random.Random(seed)
    groups: Dict[int, List[Dict]] = {}
    for item in data:
        groups.setdefault(item["label"], []).append(item)

    sampled: List[Dict] = []
    for label in sorted(groups.keys()):
        items = groups[label]
        k = min(n_per_class, len(items))
        sampled.extend(rng.sample(items, k))
    rng.shuffle(sampled)
    return sampled
