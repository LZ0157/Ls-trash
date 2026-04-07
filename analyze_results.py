import json
from collections import Counter

with open("eval_results_details.json", "r", encoding="utf-8") as f:
    details = json.load(f)

# 1. API 错误统计
api_errors = sum(1 for d in details if "API调用失败" in d.get("reason", "") or "解析失败" in d.get("reason", ""))
print(f"API调用失败次数: {api_errors}/{len(details)}")
if api_errors > 20:
    print("⚠️ API错误过多！大量样本被默认判为label=0，严重拖低准确率")

# 2. 预测分布
pred_dist = Counter(d["pred"] for d in details)
print(f"\n模型预测分布: {dict(sorted(pred_dist.items()))}")
if 2 not in pred_dist and 3 not in pred_dist:
    print("⚠️ 验证：模型从未预测过标签2和3！")

# 3. 各标签被错误判为0的比例
for label in [1, 2, 3]:
    samples = [d for d in details if d["true"] == label]
    wrong_as_0 = sum(1 for d in samples if d["pred"] == 0)
    print(f"\n标签{label} → 被错误预测为0的比例: {wrong_as_0}/{len(samples)} = {wrong_as_0/len(samples):.1%}")

# 4. 展示几条label=2/3但预测为0的具体案例
print("\n" + "=" * 60)
print("【label=2或3但被误判为0的典型案例】")
for d in details:
    if d["true"] in [2, 3] and d["pred"] == 0:
        print(f"  ID={d['id']} | 真实={d['true']} 预测={d['pred']}")
        print(f"  理由: {d['reason'][:80]}...")
        break  # 只看一条
