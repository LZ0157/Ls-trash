import json
import time
from zhipuai import ZhipuAI
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random

# ================= 配置区 =================
API_KEY = "a44ef940edc248fbbfc0c29bcd6bebd0.cjCxPYv7I25vjw4m"
DATA_FILE = "cdconv.txt"
MODEL_NAME = "glm-4-flash"   # 想要更高准确率？改成 "glm-4-plus"
# ==========================================

client = ZhipuAI(api_key=API_KEY)

# 固定随机种子，保证每次抽样一致，结果可复现
random.seed(42)


# ================= 核心改进：分类型的 Few-shot 示例 =================
FEW_SHOT = """
【以下是判断示例，请学习后判断新对话】

示例1（无矛盾）：
用户: 你是哪里人？
智能体: 我是北京人，从小在胡同里长大。
用户: 北京有什么好吃的吗？
智能体: 北京烤鸭、炸酱面都很好吃，推荐你去试试。
→ 关键事实: 智能体是北京人。最后回复推荐北京美食，与身份一致。
结论: {{"thinking": "推荐北京美食与北京人身份一致，无矛盾", "predicted_label": 0}}

示例2（句内矛盾）：
用户: 你喜欢运动吗？
智能体: 我每天都坚持跑步锻炼。
用户: 那你最近在练什么？
智能体: 我最讨厌运动了，从来不去跑步。
→ 关键事实: 智能体之前说每天跑步。最后回复说自己讨厌运动从不跑步，前后自相矛盾。
结论: {{"thinking": "之前说每天跑步，现在说从不跑步，自身前后矛盾", "predicted_label": 1}}

示例3（角色混淆）：
用户: 请问你是医生还是护士？
智能体: 我是一名资深外科医生，从医二十年。
用户: 那帮我开个处方吧。
智能体: 我只是个护士，没有处方权哦。
→ 关键事实: 智能体之前自称是外科医生。最后回复说自己是护士，身份前后矛盾。
结论: {{"thinking": "先说自己是医生，后说自己是护士，角色身份矛盾", "predicted_label": 2}}

示例4（历史矛盾）：
用户: 你昨天去哪里了？
智能体: 我昨天一整天都在公司加班。
用户: 听说你昨天去旅游了？
智能体: 是啊，昨天去了趟西湖，风景特别好。
→ 关键事实: 智能体之前说昨天在公司加班。最后回复说昨天去了西湖旅游，时间线上的事实矛盾。
结论: {{"thinking": "先说昨天在公司加班，后说昨天去了西湖旅游，历史事实矛盾", "predicted_label": 3}}
"""


def build_prompt(context, persona=None):
    """改进：显式标注每轮角色(u1/b1/u2/b2)，让模型明确知道要对比谁"""
    u1 = context[0]["content"] if len(context) > 0 else ""
    b1 = context[1]["content"] if len(context) > 1 else ""
    u2 = context[2]["content"] if len(context) > 2 else ""
    b2 = context[3]["content"] if len(context) > 3 else ""

    prompt = f"""{FEW_SHOT}
请判断以下对话中智能体的最后一轮回复(b2)是否存在矛盾：

用户(u1): {u1}
智能体(b1): {b1}
用户(u2): {u2}
智能体(b2): {b2}

请先提取前文关键事实，再逐一对比b2，最后按以下标签输出：
0 = 无矛盾
1 = 句内矛盾（b2自身前后冲突）
2 = 角色混淆（b2与b1中已建立的身份/角色矛盾）
3 = 历史矛盾（b2与前文已确认的事实矛盾）

严格以JSON输出：
{{"thinking": "提取的关键事实和对比分析", "predicted_label": 0或1或2或3}}"""

    return prompt


def call_api_for_evaluation(prompt):
    """调用API，带系统角色设定"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的对话一致性检测系统。你必须仔细对比对话中的每一轮发言，找出所有类型的矛盾。不要偷懒，不要全部判为0。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.05,  # 降低随机性
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)

        thinking = result_json.get("thinking", "")
        label = int(result_json.get("predicted_label", 0))

        # 强制映射
        if label not in [0, 1, 2, 3]:
            label = 0

        return label, thinking

    except Exception as e:
        print(f"  ⚠️ API错误: {e}")
        return -1, f"API出错:{e}"  # ← 改成-1，和正常预测区分开


def load_cdconv_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                context = [
                    {"role": "user", "content": item.get("u1", "")},
                    {"role": "bot", "content": item.get("b1", "")},
                    {"role": "user", "content": item.get("u2", "")},
                    {"role": "bot", "content": item.get("b2", "")}
                ]
                label = int(item.get("label", 0))
                data.append({
                    "dialogue_id": f"{item.get('file', 'unknown')}_{item.get('method', 'unknown')}",
                    "context": context,
                    "label": label,
                    "persona": None
                })
            except Exception as e:
                continue
    return data


def run_evaluation():
    print(f"正在加载数据: {DATA_FILE}...")
    dataset = load_cdconv_data(DATA_FILE)
    if not dataset:
        print("数据集为空！")
        return

    print(f"原始数据: {len(dataset)} 条")

    # ================= 分层抽样 500 条 =================
    SAMPLE_SIZE = 500
    random.seed(42)  # 再次确认种子

    groups = {}
    for item in dataset:
        label = item["label"]
        if label not in groups:
            groups[label] = []
        groups[label].append(item)

    print("标签分布:")
    for label, items in sorted(groups.items()):
        print(f"  标签 {label}: {len(items)} 条")

    subset = []
    num_classes = len(groups)
    for label, items in groups.items():
        sample_count = min(len(items), max(1, SAMPLE_SIZE // num_classes))
        subset.extend(random.sample(items, sample_count))

    if len(subset) < SAMPLE_SIZE:
        remaining = [item for item in dataset if item not in subset]
        subset.extend(random.sample(remaining, min(len(remaining), SAMPLE_SIZE - len(subset))))

    random.shuffle(subset)
    dataset = subset[:SAMPLE_SIZE]
    print(f"\n均衡抽样: {len(dataset)} 条\n")

    true_labels = []
    pred_labels = []
    details = []
    api_error_count = 0  # ← 新增：记录API错误数

    for idx, item in enumerate(dataset):
        context = item["context"]
        true_label = int(item["label"])

        prompt = build_prompt(context)
        pred_label, reason = call_api_for_evaluation(prompt)

        # API失败的样本不计入评估
        if pred_label == -1:
            api_error_count += 1
            print(f"  [{idx + 1}/{len(dataset)}] ❌ API失败，跳过 ID: {item['dialogue_id']}")
            time.sleep(2)  # API失败后多等一会儿
            continue

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        details.append({
            "id": item.get("dialogue_id", idx),
            "true": true_label,
            "pred": pred_label,
            "reason": reason
        })

        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{len(dataset)}] 进度更新... (API错误累计: {api_error_count})")

        time.sleep(0.5)

    # ================= 报告 =================
    print("\n" + "=" * 60)
    print(f"评测结果报告 (模型: {MODEL_NAME})")
    print(f"有效样本: {len(true_labels)}/{SAMPLE_SIZE} (API失败: {api_error_count})")
    print("=" * 60)

    if len(true_labels) == 0:
        print("没有有效样本！请检查API连接。")
        return

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    true_binary = [0 if l == 0 else 1 for l in true_labels]
    pred_binary = [0 if l == 0 else 1 for l in pred_labels]

    acc_binary = accuracy_score(true_binary, pred_binary)
    f1_binary = f1_score(true_binary, pred_binary, average='binary')

    print(f"\n【二分类（0=无矛盾 vs 1=有矛盾）】")
    print(f"准确率: {acc_binary:.4f}")
    print(f"F1:     {f1_binary:.4f}")

    print(f"\n【四分类（0/1/2/3）】")
    print(f"准确率:   {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n各类别:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["无矛盾(0)", "句内矛盾(1)", "角色混淆(2)", "历史矛盾(3)"],
        zero_division=0
    ))

    # 预测分布统计
    from collections import Counter
    pred_dist = Counter(pred_labels)
    print(f"预测分布: {dict(sorted(pred_dist.items()))}")

    with open("eval_results_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至 eval_results_details.json")


if __name__ == "__main__":
    run_evaluation()
