import json
import time
from zhipuai import ZhipuAI
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ================= 配置区 =================
API_KEY = "a44ef940edc248fbbfc0c29bcd6bebd0.cjCxPYv7I25vjw4m"  # 替换为你的 API Key
DATA_FILE = "cdconv.txt"  # 数据文件路径（如果是真实数据，改成真实的路径如 cdconv.txt）
MODEL_NAME = "glm-4-flash"  # 使用 flash 跑测试集便宜且快，最终发论文可换成 glm-4-plus
# ==========================================

client = ZhipuAI(api_key=API_KEY)


def load_cdconv_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            # 按照 CDConv 原始格式解析
            dialogue_id = parts[0]
            label = int(parts[5])
            persona = parts[6].split("|") if len(parts) > 6 else None

            # 拼接成我们需要的 context 格式
            context = [
                {"role": "user", "content": parts[1]},
                {"role": "bot", "content": parts[2]},
                {"role": "user", "content": parts[3]},
                {"role": "bot", "content": parts[4]}
            ]

            data.append({
                "dialogue_id": dialogue_id,
                "context": context,
                "label": label,
                "persona": persona
            })
    return data


def build_prompt(context, persona=None):
    """构建评估 Prompt，体现你基于文献设计 Prompt 的工作量"""

    # 将对话历史拼接成字符串
    dialogue_str = ""
    for turn in context:
        role = "用户" if turn["role"] == "user" else "智能体"
        dialogue_str += f"{role}: {turn['content']}\n"

    persona_str = ""
    if persona and len(persona) > 0:
        persona_str = f"【智能体预设人设/背景】\n{'；'.join(persona)}\n\n"

    prompt = f"""你是一个专业的对话一致性检测专家。请判断以下多轮对话中，智能体的最后一轮回复是否存在不一致或矛盾。

{persona_str}【对话记录】
{dialogue_str}

【不一致类型定义】
0: 无矛盾（回复逻辑自洽，与人设和历史均一致）
1: 句内矛盾（回复本身存在事实或逻辑冲突）
2: 角色混淆（回复与人设、身份设定产生冲突）
3: 历史矛盾（回复与之前的对话内容产生冲突）

【输出要求】
请严格以 JSON 格式输出，仅包含以下字段：
{{
    "predicted_label": 0,
    "reason": "用一句话简短说明判断理由。"
}}
"""
    return prompt


def call_api_for_evaluation(prompt):
    """调用 API 并解析结果，包含容错机制"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 保持客观
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)

        # 提取 label 并确保是整数
        label = int(result_json.get("predicted_label", 0))
        # 强制将异常值映射到 0-3 (大模型偶尔会抽风输出 4 或 -1)
        if label not in [0, 1, 2, 3]:
            label = 0
        reason = result_json.get("reason", "")

        return label, reason

    except Exception as e:
        print(f"API 调用或解析出错: {e}")
        return 0, "解析失败，默认判为无矛盾"


def run_evaluation():
    print(f"正在加载数据: {DATA_FILE}...")
    dataset = load_cdconv_data(DATA_FILE)
    if not dataset:
        print("数据集为空或格式解析失败，请检查文件！")
        return

    print(f"数据加载成功，共 {len(dataset)} 条样本。开始调用 API 评测...")

    true_labels = []
    pred_labels = []
    details = []

    for idx, item in enumerate(dataset):
        context = item["context"]
        true_label = int(item["label"])
        persona = item.get("persona", None)  # 兼容没有 persona 字段的数据

        prompt = build_prompt(context, persona)
        pred_label, reason = call_api_for_evaluation(prompt)

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        details.append({
            "id": item.get("dialogue_id", idx),
            "true": true_label,
            "pred": pred_label,
            "reason": reason
        })

        # 打印进度
        print(
            f"[{idx + 1}/{len(dataset)}] ID: {details[-1]['id']} | 真实: {true_label} | 预测: {pred_label} | 理由: {reason}")

        # 智谱 API 有频率限制，加个小延时防止报错
        time.sleep(0.5)

        # ================= 计算指标 (毕设论文的核心素材) =================
    print("\n" + "=" * 50)
    print("评测结果报告")
    print("=" * 50)

    acc = accuracy_score(true_labels, pred_labels)
    # 计算 Macro F1 (多分类任务看这个指标最严谨)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f"总体准确率: {acc:.4f}")
    print(f"Macro F1 分数: {macro_f1:.4f}")
    print("\n各类别详细指标:")
    # target_names 必须和你的 label 0,1,2,3 一一对应
    print(classification_report(true_labels, pred_labels,
                                target_names=["无矛盾(0)", "句内矛盾(1)", "角色混淆(2)", "历史矛盾(3)"]))

    # 保存详细结果，方便你写论文时分析 Bad Case
    with open("draftpaper2.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print("\n详细预测结果已保存至 eval_results_details.json")


if __name__ == "__main__":
    run_evaluation()
