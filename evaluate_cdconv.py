import json
import time
from zhipuai import ZhipuAI
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
# ================= 配置区 =================
API_KEY = "a44ef940edc248fbbfc0c29bcd6bebd0.cjCxPYv7I25vjw4m"  # 替换为你的 API Key
DATA_FILE = "cdconv.txt"  # 数据文件路径（如果是真实数据，改成真实的路径如 cdconv.txt）
MODEL_NAME = "glm-4-flash"  # 使用 flash 跑测试集便宜且快，最终发论文可换成 glm-4-plus
# ==========================================

client = ZhipuAI(api_key=API_KEY)

def load_cdconv_data(file_path):
    """
    专门针对你下载的 JSON Lines 格式的 cdconv.txt 进行解析
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # 1. 按每一行的 JSON 格式解析
                item = json.loads(line)

                # 2. 将 u1, b1, u2, b2 映射成我们系统需要的 context 格式
                context = [
                    {"role": "user", "content": item.get("u1", "")},
                    {"role": "bot", "content": item.get("b1", "")},
                    {"role": "user", "content": item.get("u2", "")},
                    {"role": "bot", "content": item.get("b2", "")}
                ]

                # 3. 提取标签 (注意：确保这里的 label 是 0,1,2,3 的整数)
                # 你的数据里 label 是 0，通常 0 代表无矛盾，其他代表不同类型矛盾
                label = int(item.get("label", 0))

                # 4. 组装成标准结构 (因为原数据没有显式的 persona 字段，这里设为 None)
                data.append({
                    "dialogue_id": f"{item.get('file', 'unknown')}_{item.get('method', 'unknown')}",
                    "context": context,
                    "label": label,
                    "persona": None
                })

            except json.JSONDecodeError as e:
                print(f"解析单行 JSON 失败，跳过。错误: {e}")
                continue
            except Exception as e:
                print(f"处理数据时发生未知错误，跳过。错误: {e}")
                continue

    return data


def build_prompt(context, persona=None):
    dialogue_str = ""
    for turn in context:
        role = "用户" if turn["role"] == "user" else "智能体"
        dialogue_str += f"{role}: {turn['content']}\n"

    prompt = f"""你是一个极其严格、专挑毛病的对话逻辑质检专家。你的任务是判断智能体的最后一轮回复是否存在逻辑漏洞或矛盾。

【对话记录】
{dialogue_str}

【评判标准】
0: 无矛盾（前后逻辑完全自洽）
1: 有矛盾（包括：句内自我打脸、前后说法冲突、违背常理或隐含人设）

【评判示例】
示例1：
用户: 你喜欢狗吗？
智能体: 喜欢,不过我妈不让养狗。
用户: 你真的喜欢狗吗？
智能体: 我很喜欢狗,我妈说我连自己都照顾不好,还养狗?
思考过程: "不让养"和"连自己都照顾不好还养狗"逻辑是一致的，没有冲突。
结论: {{"thinking": "前后逻辑一致，无冲突", "predicted_label": 0}}

示例2：
用户: 你平时有什么爱好？
智能体: 我是个宅男，周末基本都在家打游戏，从不出门。
用户: 那上周六在爬山的人不是你吗？
智能体: 哦，那是我，我每周六都去爬山。
思考过程: 前文说"周末基本都在家打游戏从不出门"，后文说"每周六都去爬山"，存在直接的事实冲突。
结论: {{"thinking": "前文称从不出门，后文承认每周六去爬山，存在历史事实冲突", "predicted_label": 1}}

【你的任务】
请严格按照示例格式，先输出思考过程，再输出预测标签。
请严格以 JSON 格式输出：
{{
    "thinking": "这里写下你的推理过程，对比前后文是否有冲突",
    "predicted_label": 0
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
        reason = result_json.get("thinking", "")

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

    print(f"原始数据加载成功，共 {len(dataset)} 条样本。")

    # ================= 新增：分层抽样 500 条 =================
    SAMPLE_SIZE = 500

    # 1. 按标签分组
    groups = {}
    for item in dataset:
        label = item["label"]
        if label not in groups:
            groups[label] = []
        groups[label].append(item)

    # 打印一下原始数据的分布情况（写论文时有用的素材）
    print("原始数据标签分布:")
    for label, items in sorted(groups.items()):
        print(f"  标签 {label}: {len(items)} 条")

    # 2. 均衡抽取
    subset = []
    num_classes = len(groups)
    for label, items in groups.items():
        # 计算每个类别平均分多少条，如果某类不够分，就全拿走
        sample_count = min(len(items), max(1, SAMPLE_SIZE // num_classes))
        subset.extend(random.sample(items, sample_count))

    # 3. 如果抽完后不够500条，从剩下的数据里随机补齐
    if len(subset) < SAMPLE_SIZE:
        remaining_ids = set(id(item) for item in dataset) - set(id(item) for item in subset)
        remaining_items = [item for item in dataset if id(item) in remaining_ids]
        subset.extend(random.sample(remaining_items, min(len(remaining_items), SAMPLE_SIZE - len(subset))))

    # 4. 打乱顺序（防止同一类的数据挤在一起）
    random.shuffle(subset)
    dataset = subset[:SAMPLE_SIZE]  # 最终截断为500条
    # =========================================================

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
        time.sleep(0.3)

        # ================= 计算指标 (毕设论文的核心素材) =================
    print("\n" + "=" * 50)
    print("评测结果报告")
    print("=" * 50)

    acc = accuracy_score(true_labels, pred_labels)
    # 计算 Macro F1 (多分类任务看这个指标最严谨)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    true_labels_binary = [0 if l == 0 else 1 for l in true_labels]
    pred_labels_binary = [0 if l == 0 else 1 for l in pred_labels]

    acc_binary = accuracy_score(true_labels_binary, pred_labels_binary)
    f1_binary = f1_score(true_labels_binary, pred_labels_binary, average='binary')
    print(f"\n【二分类结果（仅区分是否有矛盾）】")
    print(f"二分类准确率: {acc_binary:.4f}")
    print(f"二分类 F1 分数: {f1_binary:.4f}")

    print(f"总体准确率: {acc:.4f}")
    print(f"Macro F1 分数: {macro_f1:.4f}")
    print("\n各类别详细指标:")
    # target_names 必须和你的 label 0,1,2,3 一一对应
    print(classification_report(true_labels, pred_labels,
                                target_names=["无矛盾(0)", "句内矛盾(1)", "角色混淆(2)", "历史矛盾(3)"]))

    # 保存详细结果，方便你写论文时分析 Bad Case
    with open("eval_results_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print("\n详细预测结果已保存至 eval_results_details.json")


if __name__ == "__main__":
    run_evaluation()
