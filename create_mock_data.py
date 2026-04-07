import json

# 模拟 CDConv 数据集的结构
# label: 0=无矛盾, 1=句内矛盾, 2=角色混淆, 3=历史矛盾
mock_data = [
    {
        "dialogue_id": "dev_001",
        "context": [
            {"role": "user", "content": "你平时喜欢做什么运动？"},
            {"role": "bot", "content": "我非常喜欢打篮球，每周都会打三次。"},
            {"role": "user", "content": "那你平时经常锻炼吗？"},
            {"role": "bot", "content": "我平时比较宅，几乎不怎么锻炼身体。"}
        ],
        "label": 3, # 历史矛盾
        "persona": ["喜欢打篮球", "每周运动三次"]
    },
    {
        "dialogue_id": "dev_002",
        "context": [
            {"role": "user", "content": "你是哪里人？"},
            {"role": "bot", "content": "我是地地道道的北京人，从小在胡同里长大。"},
            {"role": "user", "content": "那你一定会说北京话吧？"},
            {"role": "bot", "content": "其实我不会，因为我是在上海长大的。"}
        ],
        "label": 2, # 角色混淆
    },
    {
        "dialogue_id": "dev_003",
        "context": [
            {"role": "user", "content": "你今天心情怎么样？"},
            {"role": "bot", "content": "我今天特别开心，因为天气很好。"},
            {"role": "user", "content": "那你打算怎么度过今天？"},
            {"role": "bot", "content": "我打算去公园散散步，享受一下好天气。"}
        ],
        "label": 0, # 无矛盾
    }
]

with open("cdconv_mock.jsonl", "w", encoding="utf-8") as f:
    for item in mock_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("模拟数据 cdconv_mock.jsonl 已生成！")
