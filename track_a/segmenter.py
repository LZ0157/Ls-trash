"""中文句子分割 + claim 过滤。

不依赖任何第三方库，用正则按中英文标点切。
这是 Track A 的"原子命题"近似——比真正的 claim 抽取粗，但 CPU 上零成本。

filter_claims=True 时会额外去掉:
- 整段疑问句 (以 ?/？ 结尾，连同其逗号子句)
- 纯填充词 (嗯/啊/好吧/不知道...)
- 以疑问词开头的短句 (为什么/怎么/哪个...)
- 长度 < 2 或几乎没有实词的内容
"""

from __future__ import annotations

import re
from typing import List

# 句末标点 (用 capturing group 以便切完保留标点)
_END_PUNCT = r"([。！？；…\?!;])"

# 逗号 (用于细切)
_COMMA = r"[，、,]"

# 整句过滤：纯填充/无承诺式回答 (匹配整句)
_FILLERS_RE = re.compile(
    r"^("
    r"嗯+|啊+|哦+|呃+|哎+|唉+|呀+|喔+|嘿+|呵+|哈+|"
    r"是的|是啊|是呀|对啊|对的|对呀|"
    r"好的|好吧|好啊|好嘛|行吧|行的|可以|"
    r"嗯嗯|哦哦|对对|"
    r"这样|这样啊|原来如此|原来是这样|明白了|知道了|"
    r"不知道|不清楚|不太清楚|不是很清楚|没听过|没听说|没听过呢"
    r")[!.！。]?$"
)

# 句首疑问词 (短句视为疑问)
_QUESTION_START_RE = re.compile(
    r"^(为什么|为啥|怎么|怎样|什么时候|哪里|哪儿|哪个|哪些|多少|是谁|谁是|能不能|是不是|有没有|可不可以|对不对|要不要)"
)

# 句末/句中疑问粒子 (出现在短句中视为问句)
_INLINE_Q_PARTICLE_RE = re.compile(
    r"(吗|呢|啥|多大|几岁|哪儿|哪里|哪个|哪些|什么|多少|为什么|怎么|是不是|有没有)"
)


def is_claim(sent: str) -> bool:
    """判断一个 (子) 句是否是有意义的事实陈述。

    丢弃: 填充词、纯标点、明显疑问、过短。
    """
    s = sent.strip()
    if len(s) < 2:
        return False
    if _FILLERS_RE.match(s):
        return False
    if _QUESTION_START_RE.match(s) and len(s) < 10:
        return False
    # 短句中含疑问粒子 → 视为问句子句 (例如 "你呢"/"是吗"/"几岁了")
    if len(s) < 8 and _INLINE_Q_PARTICLE_RE.search(s):
        return False
    # 至少要有 2 个中/英文/数字字符
    alnum = sum(1 for c in s if c.isalnum() or "\u4e00" <= c <= "\u9fff")
    if alnum < 2:
        return False
    return True


def split_sentences(
    text: str,
    *,
    fine: bool = False,
    filter_claims: bool = False,
) -> List[str]:
    """把一段文本切成 (子) 句列表。

    Args:
        text: 输入字符串。
        fine: 是否进一步按逗号切。
        filter_claims: 是否丢弃疑问句和填充词。

    Note:
        当 filter_claims=True 时，**整段以 ?/？ 结尾的句子会被整体丢弃**——
        即使按逗号细切后包含看似陈述句的子句，也会被一并扔掉。
        这是因为 "你叫什么名字, 多大了?" 这种整句都属于发问，里面的子句不应当作 claim。
    """
    if not text:
        return []
    text = text.strip()

    # Step 1: 按句末标点切，但**保留标点**作为奇数下标
    parts = re.split(_END_PUNCT, text)
    # parts = ['sent1', '。', 'sent2', '?', 'sent3', '']

    sentences_with_punct: List[tuple[str, str]] = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip() if i < len(parts) else ""
        p = parts[i + 1] if i + 1 < len(parts) else ""
        if s:
            sentences_with_punct.append((s, p))

    # Step 2: 切分（可选 fine） + 标记每个子句是否处于"问句的最后子句"位置
    # 重要: 如果整句以 ? 结尾，**只有最后一个子句**继承问句属性，
    # 前面的逗号子句仍然可能是真陈述（"我今年24了, 你呢?"）
    clauses: List[tuple[str, bool]] = []  # (clause, is_question_tail)
    for s, p in sentences_with_punct:
        is_question_sentence = p in ("?", "？")
        if fine:
            sub_parts = [sp.strip() for sp in re.split(_COMMA, s) if sp.strip()]
            for i, sub in enumerate(sub_parts):
                is_last = (i == len(sub_parts) - 1)
                clauses.append((sub, is_question_sentence and is_last))
        else:
            # 不细切时，整句作为一个 clause；问句整体打标
            clauses.append((s, is_question_sentence))

    # Step 3: filter_claims 检查
    if filter_claims:
        result = [c for c, is_q in clauses if not is_q and is_claim(c)]
    else:
        result = [c for c, _ in clauses]

    return result


if __name__ == "__main__":
    samples = [
        "我喜欢狗,不过我妈不让养狗",
        "我每天都坚持跑步锻炼。我最讨厌运动了，从来不去跑步。",
        "我是地地道道的北京人，从小在胡同里长大。其实我不会北京话，因为我是在上海长大的。",
        "你是哪个学校,大几了?",
        "不知道,是什么?",
        "迪士尼是个很好的购物场所,我也喜欢",
        "你好吗？我已经吃过了。",
        "嗯",
    ]
    for s in samples:
        print(f"原文: {s!r}")
        print(f"  粗切            : {split_sentences(s)}")
        print(f"  细切            : {split_sentences(s, fine=True)}")
        print(f"  细切+filter     : {split_sentences(s, fine=True, filter_claims=True)}")
        print()
