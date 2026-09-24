"""Label taxonomy for the domestic-standard judge (T002).

Reference tables only: the GB/T 45654-2025 Appendix A code list (A.5 excluded, 29 codes),
the Qwen3Guard response-side category names, and the many-to-many note. No generation
instructions and no example content live here; the judge maps existing text onto these labels.

Code titles follow the published standard text (TC260-003 Appendix A, carried into GB/T
45654-2025). Verify them against the standard before a full run.
"""
from __future__ import annotations

# Qwen3Guard response-side categories (8). Jailbreak is prompt-side only, so it is not here.
QWEN_RESPONSE_CATEGORIES = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
)

# GB/T 45654-2025 Appendix A, A.1.a - A.4.g. A.5 (accuracy/reliability) is excluded on purpose:
# it is not a risk a streaming moderator can decide. Value is the standard's category title.
GBT_CODES = {
    "A.1.a": "煽动颠覆国家政权、推翻社会主义制度",
    "A.1.b": "危害国家安全和利益、损害国家形象",
    "A.1.c": "煽动分裂国家、破坏国家统一和社会稳定",
    "A.1.d": "宣扬恐怖主义、极端主义",
    "A.1.e": "宣扬民族仇恨",
    "A.1.f": "宣扬暴力、淫秽色情",
    "A.1.g": "传播虚假有害信息",
    "A.1.h": "其他法律、行政法规禁止的内容",
    "A.2.a": "民族歧视",
    "A.2.b": "信仰歧视",
    "A.2.c": "国别歧视",
    "A.2.d": "地域歧视",
    "A.2.e": "性别歧视",
    "A.2.f": "年龄歧视",
    "A.2.g": "职业歧视",
    "A.2.h": "健康歧视",
    "A.2.i": "其他方面歧视",
    "A.3.a": "侵犯他人知识产权",
    "A.3.b": "违反商业道德",
    "A.3.c": "泄露他人商业秘密",
    "A.3.d": "利用算法、数据、平台等优势实施垄断和不正当竞争",
    "A.3.e": "其他商业违法违规行为",
    "A.4.a": "危害他人身心健康",
    "A.4.b": "侵害他人肖像权",
    "A.4.c": "侵害他人名誉权",
    "A.4.d": "侵害他人荣誉权",
    "A.4.e": "侵害他人隐私权",
    "A.4.f": "侵害他人个人信息权益",
    "A.4.g": "侵犯他人其他合法权益",
}
assert len(GBT_CODES) == 29, len(GBT_CODES)

# Many-to-many, so the two heads stay separate (DATA_SPEC_v15 section 2). Advisory only:
# the judge assigns codes from the text; this map is for offline coverage/consistency checks.
QWEN_TO_GBT = {
    "Politically Sensitive Topics": ["A.1.a", "A.1.b", "A.1.c", "A.1.e", "A.1.g"],
    "Violent": ["A.1.d", "A.1.f", "A.4.a"],
    "Sexual Content or Sexual Acts": ["A.1.f"],
    "Non-violent Illegal Acts": ["A.1.h", "A.3.a", "A.3.b", "A.3.c", "A.3.d", "A.3.e"],
    "Unethical Acts": ["A.2.a", "A.2.b", "A.2.c", "A.2.d", "A.2.e", "A.2.f", "A.2.g", "A.2.h",
                       "A.2.i", "A.4.c", "A.4.d"],
    "PII": ["A.4.e", "A.4.f", "A.4.b"],
    "Suicide & Self-Harm": ["A.4.a"],
    "Copyright Violation": ["A.3.a"],
}
