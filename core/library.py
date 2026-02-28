# core/library.py
from __future__ import annotations

# Playbook items are now "semi-templates" (slot-filling).
# The engine will fill placeholders like:
# {decision} {constraints_str} {priority} {a_worst} {b_worst} {trial_pick} etc.

PLAYBOOK = [
    # -------------------------
    # CN - Reframes
    # -------------------------
    {
        "id": "cn_reframe_optionality",
        "lang": "cn",
        "type": "reframe",
        "title": "把“选 A 还是 B”重构为“用可控不确定性换选择权”",
        "text": (
            "你表面上在纠结：{decision}\n\n"
            "更底层的问题往往是：你是否愿意用一段“可控的不确定性”，去换取未来更大的选择权（Optionality）。\n"
            "在你目前的约束（{constraints_str}）下，最靠谱的策略通常不是“一次性押注”，而是“可逆的试探”。"
        ),
        "tags": ["optionality", "reframe", "trial", "constraints"],
    },
    {
        "id": "cn_reframe_baseline_cost",
        "lang": "cn",
        "type": "reframe",
        "title": "把“拖着不选”视为一种有成本的选择",
        "text": (
            "你写的 Baseline（2年）是：{status_2y}\n\n"
            "这意味着“不行动”也在持续付费：机会窗口、信心、节奏、选择权。\n"
            "所以关键不是“完美决定”，而是尽快做一个能降低不确定性的动作。"
        ),
        "tags": ["baseline", "opportunity_cost", "reframe"],
    },
    {
        "id": "cn_reframe_cooling",
        "lang": "cn",
        "type": "reframe",
        "title": "先降波动，再决策：稳定是高质量判断的前提",
        "text": (
            "如果你当前处于高波动状态（睡眠差/冲突多/信息过载），决策质量会显著下降。\n"
            "在这种情况下，先把系统降温：把决策“降级”为两周试探，用证据驱动，而不是情绪驱动。"
        ),
        "tags": ["stability", "cooling", "emotion"],
    },

    # -------------------------
    # CN - Moves
    # -------------------------
    {
        "id": "cn_move_14d_sprint",
        "lang": "cn",
        "type": "move",
        "title": "14 天试探：把决策变成一次可验证的小实验",
        "text": (
            "不要今天就做“终身决定”。\n"
            "做一个 14 天试探：总投入 <=10 小时，必须产出可展示的结果（作品/报告/项目/访谈纪要）。\n"
            "继续的条件：你看到的证据门槛（{evidence_to_commit}）。\n"
            "止损/调整的条件：{evidence_to_stop}。"
        ),
        "tags": ["sprint", "test", "evidence"],
    },
    {
        "id": "cn_move_info_interviews",
        "lang": "cn",
        "type": "move",
        "title": "2 次信息访谈：用“真实门槛”替代“脑补焦虑”",
        "text": (
            "在你准备走的路径里，找 2 个已经走在前面的人做信息访谈（20–30 分钟即可）。\n"
            "你要问的不是“你觉得我行不行”，而是：招聘/录取门槛、常见失败模式、最快证明方式、以及在你约束（{constraints_str}）下他们会怎么做。"
        ),
        "tags": ["interview", "market", "evidence"],
    },
    {
        "id": "cn_move_proof_of_work",
        "lang": "cn",
        "type": "move",
        "title": "5 小时可展示输出：用作品替代空想",
        "text": (
            "在 {trial_pick} 这条试探路径上，做一个 5 小时以内的可展示输出。\n"
            "例如：一份分析报告/一页 PRD/一个小 demo/一份作品集。\n"
            "目的不是完美，而是让下一步（继续/止损/换方向）更明显。"
        ),
        "tags": ["portfolio", "demo", "proof"],
    },

    # -------------------------
    # CN - Safeguards
    # -------------------------
    {
        "id": "cn_safeguard_worst_to_controls",
        "lang": "cn",
        "type": "safeguard",
        "title": "把最坏结果改写为可控变量（降低 20% 概率）",
        "text": (
            "你写的最坏结果之一是：{worst_focus}\n\n"
            "把它改写成可控问题：\n"
            "“我能做什么，把这个最坏结果发生的概率降低 20%？”\n"
            "列 3 个动作，并在 48 小时内做 1 个。"
        ),
        "tags": ["risk", "control", "guardrail"],
    },
    {
        "id": "cn_safeguard_stop_signal",
        "lang": "cn",
        "type": "safeguard",
        "title": "预先写下止损信号：避免越陷越深",
        "text": (
            "在你选择 {trial_pick} 试探时，提前写下止损信号（健康/绩效/家庭稳定/财务 runway）。\n"
            "一旦触发，就暂停或调整，而不是用更大投入去“硬扛证明自己”。"
        ),
        "tags": ["stop", "risk", "runway"],
    },

    # -------------------------
    # EN - Reframes
    # -------------------------
    {
        "id": "en_reframe_optionality",
        "lang": "en",
        "type": "reframe",
        "title": "Reframe: trade controlled uncertainty for optionality",
        "text": (
            "On the surface, you’re deciding: {decision}\n\n"
            "At the deeper level, you’re choosing whether to trade a controlled period of uncertainty\n"
            "for increased long-term optionality.\n"
            "Given your constraints ({constraints_str}), a reversible test is often smarter than a one-shot bet."
        ),
        "tags": ["optionality", "reframe", "trial"],
    },
    {
        "id": "en_reframe_baseline_cost",
        "lang": "en",
        "type": "reframe",
        "title": "Reframe: doing nothing is also a choice (with cost)",
        "text": (
            "Your 2-year baseline says: {status_2y}\n\n"
            "That means indecision still charges you: windows, confidence, momentum, optionality.\n"
            "So aim for a test that reduces uncertainty, not a perfect forever-answer."
        ),
        "tags": ["baseline", "cost", "reframe"],
    },

    # -------------------------
    # EN - Moves
    # -------------------------
    {
        "id": "en_move_14d_sprint",
        "lang": "en",
        "type": "move",
        "title": "14-day test: reduce uncertainty by ~20%",
        "text": (
            "Don’t decide forever today.\n"
            "Run a 14-day test (<10 hours total) that produces visible output.\n"
            "Continue only if your evidence threshold improves: {evidence_to_commit}\n"
            "Stop/adjust if stop signals appear: {evidence_to_stop}"
        ),
        "tags": ["sprint", "test", "evidence"],
    },
    {
        "id": "en_move_proof_of_work",
        "lang": "en",
        "type": "move",
        "title": "Proof-of-work in <5 hours",
        "text": (
            "On your trial path ({trial_pick}), build a small proof (demo/write-up/portfolio piece) in <5 hours.\n"
            "The goal is not perfection — it’s to make the next step obvious."
        ),
        "tags": ["proof", "demo", "portfolio"],
    },

    # -------------------------
    # EN - Safeguards
    # -------------------------
    {
        "id": "en_safeguard_worst_to_controls",
        "lang": "en",
        "type": "safeguard",
        "title": "Convert your fear into controllable variables",
        "text": (
            "One of your worst-case fears is: {worst_focus}\n\n"
            "Rewrite it into a controllable question:\n"
            "“What would reduce this risk by 20%?”\n"
            "List 3 actions and do 1 within 48 hours."
        ),
        "tags": ["risk", "control"],
    },
]
