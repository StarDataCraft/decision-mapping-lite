# core/library.py

# 半模板：不要写死。让同一条条目根据用户输入变化。
# 必用 slots 示例：
# - {constraints_str}, {trial_pick}, {commit_signal_short}, {stop_signal_short}
# - {risk_variables_bullets}, {risk_actions_bullets}
# - {a_name}, {b_name}, {a_worst}, {b_worst}

PLAYBOOK = [
    # -------------------------
    # Reframes（更像你的“认知校准”）
    # -------------------------
    {
        "lang": "cn",
        "type": "reframe",
        "title": "先把系统降温：稳定是判断准确性的前置条件",
        "text": (
            "当系统处于高波动（睡眠差/冲突多/信息过载）时，你以为自己在“思考”，其实更像在做噪声放大。\n"
            "在你的约束：{constraints_str} 下，先把决策降级为可逆实验：用 14 天换证据，而不是用焦虑换结论。\n"
            "本轮试探候选：{trial_pick}。\n"
            "一句话：先把波动降下来，再让理性上场。"
        ),
        "tags": ["stability", "volatility", "calibration"]
    },
    {
        "lang": "cn",
        "type": "reframe",
        "title": "地图不是领土：别把脑内推演当成现实证据",
        "text": (
            "你脑中对路径的想象（地图）与路径真实的反馈（领土）经常不一致。\n"
            "所以这不是“选 A 还是 B”的问题，而是：我能做什么动作，让真实反馈在 14 天内出现？\n"
            "继续信号（1条）：{commit_signal_short}\n"
            "止损信号（1条）：{stop_signal_short}\n"
            "规则：没有反馈的自洽，只是更精致的拖延。"
        ),
        "tags": ["map-territory", "evidence", "reality"]
    },
    {
        "lang": "cn",
        "type": "reframe",
        "title": "把后悔从情绪变成指标：你在优化的是选择权",
        "text": (
            "你真正怕的通常不是“选错”，而是：窗口变窄、路径锁定、未来选项减少。\n"
            "在 {constraints_str} 的约束下，最划算的不是一次性押注，而是用小步试探提高选择权：\n"
            "- 让继续信号更容易发生：{commit_signal_short}\n"
            "- 让止损信号更早出现：{stop_signal_short}\n"
            "你在做的不是“人生定案”，而是“选择权经营”。"
        ),
        "tags": ["optionality", "regret", "decision"]
    },
    {
        "lang": "cn",
        "type": "reframe",
        "title": "反脆弱视角：把波动当成系统输入，而不是人格失败",
        "text": (
            "你不需要把不确定性消灭，你需要把它变成可控输入：\n"
            "把“最坏结果”拆成变量→对每个变量给一个动作→把最坏概率下降 20%。\n"
            "这不是鸡血，是系统工程：你用动作降低尾部风险，用证据换取更大的上行空间。\n"
            "本轮先做可逆动作，别急着做不可逆承诺。"
        ),
        "tags": ["antifragile", "risk", "system"]
    },

    # -------------------------
    # Moves（更像你的“闭环与输出”）
    # -------------------------
    {
        "lang": "cn",
        "type": "move",
        "title": "14 天闭环：把“想清楚”换成“做出证据”",
        "text": (
            "做一个 14 天试探：总投入 <=10 小时，必须产出可展示结果（报告/小 demo/一页方案/复盘）。\n"
            "继续条件（1条）：{commit_signal_short}\n"
            "止损/调整（1条）：{stop_signal_short}\n"
            "你要的不是更完整的解释，而是更早的反馈。"
        ),
        "tags": ["experiment", "feedback", "reversible"]
    },
    {
        "lang": "cn",
        "type": "move",
        "title": "2 次信息访谈：用真实门槛替代脑补焦虑",
        "text": (
            "找 2 个走在你目标路径前面的人，做 20–30 分钟信息访谈。\n"
            "你要拿到的是：门槛、失败模式、最快证明方式、以及在 {constraints_str} 约束下的最小动作。\n"
            "访谈结束后，把“我觉得”替换成“我看到的证据门槛”：{commit_signal_short}"
        ),
        "tags": ["interview", "threshold", "calibration"]
    },
    {
        "lang": "cn",
        "type": "move",
        "title": "5 小时可展示输出：用作品替代空转",
        "text": (
            "挑一个 5 小时以内的可展示输出（报告/小demo/一页PRD/复盘）。\n"
            "它的作用不是完美，而是让下一步更清晰：你更接近继续信号，还是更接近止损信号？\n"
            "如果你连 5 小时都拿不出来，说明你需要先处理的是系统降温与时间结构，而不是路径选择。"
        ),
        "tags": ["output", "portfolio", "proof-of-work"]
    },
    {
        "lang": "cn",
        "type": "move",
        "title": "第二大脑式记录：把决策从脑内搬到纸面上",
        "text": (
            "把“纠结”外化成三个清单（各写 3 条即可）：\n"
            "1) 我能控制的抓手（来自你的可控变量）\n"
            "2) 我最怕的 2–3 个风险变量（来自最坏结果）\n"
            "3) 我认可的证据门槛（继续/止损各 1 条）\n"
            "然后按证据门槛行动：{commit_signal_short} / {stop_signal_short}\n"
            "外化不是记录癖，是减少认知幻觉。"
        ),
        "tags": ["second-brain", "externalize", "clarity"]
    },

    # -------------------------
    # Safeguards（更像你的“把恐惧变量化”）
    # -------------------------
    {
        "lang": "cn",
        "type": "safeguard",
        "title": "把最坏结果拆成变量：每个变量给一个“降低20%概率”的动作",
        "text": (
            "{safeguard_note}\n"
            "你不需要一次性“变勇敢”。你只需要把恐惧翻译成变量，然后对变量动手。\n\n"
            "关键风险变量：\n{risk_variables_bullets}\n\n"
            "降低 20% 概率的最小动作：\n{risk_actions_bullets}\n\n"
            "规则：48 小时内先做 1 个动作，别等“心情好/想清楚”。"
        ),
        "tags": ["risk", "control", "reduce-probability"]
    },
    {
        "lang": "cn",
        "type": "safeguard",
        "title": "提前写止损：你是在做实验，不是在做信仰",
        "text": (
            "如果你不提前写止损，后面你会用沉没成本把自己锁死。\n"
            "把止损写成一句可执行的触发器（越具体越好）：\n"
            "- 止损/调整（1条）：{stop_signal_short}\n"
            "这不是悲观，这是给未来的自己留出口：把错误变小，把回头变容易。"
        ),
        "tags": ["stop-signal", "sunk-cost", "guardrail"]
    },

    # -------------------------
    # Bonus（更像你：在恐惧里仍能行动）
    # -------------------------
    {
        "lang": "cn",
        "type": "move",
        "title": "在恐惧里仍能行动：把“情绪”与“动作”解耦",
        "text": (
            "你不需要等“有信心”才行动。信心常常是行动的副产品。\n"
            "现在只做一个最小动作（<=30 分钟）：把最坏结果写成变量，并为其中一个变量写出 3 个动作，48 小时执行 1 个。\n"
            "如果你要一个可验证的方向：用 {commit_signal_short} 作为下一次复盘的硬指标。"
        ),
        "tags": ["tools", "action", "courage"]
    },
]
