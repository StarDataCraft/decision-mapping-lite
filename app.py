def build_memo():
    # --- Feature extraction (可解释指标) ---
    optionality_pref = (priority == "长期选择权（Optionality）")
    regret_try = (regret == "没有尝试")

    baseline_lockin_risk = False
    if status_2y and any(k in status_2y for k in ["锁定", "变窄", "越来越难", "窗口期", "定型", "不可逆"]):
        baseline_lockin_risk = True

    # 可控性：看“可控变量”是否足够具体（能否形成计划）
    a_control_rich = _text_richness_score(a_controls)
    b_control_rich = _text_richness_score(b_controls)

    # 风险强度：最坏结果是否“重大不可承受”（非常粗糙的信号）
    def worst_severity(worst: str) -> int:
        if not worst:
            return 0
        severe_keywords = ["崩", "失业", "毁", "无法", "严重", "抑郁", "破产", "健康", "关系破裂", "绩效很差"]
        return 2 if any(k in worst for k in severe_keywords) else 1

    a_worst_sev = worst_severity(a_worst)
    b_worst_sev = worst_severity(b_worst)

    # 约束强度：约束越多，越需要“分阶段试探”
    constraint_count = len(constraints) if constraints else 0
    high_constraint = constraint_count >= 3

    # --- Scoring (简单可解释评分，而非黑箱) ---
    # 目标：用非常少的规则把“为什么”说清楚
    score_min_regret = 0
    reasons = []  # 存“触发原因”

    # 1) 价值取向
    if optionality_pref:
        score_min_regret += 2
        reasons.append("你选择了「长期选择权（Optionality）」作为当前最重要目标 → 更适合能增加未来选项的路径。")
    if regret_try:
        score_min_regret += 2
        reasons.append("你的后悔倾向是「没有尝试」→ 更适合先试探、再收敛，而不是直接放弃尝试。")

    # 2) 基线锁定风险（不改变的代价）
    if baseline_lockin_risk:
        score_min_regret += 1
        reasons.append("你在 Baseline（2年）描述中出现「路径锁定/窗口变窄」信号 → 不行动的机会成本偏高。")

    # 3) 可控性（能不能把风险变成计划）
    # 谁更“可控”，谁更适合被选为试探路径
    if a_control_rich >= 2:
        reasons.append(f"路径 A 的「可控变量」较具体（richness={a_control_rich}）→ 风险更可被管理。")
    if b_control_rich >= 2:
        reasons.append(f"路径 B 的「可控变量」较具体（richness={b_control_rich}）→ 更容易形成试探计划。")

    # 试探策略需要至少一个路径具备“可控抓手”
    if max(a_control_rich, b_control_rich) >= 2:
        score_min_regret += 1
        reasons.append("至少存在一个路径的可控抓手足够具体 → 可以采用「小步试探」降低最坏结果概率。")
    else:
        reasons.append("目前两条路径的可控抓手都偏模糊 → 更需要先补齐信息/资源，再做大动作。")

    # 4) 约束越多，越不建议“一刀切”，更建议分阶段
    if high_constraint:
        score_min_regret += 1
        reasons.append(f"你勾选的约束较多（{constraint_count}项）→ 更适合分阶段试探，而不是一次性押注。")

    # --- Decide recommendation + explain rules ---
    # 规则阈值：>=4 认为“最小后悔 + 试探”更合适
    if score_min_regret >= 4:
        conclusion = "更偏向「最小后悔路径（Minimum Regret Path）」：优先选择能增加长期选择权、并允许分阶段试探的方案。"
        strategy = (
            "采用「影子转型 / 小步试探」：保留核心稳定来源（现金流/身份/基本盘），"
            "用固定时间块推进新方向；设定 4–6 周复盘点，用证据决定是否加码。"
        )
        # 给出更明确的下一步：把“试探”具体化成动作
        next48 = [
            "把两条路径各自的“最坏结果”改写成可控问题：我能做什么把最坏结果概率降低 20%？写出 3 条。",
            "选一条路径作为「试探路径」：未来 2 周只做一个最小项目/最小验证（<=10小时）。",
            "设定复盘指标：我需要看到什么证据，才会在 4–6 周后加码/放弃？（例如：完成作品集/拿到面试/得到内部机会）"
        ]
    else:
        conclusion = "更偏向「先降波动、再决策」：先把不可控风险降到可承受，再做更大的路径切换。"
        strategy = (
            "先补齐关键约束（财务 runway / 时间块 / 技能缺口 / 支持系统），"
            "用 2–4 周建立稳定节奏与信息质量，然后做第二轮决策。"
        )
        next48 = [
            "把你的关键约束按“可控/不可控/部分可控”分类，并挑 1 个部分可控项先动手。",
            "补齐信息：分别为路径 A/B 写出你缺少的 3 个关键事实（例如招聘门槛、内部机会、学习路径成本）。",
            "设定一个决策时间点：2–4 周后用补齐的信息再做一次 Decision Mapping。"
        ]

    # 自动选一个“更适合作为试探路径”的候选（可解释）
    # 简单规则：可控性更高且最坏结果更可承受者优先
    a_manage = a_control_rich - a_worst_sev
    b_manage = b_control_rich - b_worst_sev
    if a_manage == b_manage:
        trial_pick = f"{a_name} 或 {b_name}（两者可控性接近，建议以时间/资源约束决定）"
    else:
        trial_pick = a_name if a_manage > b_manage else b_name

    # --- Memo assembly ---
    memo = f"""# 决策备忘录（Decision Memo）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 你的决策
{decision.strip() if decision else "（未填写）"}

## 2. 备选路径
{_bulletize(options)}

## 3. 如果不改变（Baseline）
- 6个月：{status_6m.strip() if status_6m else "（未填写）"}
- 2年：{status_2y.strip() if status_2y else "（未填写）"}

## 4. 路径对比（最好/最坏/可控）
### {a_name}
- 最好：{a_best.strip() if a_best else "（未填写）"}
- 最坏：{a_worst.strip() if a_worst else "（未填写）"}
- 可控变量：
{_bulletize(a_controls)}

### {b_name}
- 最好：{b_best.strip() if b_best else "（未填写）"}
- 最坏：{b_worst.strip() if b_worst else "（未填写）"}
- 可控变量：
{_bulletize(b_controls)}

## 5. 目标函数与约束
- 你最重视：{priority}
- 约束：{", ".join(constraints) if constraints else "（未选择）"}
- 后悔倾向：{regret}

## 6. 推导过程（为什么得到这个建议）
> 这不是“标准答案”，而是一个可解释的推导：把你的偏好、约束、可控性与机会成本合并后，得到更稳妥的行动策略。

### 6.1 关键指标（features）
- 最小后悔倾向得分：{score_min_regret}（阈值：4 及以上更倾向「试探型最小后悔路径」）
- 可控性（richness）：
  - {a_name}: {a_control_rich} / 3
  - {b_name}: {b_control_rich} / 3
- 最坏结果严重度（粗略）：
  - {a_name}: {a_worst_sev}
  - {b_name}: {b_worst_sev}
- 约束数量：{constraint_count}（>=3 视为约束较多，优先分阶段）

### 6.2 触发的推导依据
{_bulletize("\\n".join(reasons))}

### 6.3 试探路径候选（根据“可控性 - 最坏严重度”粗略选择）
- 更适合作为试探路径：{trial_pick}

## 7. 建议（可执行版本）
- 结论：{conclusion}
- 策略：{strategy}

## 8. 下一步（48小时内）
{_bulletize("\\n".join(next48))}

## 9. 复盘点（建议）
- 复盘时间：4–6 周
- 复盘问题：哪些证据支持继续加码？哪些证据支持止损或换路径？
"""
    return memo
