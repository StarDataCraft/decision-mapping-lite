import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Decision Mapping Lite", page_icon="🧭", layout="centered")

st.title("🧭 Decision Mapping Lite")
st.caption("把一个重要选择从“纠结”变成“可执行”。（你的思考风格版：重构问题 → 约束 → 可控性 → 证据门槛 → 试探路径）")

with st.expander("使用说明（30秒）", expanded=True):
    st.markdown(
        """
- 只聚焦一个决策（不要同时处理多个）。
- 尽量写具体：用**事实**而不是情绪形容词。
- 输出不是“标准答案”，而是一份**可解释的推导链** + **可执行下一步**。
"""
    )

st.divider()


# ---------------------------
# Helpers
# ---------------------------
def _bulletize(text: str) -> str:
    """把用户输入里可能的顿号/项目符号统一成 markdown 列表展示。"""
    if not text:
        return "（未填写）"
    t = text.strip()
    # 兼容用户用 "•" 或 " - " 或换行
    lines = [ln.strip(" \t•-") for ln in t.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) <= 1:
        return t
    return "\n".join([f"- {ln}" for ln in lines])


def _text_richness_score(text: str) -> int:
    """粗略判断：描述是否具体（越具体越可推导）。0~3"""
    if not text or not text.strip():
        return 0
    t = text.strip()
    score = 0
    if len(t) >= 30:
        score += 1
    if "\n" in t:
        score += 1
    verbs = ["每周", "每天", "固定", "完成", "参与", "整理", "申请", "复盘", "构建", "提交", "发布", "联系", "输出"]
    if any(v in t for v in verbs):
        score += 1
    return score


def _worst_severity(worst: str) -> int:
    """最坏结果严重度：0/1/2（启发式）"""
    if not worst:
        return 0
    severe_keywords = ["崩", "失业", "毁", "无法", "严重", "抑郁", "破产", "健康", "关系破裂", "绩效很差", "重病"]
    return 2 if any(k in worst for k in severe_keywords) else 1


# ---------------------------
# Inputs
# ---------------------------
decision = st.text_area(
    "1) 你正在面对的关键决策是什么？",
    placeholder="例如：是否转行 / 是否读博 / 是否接受offer / 是否创业",
    height=80,
)

options = st.text_area(
    "2) 你的备选路径有哪些？（至少2个）",
    placeholder="A: ...\nB: ...\nC: ...（可选）",
    height=110,
)

status_6m = st.text_area("3) 如果你什么都不改变：6个月后最可能是什么状态？", height=80)
status_2y = st.text_area("4) 如果你什么都不改变：2年后最可能是什么状态？", height=80)

st.subheader("路径拆解（A / B 必填）")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 路径 A")
    a_name = st.text_input("A 的名字（简短）", value="路径A")
    a_best = st.text_area("A：最好结果", height=80)
    a_worst = st.text_area("A：最坏结果", height=80)
    a_controls = st.text_area("A：可控变量（你能做什么来降低风险）", height=90)

with col2:
    st.markdown("### 路径 B")
    b_name = st.text_input("B 的名字（简短）", value="路径B")
    b_best = st.text_area("B：最好结果", height=80)
    b_worst = st.text_area("B：最坏结果", height=80)
    b_controls = st.text_area("B：可控变量（你能做什么来降低风险）", height=90)

st.subheader("目标与约束")

priority = st.radio(
    "5) 现在对你最重要的是哪一个？",
    ["稳定性", "收入", "成长", "自由度", "长期选择权（Optionality）"],
    index=4,
)

constraints = st.multiselect(
    "6) 你现在的真实约束是？（多选）",
    ["财务", "时间", "家庭", "技能差距", "情绪耐受度", "健康", "地理位置", "身份/自我叙事", "其他"],
)

regret = st.radio(
    "7) 5年后的你回看今天：更可能后悔哪一种？",
    ["没有尝试", "冒险失败"],
    index=0,
)

st.subheader("证据与试探（你的风格核心）")

evidence_to_commit = st.text_area(
    "8) 你需要看到什么证据，才会对某条路径“加码/承诺”？（证据门槛）",
    placeholder="例如：完成一个可展示项目；拿到2次面试；内部找到mentor；每周稳定投入10小时持续4周…",
    height=90,
)

evidence_to_stop = st.text_area(
    "9) 你需要看到什么信号，才会“止损/换路径”？（止损条件）",
    placeholder="例如：连续4周无法投入；健康/家庭压力显著上升；投入没有产生可验证输出…",
    height=90,
)

partial_control = st.text_area(
    "10) 你现在最关键的「部分可控」变量是什么？你准备怎么把它往有利方向推一点？",
    placeholder="例如：时间块（部分可控）→ 早晨固定45分钟；情绪耐受（部分可控）→ 先把决策拆成两周试探…",
    height=90,
)

identity_anchor = st.text_area(
    "11) 这个决策与你想成为的那种人有什么关系？（身份轨迹锚）",
    placeholder="例如：我想成为能在不确定中保持稳定推进的人；我想拥有更高选择权…",
    height=80,
)

st.divider()


# ---------------------------
# Memo builder (your-style)
# ---------------------------
def build_memo() -> str:
    # --- Feature extraction (可解释指标) ---
    optionality_pref = priority == "长期选择权（Optionality）"
    regret_try = regret == "没有尝试"

    baseline_lockin_risk = False
    if status_2y and any(k in status_2y for k in ["锁定", "变窄", "越来越难", "窗口期", "定型", "不可逆", "被困", "停滞"]):
        baseline_lockin_risk = True

    a_control_rich = _text_richness_score(a_controls)
    b_control_rich = _text_richness_score(b_controls)

    a_worst_sev = _worst_severity(a_worst)
    b_worst_sev = _worst_severity(b_worst)

    constraint_count = len(constraints) if constraints else 0
    high_constraint = constraint_count >= 3

    # --- Scoring (可解释评分) ---
    score_min_regret = 0
    reasons = []

    if optionality_pref:
        score_min_regret += 2
        reasons.append("你把「长期选择权」放在首位 → 更适合优先选择能增加未来选项的策略。")

    if regret_try:
        score_min_regret += 2
        reasons.append("你的后悔倾向是「没有尝试」→ 更适合先试探、再收敛，而不是直接回避尝试。")

    if baseline_lockin_risk:
        score_min_regret += 1
        reasons.append("Baseline（2年）里出现「路径锁定/窗口变窄」信号 → 不行动的机会成本偏高。")

    if max(a_control_rich, b_control_rich) >= 2:
        score_min_regret += 1
        reasons.append("至少一条路径的可控抓手足够具体 → 可以用「小步试探」把风险变成计划。")
    else:
        reasons.append("两条路径的可控抓手都偏模糊 → 先补信息/资源，再做大动作会更稳。")

    if high_constraint:
        score_min_regret += 1
        reasons.append(f"你当前约束较多（{constraint_count}项）→ 更适合分阶段，而不是一次性押注。")

    # --- Decide recommendation ---
    # 阈值：>=4 倾向“最小后悔 + 试探”
    if score_min_regret >= 4:
        conclusion = "更偏向「最小后悔路径（Minimum Regret Path）」：用可控的不确定性，去换取长期选择权的上升。"
        strategy = (
            "采用「影子转型 / 小步试探」：保留核心稳定来源（现金流/身份/基本盘），"
            "用固定时间块推进新方向；设定 4–6 周复盘点，用证据决定是否加码。"
        )
        next48 = [
            "把两条路径各自的“最坏结果”改写成可控问题：我能做什么把最坏结果概率降低 20%？写出 3 条。",
            "选一个最小验证（<=10小时/两周内完成）：一个作品/一次访谈/一次内部尝试/一个小交付。",
            "写下你的“证据门槛”：达到什么证据就加码？出现什么信号就止损？并设定复盘时间点。",
        ]
    else:
        conclusion = "更偏向「先降波动、再决策」：先让系统稳定，再谈方向切换。"
        strategy = (
            "先补齐关键约束（财务 runway / 时间块 / 技能缺口 / 支持系统），"
            "用 2–4 周建立稳定节奏与信息质量，然后做第二轮决策。"
        )
        next48 = [
            "把你的关键约束按“可控/不可控/部分可控”分类，并挑 1 个部分可控项先动手。",
            "为路径 A/B 各写出你缺少的 3 个关键事实（例如招聘门槛、内部机会、学习路径成本）。",
            "设定一个决策时间点：2–4 周后用补齐的信息再做一次 Decision Mapping。",
        ]

    # --- Trial pick (可解释的试探路径候选) ---
    a_manage = a_control_rich - a_worst_sev
    b_manage = b_control_rich - b_worst_sev
    if a_manage == b_manage:
        trial_pick = f"{a_name} 或 {b_name}（两者可控性接近，建议以约束与证据门槛决定）"
    else:
        trial_pick = a_name if a_manage > b_manage else b_name

    # --- Your-style narrative reasoning ---
    reframe = (
        "你表面上在做的是“选 A 还是选 B”，但更底层的问题通常是："
        "你是否愿意用一段可控的不确定性，去换取长期选择权的上升。"
    )

    logic_chain = f"""
我使用的推导顺序是：先看目标函数 → 再看约束 → 再看不行动的代价（Baseline）→ 再看两条路径的可控抓手 → 最后用证据门槛把“试探”落地。

- 目标函数：你最重视「{priority}」，且后悔倾向为「{regret}」。
- 约束：你当前约束为 {", ".join(constraints) if constraints else "（未选择）"}。约束越多，越不建议“一刀切”，分阶段更稳。
- Baseline 代价：如果维持现状，2年后你描述的状态是「{(status_2y or "（未填写）").strip()}」。这意味着“不行动也在付费”（机会成本/锁定）。
- 可控抓手：{a_name} 可控性（richness={a_control_rich}），{b_name} 可控性（richness={b_control_rich}）。可控抓手越具体，越适合试探。
- 因此结论倾向：{conclusion}
""".strip()

    control_section = f"""
## 5. 可控 / 不可控 / 部分可控（把焦虑变成变量）
- 可控（你能直接改变的）：来自两条路径的「可控变量」清单（见上）。
- 不可控（你无法决定的）：市场/组织/他人决策/宏观环境等（你的动作是“对冲”，不是内耗）。
- 部分可控（最值得投入的）：{partial_control.strip() if partial_control else "（未填写）"}
""".strip()

    # ---------
    # IMPORTANT FIX:
    # Do NOT put expressions containing backslashes (like "\n") inside f-string { }.
    # Precompute the joined texts first.
    # ---------
    reasons_text = "\n".join(reasons)
    next48_text = "\n".join(next48)

    # Also precompute bulletized versions (keeps f-string clean)
    reasons_bullets = _bulletize(reasons_text)
    next48_bullets = _bulletize(next48_text)

    # --- Memo assembly (段落为主) ---
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

{control_section}

## 6. 目标函数与约束（你在优化什么）
你最重视的是「{priority}」。你当前的约束是：{", ".join(constraints) if constraints else "（未选择）"}。你的后悔倾向是「{regret}」。
这些信息决定了：你更适合“一次性押注”，还是“先试探再加码”。

## 7. 问题重构（Reframing）
{reframe}

## 8. 推导链（Why）
{logic_chain}

### 8.1 关键指标（features）
- 最小后悔倾向得分：{score_min_regret}（阈值：4 及以上更倾向「试探型最小后悔路径」）
- 可控性（richness，0~3）：
  - {a_name}: {a_control_rich}
  - {b_name}: {b_control_rich}
- 最坏结果严重度（粗略，0~2）：
  - {a_name}: {a_worst_sev}
  - {b_name}: {b_worst_sev}
- 约束数量：{constraint_count}（>=3 视为约束较多，优先分阶段）
- 试探路径候选（根据“可控性 - 最坏严重度”粗略选择）：{trial_pick}

### 8.2 触发的推导依据
{reasons_bullets}

## 9. 证据门槛（用证据而不是情绪做决定）
- 加码/承诺的证据：{evidence_to_commit.strip() if evidence_to_commit else "（未填写）"}
- 止损/换路径的信号：{evidence_to_stop.strip() if evidence_to_stop else "（未填写）"}

## 10. 建议（可执行版本）
结论：{conclusion}

策略：{strategy}

## 11. 下一步（48小时内）
{next48_bullets}

## 12. 身份轨迹锚（你正在成为谁）
{identity_anchor.strip() if identity_anchor else "（未填写）"}

## 13. 复盘点（建议）
- 复盘时间：4–6 周
- 复盘问题：哪些证据支持继续加码？哪些证据支持止损或换路径？
"""
    return memo


# ---------------------------
# Render button + output
# ---------------------------
ready = st.button("✅ 生成我的决策备忘录", type="primary", use_container_width=True)

if ready:
    memo = build_memo()
    st.success("已生成。建议先通读，再复制到你的笔记里。")
    st.code(memo, language="markdown")
    st.download_button(
        label="⬇️ 下载为 Markdown（.md）",
        data=memo.encode("utf-8"),
        file_name="decision_memo.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.caption("提示：如果你希望把这份 memo 升级成“可执行路线图”，可以在此基础上加：财务 runway、技能差距、时间块计划、风险对冲方案。")
