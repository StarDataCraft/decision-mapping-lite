import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Decision Mapping Lite", page_icon="🧭", layout="centered")

st.title("🧭 Decision Mapping Lite")
st.caption("把一个重要选择从“纠结”变成“可执行”。（自助版）")

with st.expander("使用说明（30秒）", expanded=True):
    st.markdown("""
- 只聚焦一个决策（不要同时处理多个）。
- 尽量写具体：用**事实**而不是情绪形容词。
- 填完后你会得到一份可复制的“决策备忘录”。
""")

st.divider()

# --- Inputs ---
decision = st.text_area("1) 你正在面对的关键决策是什么？", placeholder="例如：是否转行 / 是否读博 / 是否接受offer / 是否创业", height=80)

options = st.text_area("2) 你的备选路径有哪些？（至少2个）", placeholder="A: ...\nB: ...\nC: ...（可选）", height=100)

status_6m = st.text_area("3) 如果你什么都不改变：6个月后最可能是什么状态？", height=80)
status_2y = st.text_area("4) 如果你什么都不改变：2年后最可能是什么状态？", height=80)

st.subheader("路径拆解（A / B 必填）")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 路径 A")
    a_name = st.text_input("A 的名字（简短）", value="路径A")
    a_best = st.text_area("A：最好结果", height=80)
    a_worst = st.text_area("A：最坏结果", height=80)
    a_controls = st.text_area("A：可控变量（你能做什么来降低风险）", height=80)

with col2:
    st.markdown("### 路径 B")
    b_name = st.text_input("B 的名字（简短）", value="路径B")
    b_best = st.text_area("B：最好结果", height=80)
    b_worst = st.text_area("B：最坏结果", height=80)
    b_controls = st.text_area("B：可控变量（你能做什么来降低风险）", height=80)

st.subheader("目标与约束")

priority = st.radio(
    "5) 现在对你最重要的是哪一个？",
    ["稳定性", "收入", "成长", "自由度", "长期选择权（Optionality）"],
    index=4
)

constraints = st.multiselect(
    "6) 你现在的真实约束是？（多选）",
    ["财务", "时间", "家庭", "技能差距", "情绪耐受度", "健康", "地理位置", "身份/自我叙事", "其他"]
)

regret = st.radio(
    "7) 5年后的你回看今天：更可能后悔哪一种？",
    ["没有尝试", "冒险失败"],
    index=0
)

st.divider()

def build_memo():
    # Heuristics: minimum regret / optionality bias + controllability
    controllability_score = 0
    for txt in [a_controls, b_controls]:
        if txt and len(txt.strip()) >= 20:
            controllability_score += 1

    optionality_bias = (priority == "长期选择权（Optionality）") or (regret == "没有尝试")
    has_baseline_lockin = (len((status_2y or "").strip()) > 0)

    # Recommendation
    if optionality_bias and controllability_score >= 1:
        rec = "更偏向「最小后悔路径（Minimum Regret Path）」：优先选择能增加长期选择权、并允许你分阶段试探的方案。"
        tactic = "建议采用「影子转型 / 小步试探」：保留核心稳定来源，同时用固定时间块推进新方向；设定 4–6 周复盘点。"
    else:
        rec = "更偏向「降低短期波动」：优先把不可控风险降到可承受，再做大动作。"
        tactic = "建议先补齐关键约束（财务/健康/技能/时间），用 2–4 周建立稳定节奏，再做第二轮决策。"

    # Memo text
    memo = f"""# 决策备忘录（Decision Memo）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 你的决策
{decision.strip() if decision else "（未填写）"}

## 2. 备选路径
{options.strip() if options else "（未填写）"}

## 3. 如果不改变（Baseline）
- 6个月：{status_6m.strip() if status_6m else "（未填写）"}
- 2年：{status_2y.strip() if status_2y else "（未填写）"}

## 4. 路径对比（最好/最坏/可控）
### {a_name}
- 最好：{a_best.strip() if a_best else "（未填写）"}
- 最坏：{a_worst.strip() if a_worst else "（未填写）"}
- 可控变量：{a_controls.strip() if a_controls else "（未填写）"}

### {b_name}
- 最好：{b_best.strip() if b_best else "（未填写）"}
- 最坏：{b_worst.strip() if b_worst else "（未填写）"}
- 可控变量：{b_controls.strip() if b_controls else "（未填写）"}

## 5. 目标函数（你最重视）
- {priority}
- 约束：{", ".join(constraints) if constraints else "（未选择）"}
- 后悔倾向：{regret}

## 6. 建议（工具给出的启发，不是命令）
- 结论：{rec}
- 策略：{tactic}

## 7. 下一步（48小时内）
1) 选一个「最小可行动作」（<= 30 分钟）来降低最坏结果的概率  
2) 设定一个复盘点（建议 4–6 周）  
3) 写下：你需要看到什么证据，才会改变判断
"""
    return memo

ready = st.button("✅ 生成我的决策备忘录", type="primary", use_container_width=True)

if ready:
    memo = build_memo()
    st.success("已生成。建议你先通读一遍，再复制到笔记里。")
    st.code(memo, language="markdown")
    st.download_button(
        label="⬇️ 下载为 Markdown（.md）",
        data=memo.encode("utf-8"),
        file_name="decision_memo.md",
        mime="text/markdown",
        use_container_width=True
    )

st.caption("提示：如果你希望把这份 memo 升级成“可执行路线图”，可以在此基础上加：财务 runway、技能差距、时间块计划、风险对冲方案。")
