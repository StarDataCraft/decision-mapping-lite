# app.py
import streamlit as st
from core.engine import DecisionEngine, EngineConfig

st.set_page_config(page_title="Decision Mapping Lite", page_icon="🧭", layout="centered")

@st.cache_resource
def get_engine() -> DecisionEngine:
    return DecisionEngine(EngineConfig())

engine = get_engine()

st.title("🧭 Decision Mapping Lite (RAG-augmented)")
st.caption("规则推导 + 语义检索增强：更少“模板感”，更贴你的决策语境。")

with st.expander("使用说明（30秒）", expanded=True):
    st.markdown(
        """
- 只聚焦一个决策。
- 尽量写具体：用事实而不是情绪形容词。
- 输出不是标准答案，而是：**推导链 + 检索增强的重构/动作/护栏 + 下一步**。
"""
    )

st.divider()

decision = st.text_area("1) 你正在面对的关键决策是什么？", height=80)
options = st.text_area("2) 你的备选路径有哪些？（至少2个）", height=110, placeholder="A: ...\nB: ...\nC: ...（可选）")
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
evidence_to_commit = st.text_area("8) 你需要看到什么证据，才会对某条路径“加码/承诺”？（证据门槛）", height=90)
evidence_to_stop = st.text_area("9) 你需要看到什么信号，才会“止损/换路径”？（止损条件）", height=90)
partial_control = st.text_area("10) 你现在最关键的「部分可控」变量是什么？你准备怎么把它往有利方向推一点？", height=90)
identity_anchor = st.text_area("11) 这个决策与你想成为的那种人有什么关系？（身份轨迹锚）", height=80)

st.divider()

go = st.button("✅ 生成我的决策备忘录", type="primary", use_container_width=True)

if go:
    payload = dict(
        decision=decision,
        options=options,
        status_6m=status_6m,
        status_2y=status_2y,
        a_name=a_name, a_best=a_best, a_worst=a_worst, a_controls=a_controls,
        b_name=b_name, b_best=b_best, b_worst=b_worst, b_controls=b_controls,
        priority=priority,
        constraints=constraints,
        regret=regret,
        evidence_to_commit=evidence_to_commit,
        evidence_to_stop=evidence_to_stop,
        partial_control=partial_control,
        identity_anchor=identity_anchor,
    )

    memo = engine.build_memo_cn(payload)
    st.success("已生成（含语义检索增强）。")
    st.code(memo, language="markdown")
    st.download_button(
        "⬇️ 下载为 Markdown（.md）",
        data=memo.encode("utf-8"),
        file_name="decision_memo.md",
        mime="text/markdown",
        use_container_width=True,
    )
