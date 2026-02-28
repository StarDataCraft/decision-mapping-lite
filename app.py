# app.py
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from core.engine import DecisionEngine, EngineConfig

APP_TITLE = "Decision Mapping Lite"
APP_TAGLINE = "Turn anxiety into variables. Make reversible tests. Decide with evidence."


# ----------------------------
# Helpers
# ----------------------------
def _ensure_options_state():
    if "options" not in st.session_state:
        # 仅是默认示例（不影响逻辑；你也可以删掉这些示例文本）
        st.session_state.options = [
            {
                "name": "Option A",
                "best": "If this goes well, I will gain momentum and clarity.",
                "worst": "If this goes badly, I may lose time and confidence.",
                "controls": "- Spend 6 hours/week on the core actions\n- Ship 1 small proof-of-work in 14 days\n- Do 2 short interviews with people who already did it",
            },
            {
                "name": "Option B",
                "best": "If this goes well, I will improve my platform and long-term upside.",
                "worst": "If this goes badly, I may miss windows and feel stuck.",
                "controls": "- Break into 14-day sprint\n- Track hours + output weekly\n- Keep 2 hours/week hedge for the other option",
            },
        ]


def _add_option():
    st.session_state.options.append(
        {"name": f"Option {len(st.session_state.options) + 1}", "best": "", "worst": "", "controls": ""}
    )


def _remove_option(i: int):
    if 0 <= i < len(st.session_state.options):
        st.session_state.options.pop(i)


def _fmt_options_list(options: list[dict]) -> str:
    # 给 memo 里的“备选路径”一段可读的 bullet 文本
    lines = []
    for idx, o in enumerate(options, start=1):
        nm = (o.get("name") or f"Option {idx}").strip()
        lines.append(f"- 路径 {idx}：{nm}")
    return "\n".join(lines)


def _to_payload_cn(options: list[dict]) -> dict:
    """
    将“任意条路径”映射到 engine.py 需要的字段。
    说明：你现在的 engine.py 用的是 a/b 两条路径（最小版本）。
    所以这里采用：用户填了 2+ 条时，取前两条做本轮推导。
    但 app.py 不写死“只能两条”，UI 允许用户先脑暴 3-5 条，再挑前两条做一轮 memo。
    """
    # 至少保证有两条
    while len(options) < 2:
        options.append({"name": "Option", "best": "", "worst": "", "controls": ""})

    a = options[0]
    b = options[1]

    constraints = []
    for k in ["时间", "财务", "家庭", "情绪耐受度", "健康", "签证/地域", "资源/人脉"]:
        if st.session_state.get(f"c_{k}", False):
            constraints.append(k)
    extra_constraints = (st.session_state.get("constraints_extra") or "").strip()
    if extra_constraints:
        # 允许用户用逗号/顿号分隔
        for part in extra_constraints.replace("、", ",").split(","):
            part = part.strip()
            if part and part not in constraints:
                constraints.append(part)

    payload = {
        "decision": st.session_state.get("decision", "").strip(),
        "options": _fmt_options_list(options),
        "status_6m": st.session_state.get("status_6m", "").strip(),
        "status_2y": st.session_state.get("status_2y", "").strip(),
        # A/B（取前两条路径做本轮 memo）
        "a_name": (a.get("name") or "Option A").strip(),
        "a_best": (a.get("best") or "").strip(),
        "a_worst": (a.get("worst") or "").strip(),
        "a_controls": (a.get("controls") or "").strip(),
        "b_name": (b.get("name") or "Option B").strip(),
        "b_best": (b.get("best") or "").strip(),
        "b_worst": (b.get("worst") or "").strip(),
        "b_controls": (b.get("controls") or "").strip(),
        # objective
        "priority": st.session_state.get("priority", "长期选择权（Optionality）").strip(),
        "constraints": constraints,
        "regret": st.session_state.get("regret", "没有尝试").strip(),
        # controllability + identity
        "partial_control": st.session_state.get("partial_control", "").strip(),
        "identity_anchor": st.session_state.get("identity_anchor", "").strip(),
        # evidence signals (free-form, engine 会抽 1 条最相关的)
        "evidence_to_commit": st.session_state.get("evidence_to_commit", "").strip(),
        "evidence_to_stop": st.session_state.get("evidence_to_stop", "").strip(),
    }
    return payload


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
_ensure_options_state()

st.title(APP_TITLE)
st.caption(APP_TAGLINE)

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Describe your decision")
    st.text_area(
        "Your decision (1–2 sentences)",
        key="decision",
        height=100,
        placeholder="Example: In the next 6 months, should I switch roles, keep my current path, or build something new?",
    )

    st.subheader("2) Baseline if you don't change")
    st.text_area("6 months baseline", key="status_6m", height=80, placeholder="If I do nothing, in 6 months I will likely...")
    st.text_area("2 years baseline", key="status_2y", height=80, placeholder="If I do nothing, in 2 years I will likely...")

    st.subheader("3) Objective & constraints")
    st.text_input("Your primary objective (goal function)", key="priority", value="长期选择权（Optionality）")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("时间", key="c_时间", value=True)
        st.checkbox("财务", key="c_财务", value=True)
        st.checkbox("健康", key="c_健康", value=False)
    with c2:
        st.checkbox("家庭", key="c_家庭", value=True)
        st.checkbox("情绪耐受度", key="c_情绪耐受度", value=True)
        st.checkbox("签证/地域", key="c_签证/地域", value=False)
    with c3:
        st.checkbox("资源/人脉", key="c_资源/人脉", value=False)

    st.text_input("Extra constraints (comma-separated)", key="constraints_extra", placeholder="e.g., internship contract, graduation thesis, caregiving")

    st.text_input("Regret tendency (what you'll regret most)", key="regret", value="没有尝试")

    st.subheader("4) Partially controllable levers")
    st.text_area(
        "What is partially controllable right now?",
        key="partial_control",
        height=90,
        placeholder="Examples: daily 90-min deep work block; stable sleep; weekly review; reduce conflict/overload before deciding.",
    )

    st.subheader("5) Evidence thresholds")
    st.text_area(
        "Evidence to commit (free-form lines)",
        key="evidence_to_commit",
        height=90,
        placeholder="- If I choose Option A: within 2–4 weeks, get 3 interviews OR ship 1 demo\n- If I choose Option B: 4 weeks in a row, >=25h/week + measurable score improvement",
    )
    st.text_area(
        "Stop / pivot signals (free-form lines)",
        key="evidence_to_stop",
        height=90,
        placeholder="- If Option A: 80 applications and 0 interviews -> resume/portfolio strategy is broken\n- If Option B: 2 weeks can't study due to health/anxiety -> stabilize before pushing",
    )

    st.subheader("6) Identity anchor")
    st.text_area(
        "Who are you becoming through this decision?",
        key="identity_anchor",
        height=80,
        placeholder="Example: I want to become someone who can move steadily under uncertainty, using evidence and reversible tests.",
    )

with right:
    st.subheader("Options (you can add/remove; nothing is hardcoded)")
    btns = st.columns([1, 1, 2])
    with btns[0]:
        st.button("➕ Add option", on_click=_add_option, use_container_width=True)
    with btns[1]:
        st.button("🔄 Reset", on_click=lambda: st.session_state.pop("options", None), use_container_width=True)

    st.divider()

    # render dynamic options
    for i, opt in enumerate(st.session_state.options):
        with st.container(border=True):
            top = st.columns([6, 1])
            with top[0]:
                st.text_input(f"Option name", value=opt.get("name", ""), key=f"opt_{i}_name")
            with top[1]:
                st.button("🗑️", key=f"del_{i}", on_click=_remove_option, args=(i,))

            st.text_area("Best case", value=opt.get("best", ""), key=f"opt_{i}_best", height=70)
            st.text_area("Worst case", value=opt.get("worst", ""), key=f"opt_{i}_worst", height=70)
            st.text_area("Controllable levers (bullets)", value=opt.get("controls", ""), key=f"opt_{i}_controls", height=90)

    st.divider()

    # sync UI -> state.options
    synced = []
    for i in range(len(st.session_state.options)):
        synced.append(
            {
                "name": st.session_state.get(f"opt_{i}_name", st.session_state.options[i].get("name", "")),
                "best": st.session_state.get(f"opt_{i}_best", st.session_state.options[i].get("best", "")),
                "worst": st.session_state.get(f"opt_{i}_worst", st.session_state.options[i].get("worst", "")),
                "controls": st.session_state.get(f"opt_{i}_controls", st.session_state.options[i].get("controls", "")),
            }
        )
    st.session_state.options = synced

    st.subheader("Generate")
    colA, colB = st.columns([1, 1])
    with colA:
        generate = st.button("Generate Decision Memo (CN)", type="primary", use_container_width=True)
    with colB:
        st.download_button(
            "Download inputs (JSON)",
            data=json.dumps(_to_payload_cn(st.session_state.options), ensure_ascii=False, indent=2),
            file_name=f"decision_inputs_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    if generate:
        engine = DecisionEngine(EngineConfig())
        payload = _to_payload_cn(st.session_state.options)
        memo = engine.build_memo_cn(payload)

        st.success("Generated.")
        st.text_area("Decision Memo (CN)", value=memo, height=520)

        st.download_button(
            "Download memo (Markdown)",
            data=memo.encode("utf-8"),
            file_name=f"decision_memo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.caption("Privacy: runs on Streamlit Cloud. Do not paste sensitive personal data.")
