# app_en.py
import streamlit as st
from core.engine import DecisionEngine, EngineConfig

st.set_page_config(page_title="Decision Sprint", page_icon="🧭", layout="centered", initial_sidebar_state="collapsed")

@st.cache_resource
def get_engine() -> DecisionEngine:
    return DecisionEngine(EngineConfig())

engine = get_engine()

st.title("🧭 Decision Sprint (RAG-augmented)")
st.caption("5 questions → a 14-day test plan + retrieved reframes/moves/safeguards.")

with st.expander("How it works (30s)", expanded=True):
    st.markdown(
        """
- Answer 5 questions.
- Get a **Sprint Summary** + **retrieved** moves/safeguards.
- Not a perfect-answer tool. A **move-making** tool.
"""
    )

st.divider()

decision = st.text_area("1) What's the decision you're facing?", height=80)
baseline_12m = st.text_area("2) If you do nothing, what happens in 12 months?", height=80)
best = st.text_area("3) Best plausible upside if you try?", height=80)
worst = st.text_area("4) Worst plausible downside (main fear)?", height=80)
evidence = st.text_area("5) Evidence threshold to continue?", height=90)

st.divider()

go = st.button("✅ Generate my Sprint", type="primary", use_container_width=True)

if go:
    payload = dict(
        decision=decision,
        baseline_12m=baseline_12m,
        best=best,
        worst=worst,
        evidence=evidence,
    )
    out = engine.build_sprint_en(payload)
    st.success("Generated (retrieval-augmented).")
    st.code(out, language="markdown")
    st.download_button(
        "⬇️ Download (.md)",
        data=out.encode("utf-8"),
        file_name="decision_sprint.md",
        mime="text/markdown",
        use_container_width=True,
    )
