# core/library.py
from __future__ import annotations

PLAYBOOK = [
    {
        "id": "reframe_optionality",
        "type": "reframe",
        "title": "Trade short-term uncertainty for long-term optionality",
        "text": (
            "You’re not choosing between Option A and B.\n"
            "You’re choosing whether to trade a controlled period of uncertainty\n"
            "for an increase in long-term optionality.\n"
            "So don’t decide forever today—run a reversible test."
        ),
        "tags": ["optionality", "uncertainty", "reversible", "test"],
    },
    {
        "id": "reframe_cooling",
        "type": "reframe",
        "title": "Cool the system before deciding",
        "text": (
            "If your system is overheated (sleep, conflict, overload), decisions degrade.\n"
            "First reduce volatility, then decide. Stability is a prerequisite for good judgment."
        ),
        "tags": ["stability", "overload", "cooling"],
    },
    {
        "id": "move_14d_sprint",
        "type": "move",
        "title": "14-day sprint test (reduce uncertainty by ~20%)",
        "text": (
            "Pick one test that takes <10 hours over 14 days.\n"
            "It must produce visible output (demo/write-up/interviews/portfolio).\n"
            "Continue only if evidence improves; stop/adjust if downside signals appear."
        ),
        "tags": ["sprint", "test", "evidence", "14-day"],
    },
    {
        "id": "move_info_interviews",
        "type": "move",
        "title": "2 informational interviews",
        "text": (
            "Book 2 short chats with people already doing the target role/path.\n"
            "Ask for: hiring bar, common failure modes, fastest proof, and what they'd do in your constraints."
        ),
        "tags": ["career", "interview", "evidence"],
    },
    {
        "id": "move_proof_of_work",
        "type": "move",
        "title": "Proof-of-work in <5 hours",
        "text": (
            "Build one small proof that demonstrates capability: a mini demo, a notebook, a one-page spec,\n"
            "a risk map, a portfolio piece. Something you can show."
        ),
        "tags": ["portfolio", "demo", "proof"],
    },
    {
        "id": "safeguard_downside_to_controls",
        "type": "safeguard",
        "title": "Convert worst fear into controllable variables",
        "text": (
            "Rewrite the worst fear as a controllable question:\n"
            "“What would reduce this risk by 20%?” List 3 actions and do 1 within 48 hours."
        ),
        "tags": ["risk", "control", "safeguard"],
    },
    {
        "id": "safeguard_stop_signal",
        "type": "safeguard",
        "title": "Define stop-signal early",
        "text": (
            "Define a stop condition in advance (health, performance, family stability, runway).\n"
            "If it triggers, you pause/adjust instead of pushing harder."
        ),
        "tags": ["stop", "risk", "guardrail"],
    },
]
