# core/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime
import re

from core.library import PLAYBOOK
from core.embedder import HybridEmbedder, EmbeddingConfig


def _text_richness_score(text: str) -> int:
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
    if not worst:
        return 0
    severe_keywords = ["崩", "失业", "毁", "无法", "严重", "抑郁", "破产", "健康", "关系破裂", "绩效很差", "重病"]
    return 2 if any(k in worst for k in severe_keywords) else 1


def _bulletize(text: str) -> str:
    if not text:
        return "（未填写）"
    t = text.strip()
    lines = [ln.strip(" \t•-") for ln in t.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) <= 1:
        return t
    return "\n".join([f"- {ln}" for ln in lines])


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    raw = text.replace("\r", "").split("\n")
    out = []
    for ln in raw:
        ln = ln.strip().lstrip("•-").strip()
        if ln:
            out.append(ln)
    return out


def _concreteness_score(line: str) -> int:
    if not line:
        return 0
    s = 0
    if re.search(r"\d", line):
        s += 2
    if any(k in line for k in ["小时", "h", "周", "每天", "两周", "14 天", "48 小时", "面试", "投递", "真题", "错题", "正确率", "分数", "模拟"]):
        s += 2
    if len(line) >= 18:
        s += 1
    if any(k in line for k in ["输出", "作品", "项目", "复盘", "内推", "访谈", "计划"]):
        s += 1
    return s


def _pick_most_concrete_line(text: str) -> str:
    lines = _split_lines(text)
    if not lines:
        return ""
    scored = sorted(lines, key=lambda x: _concreteness_score(x), reverse=True)
    return scored[0]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[。.!?！？]+", text.strip())
    for p in parts:
        if p.strip():
            return p.strip()
    return text.strip()


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else (str(x) if x is not None else "")


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _safe_format(template: str, slots: Dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return template.format_map(_SafeDict({k: _safe_str(v) for k, v in slots.items()}))
    except Exception:
        return template


def _extract_risk_variables(worst_text: str) -> List[str]:
    """
    将“最坏结果”拆成 2~4 个风险变量（启发式）：
    - 先按分隔符拆
    - 再做轻量清洗
    """
    if not worst_text:
        return []
    t = worst_text.replace("；", "，").replace("。", "，").replace("、", "，")
    parts = [p.strip() for p in t.split("，") if p.strip()]
    # 去掉过长句，保留更像“变量”的短语
    cleaned = []
    for p in parts:
        p = re.sub(r"^同时|^而且|^以及|^并且", "", p).strip()
        if not p:
            continue
        # 控制长度
        if len(p) > 30:
            p = p[:30] + "…"
        if p not in cleaned:
            cleaned.append(p)
    return cleaned[:4]


def _risk_actions_for(vars_: List[str], trial_pick: str) -> List[str]:
    """
    对每个风险变量给一个“降低20%概率”的动作（rule-based mapping）。
    """
    actions = []
    for v in vars_:
        if any(k in v for k in ["不匹配", "不合适", "不对口"]):
            actions.append("用一次信息访谈 + 一次岗位JD反向拆解：把“匹配”量化为3条硬条件，48小时内改一版简历/项目对齐。")
        elif any(k in v for k in ["压力", "焦虑", "崩", "内耗"]):
            actions.append("先降波动：固定睡眠+每天90分钟深度块；把任务降级为14天试探，只做1个最小交付。")
        elif any(k in v for k in ["成长慢", "没有成长", "停滞"]):
            actions.append("给自己加一个“成长仪表盘”：每周产出1个可展示成果（报告/作品/复盘），并找1个外部反馈点。")
        elif any(k in v for k in ["成本", "转向", "窗口", "错过"]):
            actions.append("做对冲：即便主路径是“备考/学习”，也保留每周2小时的就业准备（投递/内推/作品集更新）。")
        else:
            actions.append(f"围绕“{v}”写出3个可控抓手，并在48小时内做其中1个（目标是让风险概率下降20%）。")
    # 去重 + 控制数量
    out = []
    for a in actions:
        if a not in out:
            out.append(a)
    return out[:4]


def _bullets(lines: List[str]) -> str:
    if not lines:
        return "（无）"
    return "\n".join([f"- {x}" for x in lines])


@dataclass
class EngineConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieve_k: int = 6


class DecisionEngine:
    BUILD_ID = "2026-02-28-slotfill-v2-cleaner"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.embedder = HybridEmbedder(EmbeddingConfig(model_name=cfg.embed_model))
        self._pb_texts = [
            f"{x.get('lang','')} | {x['type']} | {x['title']}\n{x['text']}\nTags: {', '.join(x.get('tags', []))}"
            for x in PLAYBOOK
        ]

    def _retrieve(self, query: str, k: int, lang: str) -> List[Dict[str, Any]]:
        idxs = self.embedder.top_k(query, self._pb_texts, k=min(max(k * 2, 8), len(PLAYBOOK)))
        candidates = [PLAYBOOK[i] for i in idxs]
        filtered = [x for x in candidates if x.get("lang") == lang]
        if len(filtered) >= k:
            return filtered[:k]
        for x in candidates:
            if x not in filtered:
                filtered.append(x)
            if len(filtered) >= k:
                break
        return filtered[:k]

    def _fill_items(self, items: List[Dict[str, Any]], slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for x in items:
            y = dict(x)
            y["text_filled"] = _safe_format(y.get("text", ""), slots)
            out.append(y)
        return out

    def build_memo_cn(self, payload: Dict[str, Any]) -> str:
        decision = (payload.get("decision") or "").strip()
        options = payload.get("options") or ""
        status_6m = payload.get("status_6m") or ""
        status_2y = payload.get("status_2y") or ""

        a_name = payload.get("a_name") or "路径A"
        a_best = payload.get("a_best") or ""
        a_worst = payload.get("a_worst") or ""
        a_controls = payload.get("a_controls") or ""

        b_name = payload.get("b_name") or "路径B"
        b_best = payload.get("b_best") or ""
        b_worst = payload.get("b_worst") or ""
        b_controls = payload.get("b_controls") or ""

        priority = payload.get("priority") or "长期选择权（Optionality）"
        constraints = payload.get("constraints") or []
        regret = payload.get("regret") or "没有尝试"

        evidence_to_commit = payload.get("evidence_to_commit") or ""
        evidence_to_stop = payload.get("evidence_to_stop") or ""
        partial_control = payload.get("partial_control") or ""
        identity_anchor = payload.get("identity_anchor") or ""

        constraints_str = "、".join(constraints) if constraints else "（未填写）"

        # ---- rule features ----
        optionality_pref = priority == "长期选择权（Optionality）"
        regret_try = regret == "没有尝试"
        baseline_lockin_risk = bool(status_2y and any(k in status_2y for k in ["窗口", "变窄", "锁定", "定型", "不可逆", "停滞"]))

        a_control_rich = _text_richness_score(a_controls)
        b_control_rich = _text_richness_score(b_controls)
        a_worst_sev = _worst_severity(a_worst)
        b_worst_sev = _worst_severity(b_worst)

        constraint_count = len(constraints) if constraints else 0
        high_constraint = constraint_count >= 3

        score_min_regret = 0
        reasons: List[str] = []

        if optionality_pref:
            score_min_regret += 2
            reasons.append("你把「长期选择权」放在首位 → 更适合优先选择能增加未来选项的策略。")
        if regret_try:
            score_min_regret += 2
            reasons.append("你的后悔倾向是「没有尝试」→ 更适合先试探、再收敛，而不是直接回避尝试。")
        if baseline_lockin_risk:
            score_min_regret += 1
            reasons.append("Baseline（2年）里出现「窗口/锁定」信号 → 不行动的机会成本偏高。")
        if max(a_control_rich, b_control_rich) >= 2:
            score_min_regret += 1
            reasons.append("至少一条路径的可控抓手足够具体 → 可以用「小步试探」把风险变成计划。")
        if high_constraint:
            score_min_regret += 1
            reasons.append(f"你当前约束较多（{constraint_count}项）→ 更适合分阶段，而不是一次性押注。")

        if score_min_regret >= 4:
            conclusion = "更偏向「最小后悔路径（Minimum Regret Path）」：用可控的不确定性，去换取长期选择权的上升。"
            strategy = "采用「小步试探」：保留基本盘，用固定时间块推进新方向；设定 4–6 周复盘点，用证据决定是否加码。"
            next48 = [
                "把最坏结果拆成 2–3 个变量，并为每个变量写一个“降低20%概率”的动作（48小时内做1个）。",
                "选一个 14 天试探：<=10小时，必须产出可展示结果（作品/访谈纪要/项目）。",
                "把证据门槛写短：只保留“继续信号1条 + 止损信号1条”，其余放在第9节。",
            ]
        else:
            conclusion = "更偏向「先降波动、再决策」：先让系统稳定，再谈方向切换。"
            strategy = "先补齐关键约束（时间块/财务/支持系统），用 2–4 周建立稳定节奏，再做第二轮决策。"
            next48 = [
                "把约束分成可控/不可控/部分可控，并挑一个部分可控项先推进。",
                "为路径 A/B 各写出缺失的 3 个关键事实（门槛/成本/回报/失败模式）。",
                "设定 2–4 周后复盘点，用新信息再生成一次 memo。",
            ]

        # ---- trial pick ----
        a_manage = a_control_rich - a_worst_sev
        b_manage = b_control_rich - b_worst_sev
        if a_manage == b_manage:
            trial_pick = f"{a_name} 或 {b_name}"
        else:
            trial_pick = a_name if a_manage > b_manage else b_name

        # ---- personalization ----
        a_best_control = _pick_most_concrete_line(a_controls)
        b_best_control = _pick_most_concrete_line(b_controls)
        risk_terms = []
        for w in ["压力", "成长", "不匹配", "窗口", "落榜", "焦虑", "家庭", "自信", "成本", "拖延", "内耗"]:
            if w in (a_worst + " " + b_worst):
                risk_terms.append(w)
        risk_terms = risk_terms[:3]
        risk_terms_str = "、".join(risk_terms) if risk_terms else "（未提取到明显风险词）"

        explanation_personal = f"""
你这里最关键的信息其实不是“路径名字”，而是你写出的 **可控抓手** 与 **风险核心**。

- 你在 {a_name} 里最具体的抓手是：{a_best_control if a_best_control else "（未写出具体行动）"}
- 你在 {b_name} 里最具体的抓手是：{b_best_control if b_best_control else "（未写出具体行动）"}
- 你反复担心的风险词集中在：{risk_terms_str}

所以建议之所以成立，不是因为“某条路更正确”，而是因为：
你已经把不确定性拆成了可行动作（抓手），也识别了风险核心（你在怕什么），下一步只需要用证据门槛把试探闭环起来。
""".strip()

        # ---- NEW: short signals for slot-filling ----
        commit_signal_short = _pick_most_concrete_line(evidence_to_commit) or "（未填写继续信号）"
        stop_signal_short = _pick_most_concrete_line(evidence_to_stop) or "（未填写止损信号）"

        # ---- NEW: risk variables + actions ----
        worst_focus = _first_sentence(a_worst) if a_worst.strip() else _first_sentence(b_worst)
        worst_focus = worst_focus or "（未填写最坏结果）"
        risk_vars = _extract_risk_variables(worst_focus)
        risk_actions = _risk_actions_for(risk_vars, trial_pick)
        risk_variables_bullets = _bullets(risk_vars)
        risk_actions_bullets = _bullets(risk_actions)

        # ---- retrieval ----
        query = "\n".join(
            [decision, options, status_2y, a_worst, b_worst, priority, constraints_str, commit_signal_short]
        ).strip()

        retrieved = self._retrieve(query, k=self.cfg.retrieve_k, lang="cn")
        reframes = [x for x in retrieved if x.get("type") == "reframe"][:2]
        moves = [x for x in retrieved if x.get("type") == "move"][:3]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:2]

        slots = dict(
            decision=decision,
            options=options,
            status_6m=status_6m,
            status_2y=status_2y,
            a_name=a_name,
            b_name=b_name,
            a_best=a_best,
            b_best=b_best,
            a_worst=a_worst,
            b_worst=b_worst,
            constraints_str=constraints_str,
            priority=priority,
            regret=regret,
            evidence_to_commit=evidence_to_commit,
            evidence_to_stop=evidence_to_stop,
            trial_pick=trial_pick,
            worst_focus=worst_focus,
            # new slots
            commit_signal_short=commit_signal_short,
            stop_signal_short=stop_signal_short,
            risk_variables_bullets=risk_variables_bullets,
            risk_actions_bullets=risk_actions_bullets,
        )

        reframes_f = self._fill_items(reframes, slots)
        moves_f = self._fill_items(moves, slots)
        safeguards_f = self._fill_items(safeguards, slots)

        def fmt_cards(items: List[Dict[str, Any]]) -> str:
            if not items:
                return "（无）"
            blocks = []
            for x in items:
                blocks.append(f"**{x.get('title','')}**\n{x.get('text_filled','')}")
            return "\n\n".join(blocks)

        reasons_bullets = _bulletize("\n".join(reasons))
        next48_bullets = _bulletize("\n".join(next48))

        control_section = f"""
## 5. 可控 / 不可控 / 部分可控（把焦虑变成变量）
- 可控（你能直接改变的）：来自两条路径的「可控变量」清单（见上）。
- 不可控（你无法决定的）：市场/组织/他人决策/宏观环境等（你的动作是“对冲”，不是内耗）。
- 部分可控（最值得投入的）：{partial_control.strip() if partial_control.strip() else "（未填写）"}
""".strip()

        memo = f"""# 决策备忘录（Decision Memo）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 你的决策
{decision if decision else "（未填写）"}

## 2. 备选路径
{_bulletize(options)}

## 3. 如果不改变（Baseline）
- 6个月：{status_6m.strip() if status_6m.strip() else "（未填写）"}
- 2年：{status_2y.strip() if status_2y.strip() else "（未填写）"}

## 4. 路径对比（最好/最坏/可控）
### {a_name}
- 最好：{a_best.strip() if a_best.strip() else "（未填写）"}
- 最坏：{a_worst.strip() if a_worst.strip() else "（未填写）"}
- 可控变量：
{_bulletize(a_controls)}

### {b_name}
- 最好：{b_best.strip() if b_best.strip() else "（未填写）"}
- 最坏：{b_worst.strip() if b_worst.strip() else "（未填写）"}
- 可控变量：
{_bulletize(b_controls)}

{control_section}

## 6. 目标函数与约束（你在优化什么）
你最重视的是「{priority}」。你当前的约束是：{", ".join(constraints) if constraints else "（未选择）"}。你的后悔倾向是「{regret}」。

## 7. 推导链（Why）
### 7.1 关键指标
- 最小后悔倾向得分：{score_min_regret}（阈值：4+ 更倾向「试探型最小后悔路径」）
- 可控性（richness，0~3）：{a_name}={a_control_rich}，{b_name}={b_control_rich}
- 最坏结果严重度（0~2）：{a_name}={a_worst_sev}，{b_name}={b_worst_sev}
- 试探路径候选：{trial_pick}

### 7.2 触发的依据
{reasons_bullets}

### 7.3 基于你输入的“解释段”（更像你自己的版本）
{explanation_personal}

## 8. 检索增强（半模板 slot-filling）：更贴你的语境的重构/动作/护栏
> 这部分来自语义检索 + 半模板填槽：避免大段抄写，优先抽“最关键的一条继续信号/止损信号”并把最坏结果拆成变量。

### 8.1 Reframes（问题重构）
{fmt_cards(reframes_f)}

### 8.2 Moves（可逆的小步试探）
{fmt_cards(moves_f)}

### 8.3 Safeguards（降低最坏结果概率）
{fmt_cards(safeguards_f)}

## 9. 证据门槛（完整版）
- 加码/承诺的证据：{evidence_to_commit.strip() if evidence_to_commit.strip() else "（未填写）"}
- 止损/换路径的信号：{evidence_to_stop.strip() if evidence_to_stop.strip() else "（未填写）"}

## 10. 建议（可执行版本）
结论：{conclusion}

策略：{strategy}

## 11. 下一步（48小时内）
{next48_bullets}

## 12. 身份轨迹锚（你正在成为谁）
{identity_anchor.strip() if identity_anchor.strip() else "（未填写）"}

## 13. 复盘点（建议）
- 复盘时间：4–6 周
- 复盘问题：哪些证据支持继续加码？哪些证据支持止损？哪些信号说明应该换路径？
"""
        return memo
