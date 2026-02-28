# core/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime
import re

from core.library import PLAYBOOK
from core.embedder import HybridEmbedder, EmbeddingConfig


# ---------------------------
# Small utilities
# ---------------------------
def _text_richness_score(text: str) -> int:
    """粗略判断描述是否具体（越具体越可推导）。0~3"""
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


def _bulletize(text: str) -> str:
    """把多行文本格式化成 markdown 列表"""
    if not text:
        return "（未填写）"
    t = text.strip()
    lines = [ln.strip(" \t•-") for ln in t.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) <= 1:
        return t
    return "\n".join([f"- {ln}" for ln in lines])


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else (str(x) if x is not None else "")


class _SafeDict(dict):
    """format_map safe dict: missing keys -> empty string"""
    def __missing__(self, key):
        return ""


def _safe_format(template: str, slots: Dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return template.format_map(_SafeDict({k: _safe_str(v) for k, v in slots.items()}))
    except Exception:
        return template


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
    """Prefer lines with numbers/time/explicit constraints; longer also helps."""
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


def _pick_most_concrete_control(controls_text: str) -> str:
    """Pick the single most concrete action line from controls."""
    lines = _split_lines(controls_text)
    if not lines:
        return ""
    scored = sorted(lines, key=lambda x: _concreteness_score(x), reverse=True)
    return scored[0]


def _extract_risk_keywords(text: str, top_n: int = 3) -> List[str]:
    """
    Simple keyword extraction without ML:
    - A tiny curated list + light frequency chunks
    """
    if not text:
        return []
    t = re.sub(r"[，。；、,.!?:：\n\r\t]+", " ", text).strip()

    curated = ["压力", "成长", "不匹配", "错过", "窗口", "失败", "落榜", "焦虑", "家庭", "自信", "成本", "拖延", "内耗"]
    hits = [w for w in curated if w in text]

    chunks = re.findall(r"[\u4e00-\u9fff]{2,8}", t)
    stop = {"可能", "同时", "但是", "因为", "如果", "这样", "这个", "那个", "自己", "很多", "然后"}
    chunks = [c for c in chunks if c not in stop]

    freq: Dict[str, int] = {}
    for c in chunks:
        freq[c] = freq.get(c, 0) + 1

    out: List[str] = []
    for h in hits:
        if h not in out:
            out.append(h)

    sorted_chunks = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    for c, _ in sorted_chunks:
        if c not in out:
            out.append(c)
        if len(out) >= top_n:
            break

    return out[:top_n]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[。.!?！？]+", text.strip())
    for p in parts:
        if p.strip():
            return p.strip()
    return text.strip()


# ---------------------------
# Engine
# ---------------------------
@dataclass
class EngineConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieve_k: int = 6


class DecisionEngine:
    BUILD_ID = "2026-02-28-slotfill-v1"

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
            reasons.append("Baseline（2年）里出现「路径锁定/窗口变窄」信号 → 不行动的机会成本偏高。")
        if max(a_control_rich, b_control_rich) >= 2:
            score_min_regret += 1
            reasons.append("至少一条路径的可控抓手足够具体 → 可以用「小步试探」把风险变成计划。")
        else:
            reasons.append("两条路径的可控抓手都偏模糊 → 先补信息/资源，再做大动作会更稳。")
        if high_constraint:
            score_min_regret += 1
            reasons.append(f"你当前约束较多（{constraint_count}项）→ 更适合分阶段，而不是一次性押注。")

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

        a_manage = a_control_rich - a_worst_sev
        b_manage = b_control_rich - b_worst_sev
        if a_manage == b_manage:
            trial_pick = f"{a_name} 或 {b_name}（两者可控性接近，建议以约束与证据门槛决定）"
        else:
            trial_pick = a_name if a_manage > b_manage else b_name

        a_best_control = _pick_most_concrete_control(a_controls)
        b_best_control = _pick_most_concrete_control(b_controls)

        worst_focus = _first_sentence(a_worst) if a_worst.strip() else _first_sentence(b_worst)
        if not worst_focus:
            worst_focus = "（未填写最坏结果）"

        risk_terms = _extract_risk_keywords((a_worst or "") + " " + (b_worst or ""), top_n=3)
        risk_terms_str = "、".join(risk_terms) if risk_terms else "（未提取到明显风险词）"

        explanation_personal = f"""
你这里最关键的信息其实不是“路径名字”，而是你写出的 **可控抓手** 与 **风险核心**。

- 你在 {a_name} 里最具体的抓手是：{a_best_control if a_best_control else "（未写出具体行动）"}
- 你在 {b_name} 里最具体的抓手是：{b_best_control if b_best_control else "（未写出具体行动）"}
- 你反复担心的风险词集中在：{risk_terms_str}

所以建议之所以成立，不是因为“某条路更正确”，而是因为：
你已经把不确定性拆成了可行动作（抓手），也识别了风险核心（你在怕什么），下一步只需要用证据门槛把试探闭环起来。
""".strip()

        query = "\n".join(
            [
                decision,
                options,
                status_2y,
                f"{a_name} worst: {a_worst}",
                f"{b_name} worst: {b_worst}",
                f"priority: {priority}",
                f"constraints: {constraints_str}",
                f"evidence: {evidence_to_commit}",
            ]
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
- 试探路径候选（可控性-最坏严重度）：{trial_pick}

### 7.2 触发的依据
{reasons_bullets}

### 7.3 基于你输入的“解释段”（更像你自己的版本）
{explanation_personal}

## 8. 检索增强（半模板 slot-filling）：更贴你的语境的重构/动作/护栏
> 这部分来自语义检索 + 半模板填槽：用你写的最坏结果/约束/证据门槛把条目“实例化”。

### 8.1 Reframes（问题重构）
{fmt_cards(reframes_f)}

### 8.2 Moves（可逆的小步试探）
{fmt_cards(moves_f)}

### 8.3 Safeguards（降低最坏结果概率）
{fmt_cards(safeguards_f)}

## 9. 证据门槛（用证据而不是情绪做决定）
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

    def build_sprint_en(self, payload: Dict[str, Any]) -> str:
        decision = (payload.get("decision") or "").strip()
        baseline_12m = (payload.get("baseline_12m") or "").strip()
        best = (payload.get("best") or "").strip()
        worst = (payload.get("worst") or "").strip()
        evidence = (payload.get("evidence") or "").strip()

        constraints_str = _safe_str(payload.get("constraints_str") or "time, money, family, emotional bandwidth")
        status_2y = _safe_str(payload.get("status_2y") or baseline_12m)

        worst_focus = _first_sentence(worst) if worst else "(empty worst-case)"

        query = "\n".join([decision, baseline_12m, best, worst, evidence]).strip()
        retrieved = self._retrieve(query, k=self.cfg.retrieve_k, lang="en")

        reframes = [x for x in retrieved if x.get("type") == "reframe"][:1]
        moves = [x for x in retrieved if x.get("type") == "move"][:2]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:1]

        slots = dict(
            decision=decision,
            status_2y=status_2y,
            constraints_str=constraints_str,
            evidence_to_commit=evidence,
            evidence_to_stop=_safe_str(payload.get("evidence_to_stop") or ""),
            trial_pick=_safe_str(payload.get("trial_pick") or "your chosen path"),
            worst_focus=worst_focus,
        )

        reframes_f = self._fill_items(reframes, slots)
        moves_f = self._fill_items(moves, slots)
        safeguards_f = self._fill_items(safeguards, slots)

        def card(items: List[Dict[str, Any]]) -> str:
            if not items:
                return "- (none)"
            return "\n".join([f"- **{x.get('title','')}**: {x.get('text_filled','')}" for x in items])

        out = f"""# Decision Sprint Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Decision
{decision or "(empty)"}

## If you do nothing (12 months)
{baseline_12m or "(empty)"}

## Best plausible upside
{best or "(empty)"}

## Worst plausible downside
{worst or "(empty)"}

## Evidence threshold (continue only if)
{evidence or "(empty)"}

---

## Reframe (slot-filled retrieval)
{card(reframes_f)}

## Suggested moves (slot-filled retrieval)
{card(moves_f)}

## Safeguard (slot-filled retrieval)
{card(safeguards_f)}

---

## Next move (within 48 hours)
Pick ONE action that:
- takes < 5 hours
- produces visible output
- reduces uncertainty by ~20%

Review again in 14 days.
"""
        return out
