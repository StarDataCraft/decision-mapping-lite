# core/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime

from core.library import PLAYBOOK
from core.embedder import HybridEmbedder, EmbeddingConfig


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


@dataclass
class EngineConfig:
    # 纯开源、本地跑：CPU 友好（Streamlit Cloud 推荐）
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieve_k: int = 6


class DecisionEngine:
    """
    Hybrid engine:
    - Rule-based, explainable reasoning (stable + controllable)
    - Retrieval-augmented suggestions using open-source embeddings (no API key)
      with safe fallback to TF-IDF (still no key) if the embedding model fails to load.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.embedder = HybridEmbedder(EmbeddingConfig(model_name=cfg.embed_model))

        # Prepare playbook texts for retrieval
        self._pb_texts = [
            f"{x['type']} | {x['title']}\n{x['text']}\nTags: {', '.join(x.get('tags', []))}"
            for x in PLAYBOOK
        ]

    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        k = k or self.cfg.retrieve_k
        idxs = self.embedder.top_k(query, self._pb_texts, k=k)
        return [PLAYBOOK[i] for i in idxs]

    def build_memo_cn(self, payload: Dict[str, Any]) -> str:
        # ---- unpack ----
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

        # ---- rule features ----
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

        # ---- decide ----
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

        # ---- trial pick ----
        a_manage = a_control_rich - a_worst_sev
        b_manage = b_control_rich - b_worst_sev
        if a_manage == b_manage:
            trial_pick = f"{a_name} 或 {b_name}（两者可控性接近，建议以约束与证据门槛决定）"
        else:
            trial_pick = a_name if a_manage > b_manage else b_name

        # ---- retrieval augmentation (open-source, no key) ----
        query = "\n".join(
            [
                decision,
                options,
                status_2y,
                f"{a_name} worst: {a_worst}",
                f"{b_name} worst: {b_worst}",
                f"priority: {priority}",
                f"constraints: {', '.join(constraints) if constraints else ''}",
                f"evidence: {evidence_to_commit}",
            ]
        ).strip()

        retrieved = self.retrieve(query, k=self.cfg.retrieve_k)

        # group retrieved by type
        reframes = [x for x in retrieved if x.get("type") == "reframe"][:2]
        moves = [x for x in retrieved if x.get("type") == "move"][:3]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:2]

        def fmt_cards(items: List[Dict[str, Any]]) -> str:
            if not items:
                return "（无）"
            return "\n\n".join([f"**{x['title']}**\n{x['text']}" for x in items])

        # ---- precompute (avoid f-string backslash issues) ----
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

## 8. 检索增强（开源模型）：与你这个决策最贴的重构/动作/护栏
> 这部分来自语义检索：根据你输入的语义，自动匹配行动库/重构库的最相关条目。

### 8.1 Reframes（问题重构）
{fmt_cards(reframes)}

### 8.2 Moves（可逆的小步试探）
{fmt_cards(moves)}

### 8.3 Safeguards（降低最坏结果概率）
{fmt_cards(safeguards)}

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
- 复盘问题：哪些证据支持继续加码？哪些证据支持止损或换路径？
"""
        return memo

    def build_sprint_en(self, payload: Dict[str, Any]) -> str:
        # Minimal English sprint with retrieval-augmented moves/safeguards (open-source, no key)
        decision = (payload.get("decision") or "").strip()
        baseline_12m = (payload.get("baseline_12m") or "").strip()
        best = (payload.get("best") or "").strip()
        worst = (payload.get("worst") or "").strip()
        evidence = (payload.get("evidence") or "").strip()

        query = "\n".join([decision, baseline_12m, best, worst, evidence]).strip()
        retrieved = self.retrieve(query, k=self.cfg.retrieve_k)

        moves = [x for x in retrieved if x.get("type") == "move"][:3]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:2]
        reframes = [x for x in retrieved if x.get("type") == "reframe"][:1]

        def card(items: List[Dict[str, Any]]) -> str:
            if not items:
                return "- (none)"
            return "\n".join([f"- **{x['title']}**: {x['text']}" for x in items])

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

## Reframe (retrieved)
{card(reframes)}

## Suggested moves (retrieved)
{card(moves)}

## Safeguards (retrieved)
{card(safeguards)}

---

## Next move (within 48 hours)
Pick ONE action that:
- takes < 5 hours
- produces visible output
- reduces uncertainty by ~20%

Review again in 14 days.
"""
        return out
