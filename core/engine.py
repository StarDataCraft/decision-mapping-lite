# core/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from core.library import PLAYBOOK
from core.embedder import HybridEmbedder, EmbeddingConfig
from core.reranker import rerank, RerankConfig


# ----------------- basic utils -----------------
def _lines(text: str) -> List[str]:
    if not text:
        return []
    raw = text.replace("\r", "\n")
    raw = raw.replace("•", "-").replace("—", "-")
    out = []
    for ln in raw.split("\n"):
        ln = ln.strip()
        ln = re.sub(r"^\s*[-\*\u2022]\s*", "", ln).strip()
        if ln:
            out.append(ln)
    return out


def _bulletize(text: str) -> str:
    ls = _lines(text)
    if not ls:
        return "（未填写）"
    if len(ls) == 1:
        return ls[0]
    return "\n".join([f"- {x}" for x in ls])


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


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[。.!?！？]+", text.strip())
    for p in parts:
        if p.strip():
            return p.strip()
    return text.strip()


# ----------------- "strong-ish" extractor (generic) -----------------
EXEC_VERBS = [
    # 中文常见可执行动词
    "写", "做", "完成", "提交", "投递", "复盘", "整理", "联系", "预约", "访谈", "搭建", "上线", "发布",
    "练习", "刷题", "背诵", "准备", "打样", "迭代", "优化", "验证", "测试", "记录", "输出", "对齐", "拆解",
    "评估", "比较", "设定", "执行", "检查", "跑通", "交付",
    # 英文动词（兼容英文版）
    "build", "ship", "draft", "write", "test", "iterate", "submit", "apply", "review", "schedule", "interview"
]

TIME_MARKERS = ["小时", "分钟", "天", "周", "两周", "14", "48", "month", "week", "day", "hour", "minute"]


def _concreteness_score(line: str) -> int:
    """
    更“可试探”的抓手：数字/时间/可执行动词/明确产出
    """
    if not line:
        return 0
    s = 0
    if re.search(r"\d", line):
        s += 3
    if any(t in line for t in TIME_MARKERS):
        s += 2
    if any(v in line for v in EXEC_VERBS):
        s += 2
    if any(k in line for k in ["输出", "产出", "作品", "demo", "报告", "简历", "PRD", "纪要", "文档", "交付", "commit"]):
        s += 2
    if len(line) >= 18:
        s += 1
    return s


def _pick_most_concrete(text: str) -> str:
    ls = _lines(text)
    if not ls:
        return ""
    return sorted(ls, key=_concreteness_score, reverse=True)[0]


def _extract_risk_terms(text: str, limit: int = 5) -> List[str]:
    """
    通用风险词表（可扩展）。用于兜底风险变量抽取。
    """
    vocab = [
        "压力", "焦虑", "内耗", "拖延", "失眠", "冲突", "过载",
        "窗口", "错过", "锁定", "成本", "时间不够", "钱不够", "资源不足",
        "不匹配", "不适合", "不确定", "失败", "后悔",
        "成长慢", "停滞", "没有进步", "能力差距", "门槛", "竞争"
    ]
    out = [w for w in vocab if w in (text or "")]
    return out[:limit]


def _extract_risk_variables(worst: str, baseline: str, max_n: int = 4) -> List[str]:
    """
    通用：优先 worst；不足则 baseline；再不足则风险词兜底
    """
    src = (worst or "").strip()
    if len(src) < 6:
        src = (baseline or "").strip()

    if not src:
        terms = _extract_risk_terms((worst or "") + " " + (baseline or ""))
        return terms[:max_n] if terms else []

    t = src.replace("；", "，").replace("。", "，").replace("、", "，")
    parts = [p.strip() for p in t.split("，") if p.strip()]

    cleaned = []
    for p in parts:
        p = re.sub(r"^(同时|而且|并且|以及|另外|导致|从而)", "", p).strip()
        if not p:
            continue
        if len(p) > 28:
            p = p[:28] + "…"
        if p not in cleaned:
            cleaned.append(p)

    if cleaned:
        return cleaned[:max_n]

    terms = _extract_risk_terms(src)
    return terms[:max_n]


# ----------------- semantic matcher (generic) -----------------
class _MiniTfidfMatcher:
    """
    超轻量：用于“信号行 -> 最相关路径”的匹配（不需要大模型）
    """
    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b"
        )

    def best_match(self, query: str, corpus: List[str]) -> Tuple[int, float]:
        if not query or not corpus:
            return -1, 0.0
        X = self.vec.fit_transform([query] + corpus)
        X = normalize(X, norm="l2")
        q = X[0]
        docs = X[1:]
        sims = (docs @ q.T).toarray().reshape(-1)
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])


# ----------------- option model -----------------
@dataclass
class Option:
    name: str
    best: str = ""
    worst: str = ""
    controls: str = ""

    def corpus(self) -> str:
        # 用于相似度匹配证据行：越全面越稳
        parts = [self.name, self.best, self.worst, self.controls]
        return "\n".join([p for p in parts if p])


def _risk_actions_generic(
    risks: List[str],
    option: Option,
    constraints_str: str,
    grip: str
) -> List[str]:
    """
    通用动作生成：不依赖“就业/考研/考公”等具体场景
    """
    out = []
    name = option.name or "该路径"
    grip = grip or "（你目前还没写出很具体的抓手）"

    for r in risks:
        r0 = r or ""
        # 1) 情绪/波动类
        if any(k in r0 for k in ["压力", "焦虑", "内耗", "失眠", "冲突", "过载"]):
            out.append(
                f"[{name}] 先降波动：连续 7 天固定睡眠窗口 + 每天 90 分钟深度块；在约束（{constraints_str}）下，"
                f"把重大决定降级为 14 天试探。"
            )
            continue

        # 2) 窗口/错过/成本类
        if any(k in r0 for k in ["窗口", "错过", "锁定", "成本"]):
            out.append(
                f"[{name}] 做对冲：保留每周 2 小时做“另一条备选路径”的最低维护（信息收集/联系关键人/最小产出），"
                f"避免把风险押成单点。"
            )
            continue

        # 3) 不匹配/不适合/不确定（定义清楚标准）
        if any(k in r0 for k in ["不匹配", "不适合", "不确定", "方向不清"]):
            out.append(
                f"[{name}] 定义匹配标准：写下 3 条“必须有”+3 条“最好有”，并用你的抓手去对齐其中 2 条：{grip}"
            )
            continue

        # 4) 能力差距/门槛/竞争（做 proof-of-work）
        if any(k in r0 for k in ["能力", "差距", "门槛", "竞争"]):
            out.append(
                f"[{name}] 做一个 5 小时以内可展示输出（demo/报告/一页方案/复盘），"
                f"用它换取外部反馈与真实门槛校准。"
            )
            continue

        # 5) 时间/钱/资源不足（把约束变成变量）
        if any(k in r0 for k in ["时间不够", "钱不够", "资源不足"]):
            out.append(
                f"[{name}] 把约束变量化：在（{constraints_str}）下，列出 3 个“降低 20% 成本”的动作（删、换、借、外包、缩小范围），48小时先做1个。"
            )
            continue

        # default：通用转化
        out.append(
            f"[{name}] 把“{r0}”改写成可控问题：我能做什么让它发生概率降低 20%？写 3 条动作，48小时内执行 1 条。"
        )

    # 去重 + 截断
    dedup = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup[:4]


# ----------------- Engine -----------------
@dataclass
class EngineConfig:
    retrieve_k: int = 6
    retrieve_pool: int = 40


class DecisionEngine:
    BUILD_ID = "2026-02-28-generic-slotfill-extractor-tfidf"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.embedder = HybridEmbedder(EmbeddingConfig())
        self.rerank_cfg = RerankConfig()
        self.sig_matcher = _MiniTfidfMatcher()

        self._pb_texts = [
            f"{x.get('lang','')} | {x['type']} | {x['title']}\n{x['text']}\nTags: {', '.join(x.get('tags', []))}"
            for x in PLAYBOOK
        ]

    def _retrieve_rerank(self, query: str, k: int, lang: str) -> List[Dict[str, Any]]:
        pool = min(max(self.cfg.retrieve_pool, k * 3), len(PLAYBOOK))
        idxs, scores = self.embedder.top_k_with_scores(query, self._pb_texts, k=pool)
        candidates = [PLAYBOOK[i] for i in idxs]
        reranked = rerank(query, candidates, scores, self.rerank_cfg)

        out: List[Dict[str, Any]] = []
        for item, _ in reranked:
            if item.get("lang") == lang:
                out.append(item)
            if len(out) >= k:
                break
        if len(out) < k:
            for item, _ in reranked:
                if item not in out:
                    out.append(item)
                if len(out) >= k:
                    break
        return out[:k]

    def _fill_items(self, items: List[Dict[str, Any]], slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for x in items:
            y = dict(x)
            y["text_filled"] = _safe_format(y.get("text", ""), slots)
            out.append(y)
        return out

    def _best_signal_for_option(self, signal_text: str, options: List[Option], target: Option) -> str:
        """
        通用：从 signal_text 的多行里，选出“最像 target 这条路径”的 1 行
        """
        ls = _lines(signal_text)
        if not ls:
            return ""

        corp = [o.corpus() for o in options]
        # 先筛出：最像 target 的候选行（按相似度排序）
        scored: List[Tuple[str, float]] = []
        for ln in ls:
            idx, sim = self.sig_matcher.best_match(ln, corp)
            # idx 是“该行最像哪条路径”
            if idx >= 0 and options[idx].name == target.name:
                scored.append((ln, sim))

        if scored:
            scored.sort(key=lambda x: (x[1], _concreteness_score(x[0])), reverse=True)
            return scored[0][0]

        # 若没有行明显对齐 target，则退化：选最具体的一条（仍然可用）
        return _pick_most_concrete(signal_text)

    def build_memo_cn(self, payload: Dict[str, Any]) -> str:
        # --- read inputs ---
        decision = (payload.get("decision") or "").strip()
        options_text = payload.get("options") or ""
        status_6m = payload.get("status_6m") or ""
        status_2y = payload.get("status_2y") or ""

        a = Option(
            name=payload.get("a_name") or "路径A",
            best=payload.get("a_best") or "",
            worst=payload.get("a_worst") or "",
            controls=payload.get("a_controls") or ""
        )
        b = Option(
            name=payload.get("b_name") or "路径B",
            best=payload.get("b_best") or "",
            worst=payload.get("b_worst") or "",
            controls=payload.get("b_controls") or ""
        )
        opts = [a, b]

        priority = payload.get("priority") or "长期选择权（Optionality）"
        constraints = payload.get("constraints") or []
        regret = payload.get("regret") or "没有尝试"

        partial_control = payload.get("partial_control") or ""
        identity_anchor = payload.get("identity_anchor") or ""

        evidence_to_commit = payload.get("evidence_to_commit") or ""
        evidence_to_stop = payload.get("evidence_to_stop") or ""

        constraints_str = "、".join(constraints) if constraints else "（未填写）"

        # --- grips: pick most concrete per option ---
        a_grip = _pick_most_concrete(a.controls)
        b_grip = _pick_most_concrete(b.controls)

        a_grip_score = _concreteness_score(a_grip)
        b_grip_score = _concreteness_score(b_grip)

        # --- choose trial_pick (generic) ---
        # 若差距不大：给“或”，否则选更可试探那条
        if abs(a_grip_score - b_grip_score) <= 2:
            trial_pick = f"{a.name} 或 {b.name}"
            target_option = a  # 用 a 做信号抽取的 primary（不重要，因为“或”会展示双方风险）
        else:
            target_option = a if a_grip_score > b_grip_score else b
            trial_pick = target_option.name

        # --- pick signals (generic semantic match) ---
        commit_signal_short = self._best_signal_for_option(evidence_to_commit, opts, target_option) or "（未填写继续信号）"
        stop_signal_short = self._best_signal_for_option(evidence_to_stop, opts, target_option) or "（未填写止损信号）"

        # --- risk terms (for explanation only) ---
        risk_terms = _extract_risk_terms(a.worst + " " + b.worst + " " + status_2y)
        risk_terms_str = "、".join(risk_terms) if risk_terms else "（未提取到明显风险词）"

        # --- safeguards ---
        show_both = "或" in trial_pick

        if show_both:
            a_vars = _extract_risk_variables(a.worst, status_2y, max_n=2)
            b_vars = _extract_risk_variables(b.worst, status_2y, max_n=2)
            a_actions = _risk_actions_generic(a_vars, a, constraints_str, a_grip)
            b_actions = _risk_actions_generic(b_vars, b, constraints_str, b_grip)

            risk_variables_bullets = (
                f"- [{a.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in a_vars]) + "\n"
                f"- [{b.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in b_vars])
            )
            risk_actions_bullets = (
                f"- [{a.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in a_actions[:2]]) + "\n"
                f"- [{b.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in b_actions[:2]])
            )
            safeguard_note = f"（两条路径都在试探候选里：{trial_pick}）"
        else:
            chosen = target_option
            chosen_grip = a_grip if chosen.name == a.name else b_grip
            vars_ = _extract_risk_variables(chosen.worst, status_2y, max_n=4)
            actions_ = _risk_actions_generic(vars_, chosen, constraints_str, chosen_grip)

            risk_variables_bullets = "\n".join([f"- {x}" for x in vars_]) or "（无）"
            risk_actions_bullets = "\n".join([f"- {x}" for x in actions_[:3]]) or "（无）"
            safeguard_note = f"（本次优先对齐：{trial_pick}）"

        # --- explanation paragraph (generic, but “like you”) ---
        explanation_personal = (
            "你这里最关键的信息其实不是“路径名字”，而是你写出的 **可控抓手** 与 **风险核心**。\n\n"
            f"- 你在 {a.name} 里最具体的抓手是：{a_grip if a_grip else '（未写出具体行动）'}\n"
            f"- 你在 {b.name} 里最具体的抓手是：{b_grip if b_grip else '（未写出具体行动）'}\n"
            f"- 你反复担心的风险词集中在：{risk_terms_str}\n\n"
            "所以建议之所以成立，不是因为“某条路更正确”，而是因为：\n"
            "你已经把不确定性拆成了可行动作（抓手），也识别了风险核心（你在怕什么），下一步只需要用证据门槛把试探闭环起来。\n\n"
            f"你当前约束是：{constraints_str}。在这种约束下，最划算的不是“立刻选对”，而是把决策降级为可逆实验：先用 14 天换证据，再用证据加码/止损。\n\n"
            f"- 本轮继续/加码（1条）：{commit_signal_short}\n"
            f"- 本轮止损/调整（1条）：{stop_signal_short}\n\n"
            "你要的不是一次性下注，而是让下一步更容易：证据更清晰、风险更可控、后悔更小。"
        )

        # --- retrieval query (lightweight embedding) ---
        query = "\n".join([
            decision, options_text, status_2y,
            a.name, a.best, a.worst, a.controls,
            b.name, b.best, b.worst, b.controls,
            priority, constraints_str, commit_signal_short, stop_signal_short
        ]).strip()

        retrieved = self._retrieve_rerank(query, k=self.cfg.retrieve_k, lang="cn")
        reframes = [x for x in retrieved if x.get("type") == "reframe"][:2]
        moves = [x for x in retrieved if x.get("type") == "move"][:3]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:1]
        if not safeguards:
            safeguards = [x for x in PLAYBOOK if x.get("lang") == "cn" and x.get("type") == "safeguard"][:1]

        slots = dict(
            constraints_str=constraints_str,
            trial_pick=trial_pick,
            commit_signal_short=commit_signal_short,
            stop_signal_short=stop_signal_short,
            risk_variables_bullets=risk_variables_bullets,
            risk_actions_bullets=risk_actions_bullets,
            safeguard_note=safeguard_note,
            a_name=a.name,
            b_name=b.name,
            a_worst=a.worst,
            b_worst=b.worst,
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

        memo = f"""# 决策备忘录（Decision Memo）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 你的决策
{decision if decision else "（未填写）"}

## 2. 备选路径
{_bulletize(options_text)}

## 3. 如果不改变（Baseline）
- 6个月：{status_6m.strip() if status_6m.strip() else "（未填写）"}
- 2年：{status_2y.strip() if status_2y.strip() else "（未填写）"}

## 4. 路径对比（最好/最坏/可控）
### {a.name}
- 最好：{a.best.strip() if a.best.strip() else "（未填写）"}
- 最坏：{a.worst.strip() if a.worst.strip() else "（未填写）"}
- 可控变量：
{_bulletize(a.controls)}

### {b.name}
- 最好：{b.best.strip() if b.best.strip() else "（未填写）"}
- 最坏：{b.worst.strip() if b.worst.strip() else "（未填写）"}
- 可控变量：
{_bulletize(b.controls)}

## 5. 可控 / 不可控 / 部分可控（把焦虑变成变量）
- 可控（你能直接改变的）：来自两条路径的「可控变量」清单（见上）。
- 不可控（你无法决定的）：市场/组织/他人决策/宏观环境等（你的动作是“对冲”，不是内耗）。
- 部分可控（最值得投入的）：{partial_control.strip() if partial_control.strip() else "（未填写）"}

## 6. 目标函数与约束（你在优化什么）
你最重视的是「{priority}」。你当前的约束是：{", ".join(constraints) if constraints else "（未选择）"}。你的后悔倾向是「{regret}」。

## 7. 推导链（Why）
### 7.1 本轮试探候选
- 试探路径候选：{trial_pick}

### 7.2 基于你输入的“解释段”（更像你自己的版本）
{explanation_personal}

## 8. 检索增强（两阶段：轻量embedding→轻量rerank）
> 这部分：先用 TF-IDF 混合 embedding 召回候选，再用 ngram/重叠率/类型偏置做轻量 rerank，提高贴合度与稳定性。

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
结论：更偏向「最小后悔路径（Minimum Regret Path）」：用可控的不确定性，去换取长期选择权的上升。
策略：采用「小步试探」：保留基本盘，用固定时间块推进新方向；设定 4–6 周复盘点，用证据决定是否加码。

## 11. 下一步（48小时内）
- 把最坏结果拆成 2–3 个变量，并为每个变量写一个“降低20%概率”的动作（48小时内做1个）。
- 选一个 14 天试探：<=10小时，必须产出可展示结果（作品/访谈纪要/项目）。
- 把证据门槛写短：只保留“继续信号1条 + 止损信号1条”，其余放在第9节。

## 12. 身份轨迹锚（你正在成为谁）
{identity_anchor.strip() if identity_anchor.strip() else "（未填写）"}

## 13. 复盘点（建议）
- 复盘时间：4–6 周
- 复盘问题：哪些证据支持继续加码？哪些证据支持止损？哪些信号说明应该换路径？
"""
        return memo
