# core/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
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


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return len(x.strip()) == 0
    if isinstance(x, list):
        return len(x) == 0
    return False


# ----------------- extractor primitives -----------------
EXEC_VERBS = [
    # CN
    "写", "做", "完成", "提交", "投递", "复盘", "整理", "联系", "预约", "访谈", "搭建", "上线", "发布",
    "练习", "刷题", "背诵", "准备", "迭代", "优化", "验证", "测试", "记录", "输出", "对齐", "拆解",
    "评估", "比较", "设定", "执行", "检查", "跑通", "交付",
    # EN
    "build", "ship", "draft", "write", "test", "iterate", "submit", "apply", "review",
    "schedule", "interview", "deliver", "publish"
]

TIME_MARKERS = [
    "小时", "分钟", "天", "周", "两周", "14", "48",
    "month", "week", "day", "hour", "minute", "hrs", "hr"
]


def _concreteness_score(line: str) -> int:
    """
    更“可试探”的抓手：数字/时间/可执行动词/明确产出
    """
    if not line:
        return 0
    s = 0
    if re.search(r"\d", line):
        s += 3
    if any(t in line.lower() for t in [m.lower() for m in TIME_MARKERS]):
        s += 2
    if any(v.lower() in line.lower() for v in EXEC_VERBS):
        s += 2
    if any(k.lower() in line.lower() for k in ["输出", "产出", "作品", "demo", "报告", "简历", "prd", "纪要", "文档", "交付", "commit", "portfolio", "proof"]):
        s += 2
    if len(line) >= 18:
        s += 1
    return s


def _pick_most_concrete(text: str) -> str:
    ls = _lines(text)
    if not ls:
        return ""
    return sorted(ls, key=_concreteness_score, reverse=True)[0]


def _extract_risk_terms(text: str, limit: int = 6) -> List[str]:
    """
    用于兜底：当无法抽到好的“名词变量”时，从常见风险词里找
    """
    vocab = [
        "压力", "焦虑", "内耗", "拖延", "失眠", "冲突", "过载",
        "窗口", "错过", "锁定", "成本", "时间不够", "钱不够", "资源不足",
        "不匹配", "不适合", "不确定", "失败", "后悔",
        "成长慢", "停滞", "没有进步", "能力差距", "门槛", "竞争",
        # EN
        "stress", "anxiety", "burnout", "miss", "window", "stuck", "lose", "confidence", "time"
    ]
    txt = (text or "")
    lo = txt.lower()
    out = []
    for w in vocab:
        if w.lower() in lo and w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


# ----------------- NEW: risk -> noun variables (CN/EN) -----------------
_EN_PATTERNS = [
    (re.compile(r"\blose\s+([a-z][a-z\s\-]{2,40})", re.I), "lose"),
    (re.compile(r"\bmiss\s+([a-z][a-z\s\-]{2,40})", re.I), "miss"),
    (re.compile(r"\bfeel\s+([a-z][a-z\s\-]{2,40})", re.I), "feel"),
    (re.compile(r"\bget\s+stuck\b", re.I), "stuck"),
    (re.compile(r"\bbe\s+stuck\b", re.I), "stuck"),
    (re.compile(r"\bburn\s*out\b", re.I), "burnout"),
]

def _clean_en_obj(s: str) -> str:
    s = re.sub(r"[\.,;:!?\)\]]+$", "", s.strip())
    s = re.sub(r"\s+", " ", s)
    # 截断太长
    if len(s) > 36:
        s = s[:36].rstrip() + "…"
    return s

def _en_obj_to_cn_var(kind: str, obj: str) -> str:
    obj = _clean_en_obj(obj)
    obj_lo = obj.lower()

    # 高频映射：让它“像你”
    if kind == "lose":
        if "time" in obj_lo:
            return "时间被吞掉（但没有产出）"
        if "confidence" in obj_lo:
            return "信心被磨损（但没有反馈）"
        if "momentum" in obj_lo:
            return "势能流失（但没有闭环）"
        if "money" in obj_lo or "cash" in obj_lo:
            return "现金流被消耗（但没有换到证据）"
        return f"关键资源流失：{obj}（但没有换到证据）"

    if kind == "miss":
        if "window" in obj_lo or "windows" in obj_lo:
            return "窗口错过（但没有对冲）"
        if "opportunity" in obj_lo:
            return "机会错过（但没有对冲）"
        return f"窗口/机会错过：{obj}（但没有对冲）"

    if kind == "feel":
        if "stuck" in obj_lo:
            return "卡住（但没有实验）"
        if "overwhelmed" in obj_lo or "overload" in obj_lo:
            return "系统过热（但仍在硬扛）"
        if "anxious" in obj_lo or "anxiety" in obj_lo:
            return "焦虑升高（但没有结构）"
        return f"主观痛感升高：{obj}（但没有动作化）"

    if kind == "stuck":
        return "卡住（但没有实验）"

    if kind == "burnout":
        return "过载/倦怠（但没有降波动）"

    return f"风险变量：{kind} {obj}"


def _extract_risk_variables_nounish(worst: str, baseline: str, max_n: int = 4) -> List[str]:
    """
    目标：不要截断残句，尽量产出“名词变量”，更像你平时写 memo 的口吻。
    优先：英文 object 抽取 -> 中文拆句 -> 兜底风险词
    """
    src = (worst or "").strip()
    if len(src) < 6:
        src = (baseline or "").strip()
    src2 = ((worst or "") + "\n" + (baseline or "")).strip()
    if not src2:
        return []

    out: List[str] = []

    # 1) EN: object extraction
    lo = src2.lower()
    if re.search(r"[a-z]", lo):
        for pat, kind in _EN_PATTERNS:
            for m in pat.finditer(src2):
                if kind in ("stuck", "burnout"):
                    var = _en_obj_to_cn_var(kind, "")
                else:
                    obj = m.group(1)
                    var = _en_obj_to_cn_var(kind, obj)
                if var and var not in out:
                    out.append(var)
                if len(out) >= max_n:
                    return out[:max_n]

    # 2) CN: split + normalize into variable-like phrases
    cn_src = src2.replace("；", "，").replace("。", "，").replace("、", "，")
    parts = [p.strip() for p in cn_src.split("，") if p.strip()]
    for p in parts:
        p = re.sub(r"^(同时|而且|并且|以及|另外|导致|从而|如果|若)\s*", "", p).strip()
        if not p:
            continue
        # 把动词句拉成变量（轻量规则）
        if any(k in p for k in ["错过", "窗口", "锁定"]):
            var = "窗口错过（但没有对冲）"
        elif any(k in p for k in ["压力", "焦虑", "内耗", "失眠", "冲突", "过载"]):
            var = "系统过热（但仍在硬扛）"
        elif any(k in p for k in ["不匹配", "不适合"]):
            var = "匹配度不足（但标准未定义）"
        elif any(k in p for k in ["成本", "代价"]):
            var = "成本上升（但没有换到证据）"
        elif any(k in p for k in ["成长慢", "停滞", "没有进步"]):
            var = "成长停滞（但缺少反馈回路）"
        else:
            # 截断但不残缺：尽量保留成分
            var = p
            if len(var) > 28:
                var = var[:28].rstrip() + "…"
        if var and var not in out:
            out.append(var)
        if len(out) >= max_n:
            return out[:max_n]

    # 3) fallback terms
    terms = _extract_risk_terms(src2, limit=max_n)
    for t in terms:
        if t not in out:
            out.append(t)
        if len(out) >= max_n:
            break

    return out[:max_n]


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
        parts = [self.name, self.best, self.worst, self.controls]
        return "\n".join([p for p in parts if p])


# ----------------- NEW: evidence auto fallback -----------------
def _parse_controls_signals(controls_text: str) -> Dict[str, Any]:
    """
    从 controls 里抽出最小可用的“证据结构”
    返回：
      hours_per_week: Optional[int]
      days: Optional[int]
      outputs: Optional[int]
      has_proof: bool
    """
    txt = (controls_text or "")
    lo = txt.lower()

    # hours/week
    hours = None
    m = re.search(r"(\d+)\s*(?:hours|hour|hrs|hr)\s*/?\s*(?:week|wk)", lo)
    if not m:
        m = re.search(r"每周\s*(\d+)\s*小时", txt)
    if m:
        try:
            hours = int(m.group(1))
        except Exception:
            hours = None

    # days window
    days = None
    m2 = re.search(r"(\d+)\s*(?:days|day)", lo)
    if not m2:
        m2 = re.search(r"(\d+)\s*天", txt)
    if m2:
        try:
            days = int(m2.group(1))
        except Exception:
            days = None

    # outputs / proof-of-work
    outputs = None
    m3 = re.search(r"ship\s+(\d+)\s+", lo)
    if not m3:
        m3 = re.search(r"(\d+)\s*(?:small\s+)?(?:proof|demo|deliverable|output)", lo)
    if not m3:
        m3 = re.search(r"产出\s*(\d+)", txt)
    if m3:
        try:
            outputs = int(m3.group(1))
        except Exception:
            outputs = None

    has_proof = any(k in lo for k in ["proof", "demo", "deliverable", "portfolio", "output", "ship"]) or any(
        k in txt for k in ["作品", "demo", "产出", "交付", "输出"]
    )

    return {
        "hours_per_week": hours,
        "days": days,
        "outputs": outputs,
        "has_proof": has_proof
    }


def _auto_evidence_fallback(option: Option, default_days: int = 14) -> Tuple[str, str]:
    """
    当用户没填 evidence_to_commit / evidence_to_stop 时，用 controls 自动生成一对最小门槛。
    """
    info = _parse_controls_signals(option.controls)
    days = info["days"] or default_days
    hours = info["hours_per_week"]
    outputs = info["outputs"]
    has_proof = info["has_proof"]

    # continue
    parts = []
    if has_proof:
        if outputs is not None:
            parts.append(f"{days} 天内完成 {outputs} 个可展示产出（demo/报告/方案/复盘）")
        else:
            parts.append(f"{days} 天内完成 1 个可展示产出（demo/报告/方案/复盘）")
    if hours is not None:
        parts.append(f"每周兑现 {hours} 小时核心动作")
    if not parts:
        parts = [f"{days} 天内产出 1 个可展示结果 + 每周兑现固定时间块"]

    commit = "；".join(parts)

    # stop
    stop_parts = []
    if has_proof:
        stop_parts.append(f"连续 {days} 天无法产出可展示结果")
    if hours is not None:
        stop_parts.append(f"连续 2 周无法兑现每周 {hours} 小时")
    if not stop_parts:
        stop_parts = [f"连续 {days} 天没有输出/没有反馈"]
    stop = "；".join(stop_parts) + " → 降级抓手/缩小实验/换验证方式"

    return commit, stop


# ----------------- risk actions generic (unchanged, but will receive better risks) -----------------
def _risk_actions_generic(
    risks: List[str],
    option: Option,
    constraints_str: str,
    grip: str
) -> List[str]:
    out = []
    name = option.name or "该路径"
    grip = grip or "（你目前还没写出很具体的抓手）"

    for r in risks:
        r0 = r or ""

        # 1) 系统过热类
        if any(k in r0 for k in ["系统过热", "压力", "焦虑", "内耗", "失眠", "冲突", "过载", "倦怠"]):
            out.append(
                f"[{name}] 先降波动：连续 7 天固定睡眠窗口 + 每天 90 分钟深度块；在约束（{constraints_str}）下，把重大决定降级为 14 天试探。"
            )
            continue

        # 2) 窗口/机会类
        if any(k in r0 for k in ["窗口错过", "机会错过", "窗口", "错过", "锁定"]):
            out.append(
                f"[{name}] 做对冲：保留每周 2 小时做“另一条备选路径”的最低维护（信息收集/联系关键人/最小产出），避免把风险押成单点。"
            )
            continue

        # 3) 匹配/标准类
        if any(k in r0 for k in ["匹配度不足", "不匹配", "不适合", "标准未定义"]):
            out.append(
                f"[{name}] 定义匹配标准：写下 3 条“必须有”+3 条“最好有”，并用你的抓手去对齐其中 2 条：{grip}"
            )
            continue

        # 4) 证据/输出类
        if any(k in r0 for k in ["没有产出", "没有反馈", "缺少反馈", "势能流失", "资源流失", "成本上升"]):
            out.append(
                f"[{name}] 建立证据回路：在 14 天内做 1 个可展示输出，并安排 1 个外部反馈点（访谈/评审/投稿/面试），把“感觉”变成“证据”。"
            )
            continue

        # 5) default
        out.append(
            f"[{name}] 把“{r0}”改写成可控问题：我能做什么让它发生概率降低 20%？写 3 条动作，48 小时内执行 1 条。"
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
    BUILD_ID = "2026-02-28-v2-diagnostic-explainer-nounrisk-autoevidence"

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
        ls = _lines(signal_text)
        if not ls:
            return ""

        corp = [o.corpus() for o in options]
        scored: List[Tuple[str, float]] = []
        for ln in ls:
            idx, sim = self.sig_matcher.best_match(ln, corp)
            if idx >= 0 and options[idx].name == target.name:
                scored.append((ln, sim))

        if scored:
            scored.sort(key=lambda x: (x[1], _concreteness_score(x[0])), reverse=True)
            return scored[0][0]

        return _pick_most_concrete(signal_text)

    # -------- NEW: diagnostic explanation (variable structure) --------
    def _build_explanation(self,
                           decision: str,
                           status_6m: str,
                           status_2y: str,
                           evidence_to_commit: str,
                           evidence_to_stop: str,
                           constraints_str: str,
                           a: Option, b: Option,
                           a_grip: str, b_grip: str,
                           commit_signal_short: str,
                           stop_signal_short: str,
                           risk_terms_str: str,
                           trial_pick: str) -> str:
        missing = []
        if _is_missing(decision):
            missing.append("决策（你到底要决定什么）")
        if _is_missing(status_6m) and _is_missing(status_2y):
            missing.append("Baseline（不改变会怎样：6个月/2年）")
        if _is_missing(evidence_to_commit) and _is_missing(evidence_to_stop):
            missing.append("证据门槛（继续/止损信号）")

        # 1) diagnostic mode if major fields missing
        if missing:
            todo = []
            # 48h补齐动作：给具体且短的填法
            if "决策（你到底要决定什么）" in missing:
                todo.append("写一句话决策：‘我在 6–12 个月内是否要 ____？’（只写1句）")
            if "Baseline（不改变会怎样：6个月/2年）" in missing:
                todo.append("补两句 Baseline：‘6个月后最可能 ____’ + ‘2年后最可能 ____’（各1句）")
            if "证据门槛（继续/止损信号）" in missing:
                todo.append(f"先不用完美：直接采用系统兜底门槛（已生成），或者你手写‘继续=____；止损=____’各1条")

            todo_str = "\n".join([f"- {x}" for x in todo]) if todo else "-（无）"

            return (
                "### 7.2 诊断式解释（因为输入缺口而不是因为你不够聪明）\n"
                f"你现在的问题**不是选项本身**，而是关键信息缺失：{ '、'.join(missing) }。\n"
                "在这种情况下，任何“建议”都会变得像模板，因为它缺少你自己的证据与语境。\n\n"
                "**48 小时补齐动作（越短越好）**\n"
                f"{todo_str}\n\n"
                "补齐后，这份 memo 会从“通用正确话”变成“只对你成立的推导”。\n\n"
                "### 7.3 仍然可用的最小推导（不等你想清楚）\n"
                "你这里最关键的信息其实不是“路径名字”，而是你写出的 **可控抓手** 与 **风险核心**。\n\n"
                f"- {a.name} 最具体抓手：{a_grip if a_grip else '（未写出具体行动）'}\n"
                f"- {b.name} 最具体抓手：{b_grip if b_grip else '（未写出具体行动）'}\n"
                f"- 当前可见的风险核心：{risk_terms_str}\n\n"
                f"在约束（{constraints_str}）下，本轮最划算的动作是把决策降级为可逆实验：先做 14 天试探（{trial_pick}），用证据再加码/止损。\n"
                f"- 继续/加码（1条）：{commit_signal_short}\n"
                f"- 止损/调整（1条）：{stop_signal_short}\n"
            )

        # 2) normal explanation mode (richer, but not fixed-shape)
        # 用“抓手差异/风险词/约束+证据”动态组织
        parts = []
        parts.append("你这里最关键的信息其实不是“路径名字”，而是你写出的 **可控抓手** 与 **风险核心**。")
        parts.append(f"- 你在 {a.name} 里最具体的抓手是：{a_grip if a_grip else '（未写出具体行动）'}")
        parts.append(f"- 你在 {b.name} 里最具体的抓手是：{b_grip if b_grip else '（未写出具体行动）'}")
        parts.append(f"- 你反复担心的风险核心：{risk_terms_str}")

        # 约束 -> 结构
        parts.append(
            f"你当前约束是：{constraints_str}。在这种约束下，最划算的不是“立刻选对”，而是把决策降级为可逆实验：先用 14 天换证据，再用证据加码/止损。"
        )
        # 证据 -> 闭环
        parts.append(f"- 本轮继续/加码（1条）：{commit_signal_short}")
        parts.append(f"- 本轮止损/调整（1条）：{stop_signal_short}")
        parts.append("你要的不是一次性下注，而是让下一步更容易：证据更清晰、风险更可控、后悔更小。")

        return "### 7.2 基于你输入的“解释段”（可变结构）\n" + "\n".join(parts)

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

        # --- choose trial_pick ---
        if abs(a_grip_score - b_grip_score) <= 2:
            trial_pick = f"{a.name} 或 {b.name}"
            target_option = a
        else:
            target_option = a if a_grip_score > b_grip_score else b
            trial_pick = target_option.name

        # --- signals (with auto fallback) ---
        commit_signal_short = self._best_signal_for_option(evidence_to_commit, opts, target_option)
        stop_signal_short = self._best_signal_for_option(evidence_to_stop, opts, target_option)

        # Auto-generate minimal evidence if missing
        if not commit_signal_short and not stop_signal_short:
            auto_commit, auto_stop = _auto_evidence_fallback(target_option, default_days=14)
            commit_signal_short = auto_commit
            stop_signal_short = auto_stop

        if not commit_signal_short:
            auto_commit, _ = _auto_evidence_fallback(target_option, default_days=14)
            commit_signal_short = auto_commit
        if not stop_signal_short:
            _, auto_stop = _auto_evidence_fallback(target_option, default_days=14)
            stop_signal_short = auto_stop

        # --- risk terms (for explanation only) ---
        risk_terms = _extract_risk_terms(a.worst + " " + b.worst + " " + status_2y)
        risk_terms_str = "、".join(risk_terms) if risk_terms else "（当前输入里风险词很泛/很少，建议把 worst 写具体一点）"

        # --- safeguards (better noun variables) ---
        show_both = "或" in trial_pick

        if show_both:
            a_vars = _extract_risk_variables_nounish(a.worst, status_2y, max_n=2)
            b_vars = _extract_risk_variables_nounish(b.worst, status_2y, max_n=2)
            a_actions = _risk_actions_generic(a_vars, a, constraints_str, a_grip)
            b_actions = _risk_actions_generic(b_vars, b, constraints_str, b_grip)

            risk_variables_bullets = (
                f"- [{a.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in (a_vars or ["（未抽到可用变量：请把 worst 写具体）"])]) + "\n"
                f"- [{b.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in (b_vars or ["（未抽到可用变量：请把 worst 写具体）"])])
            )
            risk_actions_bullets = (
                f"- [{a.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in (a_actions[:2] or ["-（无）"])]) + "\n"
                f"- [{b.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in (b_actions[:2] or ["-（无）"])])
            )
            safeguard_note = f"（两条路径都在试探候选里：{trial_pick}）"
            worst_hint = f"[{a.name}] {a.worst.strip() if a.worst.strip() else '（未填写）'}；[{b.name}] {b.worst.strip() if b.worst.strip() else '（未填写）'}"
        else:
            chosen = target_option
            chosen_grip = a_grip if chosen.name == a.name else b_grip
            vars_ = _extract_risk_variables_nounish(chosen.worst, status_2y, max_n=4)
            actions_ = _risk_actions_generic(vars_, chosen, constraints_str, chosen_grip)

            risk_variables_bullets = "\n".join([f"- {x}" for x in (vars_ or ["（未抽到可用变量：请把 worst 写具体）"])])
            risk_actions_bullets = "\n".join([f"- {x}" for x in (actions_[:3] or ["（无）"])])
            safeguard_note = f"（本次优先对齐：{trial_pick}）"
            worst_hint = chosen.worst.strip() if chosen.worst.strip() else "（未填写）"

        # --- explanation paragraph (variable structure) ---
        explanation_personal = self._build_explanation(
            decision=decision,
            status_6m=status_6m,
            status_2y=status_2y,
            evidence_to_commit=evidence_to_commit,
            evidence_to_stop=evidence_to_stop,
            constraints_str=constraints_str,
            a=a, b=b,
            a_grip=a_grip,
            b_grip=b_grip,
            commit_signal_short=commit_signal_short,
            stop_signal_short=stop_signal_short,
            risk_terms_str=risk_terms_str,
            trial_pick=trial_pick
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
            worst_hint=worst_hint,
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
- 加码/承诺的证据：{evidence_to_commit.strip() if evidence_to_commit.strip() else "（未填写：已用 controls 自动生成最小继续信号）"}
- 止损/换路径的信号：{evidence_to_stop.strip() if evidence_to_stop.strip() else "（未填写：已用 controls 自动生成最小止损信号）"}

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
