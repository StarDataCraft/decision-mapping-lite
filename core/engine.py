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
    raw = str(text).replace("\r", "\n")
    raw = raw.replace("•", "-").replace("—", "-").replace("–", "-")
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
    parts = re.split(r"[。.!?！？]+", str(text).strip())
    for p in parts:
        if p.strip():
            return p.strip()
    return str(text).strip()


def _shorten(text: str, n: int = 90) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t if len(t) <= n else (t[:n].rstrip() + "…")


# ----------------- stronger-ish extractor -----------------
EXEC_VERBS = [
    # 中文常见可执行动词
    "写", "做", "完成", "提交", "投递", "复盘", "整理", "联系", "预约", "访谈", "搭建", "上线", "发布",
    "练习", "刷题", "背诵", "准备", "打样", "迭代", "优化", "验证", "测试", "记录", "输出", "对齐", "拆解",
    "评估", "比较", "设定", "执行", "检查", "跑通", "交付", "归纳", "汇总", "制作", "起草",
    # 英文（兼容英文版）
    "build", "ship", "draft", "write", "test", "iterate", "submit", "apply", "review", "schedule", "interview",
    "prototype", "publish", "deploy"
]

TIME_MARKERS = ["小时", "分钟", "天", "周", "两周", "14", "48", "month", "week", "day", "hour", "minute"]
OUTPUT_MARKERS = ["输出", "产出", "作品", "demo", "报告", "简历", "PRD", "纪要", "文档", "交付", "commit", "push", "portfolio"]
NUMBER_PATTERN = re.compile(r"\d")


def _concreteness_score(line: str) -> int:
    """
    更“可试探”的抓手：数字/时间/可执行动词/明确产出
    """
    if not line:
        return 0
    s = 0
    if NUMBER_PATTERN.search(line):
        s += 3
    if any(t in line for t in TIME_MARKERS):
        s += 2
    if any(v in line.lower() for v in EXEC_VERBS):
        s += 2
    if any(k in line for k in OUTPUT_MARKERS):
        s += 2
    if len(line) >= 18:
        s += 1
    # 奖励“动作结构”句式（含冒号/加号/→）
    if any(sym in line for sym in ["：", ":", "+", "→", "->", "/"]):
        s += 1
    return s


def _pick_top_concrete(text: str, topn: int = 2) -> List[str]:
    ls = _lines(text)
    if not ls:
        return []
    scored = sorted([(ln, _concreteness_score(ln)) for ln in ls], key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:topn] if x[1] > 0]


def _pick_most_concrete(text: str) -> str:
    xs = _pick_top_concrete(text, topn=1)
    return xs[0] if xs else ""


def _extract_risk_terms(text: str, limit: int = 6) -> List[str]:
    """
    通用风险词表（兜底用）。不依赖分词器，强调稳定。
    """
    vocab = [
        "压力", "焦虑", "内耗", "拖延", "失眠", "冲突", "过载", "波动",
        "窗口", "错过", "锁定", "沉没成本", "成本", "机会成本",
        "不匹配", "不适合", "不确定", "失败", "后悔",
        "成长慢", "停滞", "没有进步", "能力差距", "门槛", "竞争",
        "现金流", "精力", "时间不够", "钱不够", "资源不足"
    ]
    src = (text or "")
    out = []
    for w in vocab:
        if w in src and w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _split_risk_phrases(src: str) -> List[str]:
    if not src:
        return []
    t = src.replace("；", "，").replace("。", "，").replace("、", "，").replace("\n", "，")
    parts = [p.strip() for p in t.split("，") if p.strip()]
    cleaned = []
    for p in parts:
        p = re.sub(r"^(同时|而且|并且|以及|另外|导致|从而|结果是|因此)", "", p).strip()
        if not p:
            continue
        # 去掉过泛的尾巴
        p = re.sub(r"(的风险|的问题|的情况)$", "", p).strip()
        if len(p) > 34:
            p = p[:34].rstrip() + "…"
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned


def _extract_risk_variables(worst: str, baseline: str, max_n: int = 4) -> List[str]:
    """
    通用：优先 worst；不足则 baseline；再不足则风险词兜底。
    同时更“稳”：会尝试抽出短风险短语而不是整句。
    """
    src = (worst or "").strip()
    if len(src) < 6:
        src = (baseline or "").strip()

    phrases = _split_risk_phrases(src)
    if phrases:
        return phrases[:max_n]

    # fallback: vocab terms
    terms = _extract_risk_terms((worst or "") + " " + (baseline or ""), limit=max_n)
    return terms[:max_n] if terms else []


# ----------------- lightweight playbook retriever -----------------
class PlaybookLightRetriever:
    """
    纯 TF-IDF cosine：极稳、极便宜，适合 Streamlit Cloud。
    """
    def __init__(self, pb_texts: List[str]):
        self.pb_texts = pb_texts
        self.vec = TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b"
        )
        X = self.vec.fit_transform(pb_texts)
        self.X = normalize(X, norm="l2")

    def top_k(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        if not query:
            return [], []
        q = self.vec.transform([query])
        q = normalize(q, norm="l2")
        sims = (self.X @ q.T).toarray().reshape(-1)
        if sims.size == 0:
            return [], []
        idxs = np.argsort(-sims)[:k]
        return [int(i) for i in idxs], [float(sims[i]) for i in idxs]


# ----------------- semantic matcher (tiny) -----------------
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
        r0 = (r or "").strip()
        if not r0:
            continue

        # 1) 情绪/波动类
        if any(k in r0 for k in ["压力", "焦虑", "内耗", "失眠", "冲突", "过载", "波动"]):
            out.append(
                f"[{name}] 先降波动：连续 7 天固定睡眠窗口 + 每天 90 分钟深度块；在约束（{constraints_str}）下，把重大决定降级为 14 天试探。"
            )
            continue

        # 2) 窗口/错过/锁定/成本类
        if any(k in r0 for k in ["窗口", "错过", "锁定", "沉没成本", "成本", "机会成本"]):
            out.append(
                f"[{name}] 做对冲：保留每周 2 小时做“另一条备选路径”的最低维护（信息收集/联系关键人/最小产出），避免把风险押成单点。"
            )
            continue

        # 3) 不匹配/不适合/不确定（定义标准）
        if any(k in r0 for k in ["不匹配", "不适合", "不确定", "方向不清", "看不清"]):
            out.append(
                f"[{name}] 定义匹配标准：写下 3 条“必须有”+3 条“最好有”，并用你的抓手去对齐其中 2 条：{grip}"
            )
            continue

        # 4) 能力差距/门槛/竞争（proof-of-work）
        if any(k in r0 for k in ["能力", "差距", "门槛", "竞争"]):
            out.append(
                f"[{name}] 做一个 5 小时以内可展示输出（demo/报告/一页方案/复盘），用它换取外部反馈与真实门槛校准。"
            )
            continue

        # 5) 时间/钱/资源不足（把约束变成变量）
        if any(k in r0 for k in ["时间不够", "钱不够", "资源不足", "现金流", "精力"]):
            out.append(
                f"[{name}] 把约束变量化：在（{constraints_str}）下，列出 3 个“降低 20% 成本”的动作（删、换、借、外包、缩小范围），48 小时先做 1 个。"
            )
            continue

        # default：通用转化
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
    retrieve_pool: int = 36
    prefer_light_retriever: bool = True  # 轻量优先，稳定性更高


class DecisionEngine:
    BUILD_ID = "2026-02-28-light-retrieval-strong-extractor-slotfill"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

        # playbook texts (small corpus)
        self._pb_texts = [
            f"{x.get('lang','')} | {x.get('type','')} | {x.get('title','')}\n{x.get('text','')}\nTags: {', '.join(x.get('tags', []))}"
            for x in PLAYBOOK
        ]

        # ultra-light retriever (default)
        self.light_retriever = PlaybookLightRetriever(self._pb_texts)

        # optional: hybrid embedder (can fail on some envs, so keep it safe)
        self.embedder: Optional[HybridEmbedder] = None
        try:
            self.embedder = HybridEmbedder(EmbeddingConfig())
        except Exception:
            self.embedder = None

        self.rerank_cfg = RerankConfig()
        self.sig_matcher = _MiniTfidfMatcher()

    # -------- retrieval: light first, hybrid optional --------
    def _retrieve_rerank(self, query: str, k: int, lang: str) -> List[Dict[str, Any]]:
        pool = min(max(self.cfg.retrieve_pool, k * 3), len(PLAYBOOK))

        idxs: List[int] = []
        scores: List[float] = []

        # 1) light retriever always available
        li, ls = self.light_retriever.top_k(query, k=pool)
        idxs, scores = li, ls

        # 2) optional: blend hybrid scores if available (small boost, but never required)
        if self.embedder is not None and not self.cfg.prefer_light_retriever:
            try:
                hi, hs = self.embedder.top_k_with_scores(query, self._pb_texts, k=pool)
                # blend by rank (robust)
                rank_light = {idx: r for r, idx in enumerate(idxs)}
                rank_h = {idx: r for r, idx in enumerate(hi)}
                all_ids = list(dict.fromkeys(idxs + hi))
                blended = []
                for idx in all_ids:
                    r1 = rank_light.get(idx, pool + 5)
                    r2 = rank_h.get(idx, pool + 5)
                    # lower better
                    blended.append((idx, -(r1 + r2)))
                blended.sort(key=lambda x: x[1], reverse=True)
                idxs = [i for i, _ in blended[:pool]]
                # scores not strictly used by reranker beyond ordering; keep light scores fallback
                scores = [1.0 for _ in idxs]
            except Exception:
                pass

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

    # -------- stronger signals: pick a "continue" / "stop" line --------
    def _best_signal_for_option(self, signal_text: str, options: List[Option], target: Option) -> str:
        """
        从 signal_text 多行中选出“最像 target 且最具体”的 1 行。
        若无法对齐 target：退化为最具体的一行。
        """
        ls = _lines(signal_text)
        if not ls:
            return ""

        corp = [o.corpus() for o in options]

        scored: List[Tuple[str, float, int]] = []
        for ln in ls:
            idx, sim = self.sig_matcher.best_match(ln, corp)
            conc = _concreteness_score(ln)
            # 强约束：先优先对齐 target
            if idx >= 0 and options[idx].name == target.name:
                scored.append((ln, sim, conc))

        if scored:
            scored.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)
            return scored[0][0]

        # fallback: choose most concrete overall
        best = sorted(ls, key=_concreteness_score, reverse=True)[0]
        return best

    # -------- parse options: support N options --------
    def _parse_options(self, payload: Dict[str, Any]) -> List[Option]:
        """
        支持两种输入：
        1) 通用：payload["options_struct"] = [{"name":..., "best":..., "worst":..., "controls":...}, ...]
        2) 兼容旧版 A/B：a_name/a_best/...  b_name/b_best/...
        """
        opts: List[Option] = []

        struct = payload.get("options_struct")
        if isinstance(struct, list) and struct:
            for it in struct:
                if not isinstance(it, dict):
                    continue
                name = (it.get("name") or it.get("title") or "").strip()
                if not name:
                    continue
                opts.append(Option(
                    name=name,
                    best=it.get("best") or "",
                    worst=it.get("worst") or "",
                    controls=it.get("controls") or it.get("controllables") or ""
                ))
            if opts:
                return opts

        # fallback A/B
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
        return [a, b]

    # -------- choose trial candidate(s) by grips --------
    def _choose_trial_pick(self, options: List[Option]) -> Tuple[str, Option, Dict[str, str]]:
        """
        返回：
        - trial_pick 文案（可能是 "A 或 B" 或单一路径）
        - target_option（用来抽取 continue/stop 信号的主路径）
        - grips_map: {option_name: grip_line}
        """
        grips_map: Dict[str, str] = {}
        grip_scores: List[Tuple[Option, int]] = []
        for o in options:
            g = _pick_most_concrete(o.controls)
            grips_map[o.name] = g
            grip_scores.append((o, _concreteness_score(g)))

        # 排序找 top2
        grip_scores.sort(key=lambda x: x[1], reverse=True)
        top = grip_scores[0]
        second = grip_scores[1] if len(grip_scores) > 1 else None

        if second is None:
            return top[0].name, top[0], grips_map

        # 如果 top2 差距很小：给“或”（保留可逆性）
        if abs(top[1] - second[1]) <= 2:
            trial_pick = f"{top[0].name} 或 {second[0].name}"
            target = top[0]
            return trial_pick, target, grips_map

        # 否则选更可试探那条
        return top[0].name, top[0], grips_map

    # ----------------- main memo builder (CN) -----------------
    def build_memo_cn(self, payload: Dict[str, Any]) -> str:
        # --- read inputs ---
        decision = (payload.get("decision") or "").strip()
        options_text = payload.get("options") or ""
        status_6m = payload.get("status_6m") or ""
        status_2y = payload.get("status_2y") or ""

        options = self._parse_options(payload)

        priority = payload.get("priority") or "长期选择权（Optionality）"
        constraints = payload.get("constraints") or []
        regret = payload.get("regret") or "没有尝试"

        partial_control = payload.get("partial_control") or ""
        identity_anchor = payload.get("identity_anchor") or ""

        evidence_to_commit = payload.get("evidence_to_commit") or ""
        evidence_to_stop = payload.get("evidence_to_stop") or ""

        constraints_str = "、".join([c for c in constraints if str(c).strip()]) if constraints else "（未填写）"

        # --- grips + trial pick (generic & not hard-coded) ---
        trial_pick, target_option, grips_map = self._choose_trial_pick(options)

        # --- pick signals (strong-ish) ---
        commit_signal_short = self._best_signal_for_option(evidence_to_commit, options, target_option) or "（未填写继续信号）"
        stop_signal_short = self._best_signal_for_option(evidence_to_stop, options, target_option) or "（未填写止损信号）"

        # --- risk terms (for explanation only) ---
        combined_worst = " ".join([(o.worst or "") for o in options])
        risk_terms = _extract_risk_terms(combined_worst + " " + (status_2y or ""), limit=6)
        risk_terms_str = "、".join(risk_terms) if risk_terms else "（未提取到明显风险词）"

        # --- safeguards: show for top2 if trial_pick is "A 或 B" else chosen ---
        show_two = " 或 " in trial_pick

        if show_two:
            # identify the two names
            n1, n2 = [x.strip() for x in trial_pick.split("或", 1)]
            o1 = next((o for o in options if o.name == n1), options[0])
            o2 = next((o for o in options if o.name == n2), options[1] if len(options) > 1 else options[0])

            v1 = _extract_risk_variables(o1.worst, status_2y, max_n=2)
            v2 = _extract_risk_variables(o2.worst, status_2y, max_n=2)
            a1 = _risk_actions_generic(v1, o1, constraints_str, grips_map.get(o1.name, ""))
            a2 = _risk_actions_generic(v2, o2, constraints_str, grips_map.get(o2.name, ""))

            risk_variables_bullets = (
                f"- [{o1.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in (v1 or ["（未抽取到）"])]) + "\n"
                f"- [{o2.name}] 关键风险变量：\n" + "\n".join([f"- {x}" for x in (v2 or ["（未抽取到）"])])
            )
            risk_actions_bullets = (
                f"- [{o1.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in (a1[:2] if a1 else ["（未生成动作）"])]) + "\n"
                f"- [{o2.name}] 降低20%概率动作：\n" + "\n".join([f"- {x}" for x in (a2[:2] if a2 else ["（未生成动作）"])])
            )
            safeguard_note = f"（两条路径都在试探候选里：{trial_pick}）"
        else:
            chosen = target_option
            vars_ = _extract_risk_variables(chosen.worst, status_2y, max_n=4)
            actions_ = _risk_actions_generic(vars_, chosen, constraints_str, grips_map.get(chosen.name, ""))

            risk_variables_bullets = "\n".join([f"- {x}" for x in vars_]) if vars_ else "（未抽取到）"
            risk_actions_bullets = "\n".join([f"- {x}" for x in actions_[:3]]) if actions_ else "（未生成动作）"
            safeguard_note = f"（本次优先对齐：{trial_pick}）"

        # --- explanation paragraph (more like you, less template) ---
        # pick top grips (up to 2) for explanation
        grip_pairs = sorted([(o.name, grips_map.get(o.name, "")) for o in options],
                            key=lambda x: _concreteness_score(x[1]), reverse=True)
        grip_lines = []
        for name, g in grip_pairs[:2]:
            grip_lines.append(f"- 你在 {name} 里最具体的抓手是：{g if g else '（未写出具体行动）'}")
        grips_block = "\n".join(grip_lines) if grip_lines else "- （未写出具体抓手）"

        baseline_cost_short = _shorten(status_2y, 80) if status_2y.strip() else "（未写清不行动成本）"
        decision_short = _shorten(decision, 90) if decision else "（未填写）"

        explanation_personal = (
            "你这里最关键的信息不是“路径名字”，而是你有没有把问题从脑内地图，推到现实领土。\n\n"
            f"{grips_block}\n"
            f"- 你反复出现的风险信号/关键词：{risk_terms_str}\n"
            f"- 你写到的不行动成本（Baseline）：{baseline_cost_short}\n\n"
            "所以建议之所以成立，不是因为“某条路更正确”，而是因为：\n"
            "你已经开始把不确定性拆成抓手（能做什么），并且标出了风险核心（在怕什么）。下一步要做的是：\n"
            "用证据门槛把试探闭环起来——让现实反馈出现，而不是让解释变得更漂亮。\n\n"
            f"在你的约束（{constraints_str}）下，本轮最划算的策略是可逆实验：14 天换证据，再用证据加码/止损。\n\n"
            f"- 本轮继续/加码（1条）：{commit_signal_short}\n"
            f"- 本轮止损/调整（1条）：{stop_signal_short}\n\n"
            "你要的不是一次性下注，而是让下一步更容易：证据更清晰、风险更可控、后悔更小。"
        )

        # --- retrieval query (small & dense) ---
        # 轻量检索更吃“关键词密度”，所以 query 不要太长，保留关键字段
        opt_snippets = []
        for o in options[:3]:
            opt_snippets.append("\n".join([
                o.name,
                _shorten(o.worst, 90),
                _shorten(_pick_most_concrete(o.controls), 90)
            ]))
        query = "\n".join([
            decision_short,
            _shorten(options_text, 120),
            baseline_cost_short,
            "\n".join(opt_snippets),
            priority,
            constraints_str,
            commit_signal_short,
            stop_signal_short,
            risk_terms_str
        ]).strip()

        retrieved = self._retrieve_rerank(query, k=self.cfg.retrieve_k, lang="cn")
        reframes = [x for x in retrieved if x.get("type") == "reframe"][:2]
        moves = [x for x in retrieved if x.get("type") == "move"][:3]
        safeguards = [x for x in retrieved if x.get("type") == "safeguard"][:1]
        if not safeguards:
            safeguards = [x for x in PLAYBOOK if x.get("lang") == "cn" and x.get("type") == "safeguard"][:1]

        # --- slots for library.py (and future extensions) ---
        slots = dict(
            # required slots in your library.py
            constraints_str=constraints_str,
            trial_pick=trial_pick,
            commit_signal_short=commit_signal_short,
            stop_signal_short=stop_signal_short,
            risk_variables_bullets=risk_variables_bullets,
            risk_actions_bullets=risk_actions_bullets,
            safeguard_note=safeguard_note,

            # extra helpful slots (optional)
            decision_short=decision_short,
            baseline_cost_short=baseline_cost_short,
            risk_terms_str=risk_terms_str,
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

        # --- render options section (supports N) ---
        def fmt_option(o: Option) -> str:
            return (
                f"### {o.name}\n"
                f"- 最好：{o.best.strip() if o.best.strip() else '（未填写）'}\n"
                f"- 最坏：{o.worst.strip() if o.worst.strip() else '（未填写）'}\n"
                f"- 可控变量：\n{_bulletize(o.controls)}\n"
            )

        options_block = "\n".join([fmt_option(o) for o in options]) if options else "（未填写）"

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
{options_block}

## 5. 可控 / 不可控 / 部分可控（把焦虑变成变量）
- 可控（你能直接改变的）：来自各条路径的「可控变量」清单（见上）。
- 不可控（你无法决定的）：市场/组织/他人决策/宏观环境等（你的动作是“对冲”，不是内耗）。
- 部分可控（最值得投入的）：{partial_control.strip() if partial_control.strip() else "（未填写）"}

## 6. 目标函数与约束（你在优化什么）
你最重视的是「{priority}」。你当前的约束是：{", ".join(constraints) if constraints else "（未选择）"}。你的后悔倾向是「{regret}」。

## 7. 推导链（Why）
### 7.1 本轮试探候选
- 试探路径候选：{trial_pick}

### 7.2 基于你输入的“解释段”（更像你自己的版本）
{explanation_personal}

## 8. 检索增强（两阶段：轻量检索→轻量重排）
> 默认用 TF-IDF cosine 做轻量召回（稳定/便宜），再用 ngram/重叠率/类型偏置做轻量 rerank，提高贴合度与稳定性。

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
