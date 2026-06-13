"""Deterministic-first, language- and domain-agnostic cross-document entity resolution.

Open Research Question #1: near-duplicate entities extracted by workers
("Zenith Petrochem" vs "Zenith Petrochemical", "Pinnacle Industrial Solutions, Inc."
vs "Pinnacle Industrial Solutions") that no single worker flags. This module scans the
blackboard for entity names, clusters near-duplicates with multi-signal string
similarity, and writes resolution entries so downstream workers reconcile them.

Design — written to answer the maintainer's review (multilingual / multi-domain /
extensible / not fragile):

* **No hardcoded vocabulary.** Generic boilerplate ("defined terms" like *Term Loan*,
  *Closing Date* in legal docs, or *Strategic Priority* in a strategy memo) is identified
  by its **corpus document-frequency on the actual blackboard**, never by a baked-in
  English legal blocklist. The signal is domain- and language-agnostic and re-derives
  itself per task.
* **Unicode-correct throughout.** Normalization is NFKC + ``casefold`` and every regex is
  unicode-aware, so Latin, Cyrillic, Greek, CJK, etc. are handled — not just ASCII English.
* **Pluggable, optionally model-backed extraction.** Candidate-name extraction is a swappable
  front-end (``ResolutionConfig.extract`` / an optional ``ModelCaller``). The resolution core
  operates on entity strings in *any* language or domain; the only language-sensitive part is
  isolated and replaceable.
* **Hybrid, not all-or-nothing.** The deterministic zero-token fast path resolves the confident
  majority. Ambiguous fuzzy pairs are either adjudicated by an optional ``ModelCaller`` or
  surfaced as review *candidates* with calibrated confidence — never hard-asserted, because
  string similarity alone cannot separate a true variant (Petrochem/Petrochemical) from
  distinct names sharing a prefix (Petrochem/Petroleum).

All thresholds and term hints live on ``ResolutionConfig``. Tokens used: 0 unless an optional
``ModelCaller`` is supplied for ambiguous adjudication.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .blackboard import Blackboard
from .models import Entry, EntrySource, WorkerRecord, gen_entry_id

try:  # optional — only used when a caller is supplied for ambiguous adjudication
    from .worker_dispatch import call_model as _call_model
except Exception:  # pragma: no cover - worker_dispatch may pull heavy deps
    _call_model = None


# ---------------------------------------------------------------------------
# Configuration (everything tunable lives here — no magic numbers in the body)
# ---------------------------------------------------------------------------

# Organisation / legal-form suffixes folded out before matching. This is a broad,
# MULTILINGUAL, *soft* recall aid (not a gate): unknown forms simply are not folded,
# they are never dropped. Extend via ResolutionConfig.extra_suffixes.
DEFAULT_SUFFIXES: frozenset[str] = frozenset({
    # Anglo / generic (kept conservative — distinguishing tokens like "Capital", "Trust",
    # "Bank", "Ventures" are deliberately NOT folded, so "Northbrook Capital" stays distinct).
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "lp", "llp", "partners", "partnership", "associates",
    "group", "holdings",
    # German / Dutch / Nordic
    "gmbh", "ag", "kg", "kgaa", "ohg", "mbh", "bv", "nv", "ab", "asa", "as", "oy", "oyj",
    # Romance
    "sa", "sas", "sarl", "spa", "srl", "sl", "sca", "lda",
    # Slavic (transliterated)
    "ooo", "oao", "zao", "pao",
    # East-Asian (native script tokens survive NFKC; included so they fold too)
    "株式会社", "有限会社", "合同会社", "有限公司", "股份有限公司", "주식회사",
    # Other
    "pty", "bhd", "sdn", "pte",
})


@dataclass(frozen=True)
class ResolutionConfig:
    """All tunables for entity resolution. Safe, language-neutral defaults."""

    # Composite-similarity gates.
    match_threshold: float = 0.72      # composite >= this → candidate same-entity match
    high_confidence: float = 0.85      # composite >= this → asserted (or model-confirmed)
    # Name length bounds (characters), unicode code points.
    min_name_len: int = 3
    max_name_len: int = 80
    # Cap on entries scanned per run (cost guard).
    max_entries: int = 500

    # Corpus-relative generic-term detection — REPLACES the old hardcoded legal blocklist.
    # A token appearing in >= this fraction of distinct candidate names is treated as
    # generic boilerplate (a "defined term"), not a distinguishing entity token. Inferred
    # from the blackboard itself, so it adapts to any domain or language.
    generic_doc_freq: float = 0.08     # token in >=8% of name occurrences = boilerplate
    min_corpus_for_generic: int = 8    # don't infer generics from a tiny corpus
    min_generic_occurrences: int = 3   # ...and seen >=3 times (avoid flagging a small corpus)

    # Soft, optional hints (all default-empty — NOT required, NOT English-specific).
    suffixes: frozenset[str] = DEFAULT_SUFFIXES
    extra_suffixes: frozenset[str] = frozenset()
    extra_generic_terms: frozenset[str] = frozenset()  # caller-supplied stop terms, any language
    # Join two name-like tokens across a single short lowercase connector (of/de/van/…)
    # without hardcoding any language's connector words. 0 disables.
    connector_max_len: int = 3

    # Hybrid adjudication.
    use_caller_for_ambiguous: bool = True  # if a ModelCaller is supplied, adjudicate the band
    caller_max_tokens: int = 256

    # Pluggable extraction. If provided, completely replaces the built-in unicode extractor.
    # Signature: extract(content: str, cfg: ResolutionConfig) -> list[str]
    extract: Optional[Callable[[str, "ResolutionConfig"], list[str]]] = None


DEFAULT_CONFIG = ResolutionConfig()


# ---------------------------------------------------------------------------
# Normalization (unicode-correct, language-agnostic)
# ---------------------------------------------------------------------------

def _normalize_entity(text: str, cfg: ResolutionConfig = DEFAULT_CONFIG) -> str:
    """Collapse an entity name to a canonical, script-agnostic comparison form."""
    s = unicodedata.normalize("NFKC", text).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)   # \w is unicode-aware
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    words = s.split()
    suffixes = cfg.suffixes | cfg.extra_suffixes
    kept = [w for w in words if w not in suffixes]
    # Never annihilate a name that is *entirely* suffix-like — keep the original tokens.
    return " ".join(kept) if kept else " ".join(words)


def _depluralize(norm: str) -> str:
    """Collapse a trailing plural 's' per token (soft Latin-script dup-suppression)."""
    return " ".join(
        t[:-1] if t.endswith("s") and len(t) > 3 else t
        for t in norm.split()
    )


# ---------------------------------------------------------------------------
# Similarity metrics (operate on normalized strings — already language-agnostic)
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> set[str]:
    joined = " ".join(text.split())
    if len(joined) < 3:
        return {joined}
    return {joined[i:i + 3] for i in range(len(joined) - 2)}


def _jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (0..1). Operates on unicode code points."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = max(0, max(len1, len2) // 2 - 1)
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0..1)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return 1.0 - prev[len2] / max(len1, len2)


def _token_overlap(norm1: str, norm2: str) -> float:
    t1, t2 = set(norm1.split()), set(norm2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _is_prefix_variant(norm1: str, norm2: str) -> bool:
    """True if one name is a word- or char-level prefix of the other."""
    w1, w2 = norm1.split(), norm2.split()
    if not w1 or not w2:
        return False
    shorter, longer = (w1, w2) if len(w1) <= len(w2) else (w2, w1)
    if shorter == longer[:len(shorter)]:
        return True
    if len(w1) == len(w2):
        for i in range(len(w1) - 1):
            if w1[i] != w2[i]:
                return False
        a, b = w1[-1], w2[-1]
        return a.startswith(b) or b.startswith(a)
    return False


@dataclass
class MatchResult:
    entry_a: Entry
    entry_b: Entry
    raw_name_a: str
    raw_name_b: str
    norm_a: str
    norm_b: str
    jaro_winkler: float
    levenshtein: float
    trigram_jaccard: float
    token_overlap: float
    is_prefix: bool
    composite: float
    confidence: float
    adjudication: str = "deterministic"  # deterministic | model_confirmed | model_rejected | candidate


def _composite_score(jw: float, lev: float, tri: float, tok: float, is_prefix: bool) -> float:
    base = 0.35 * jw + 0.25 * lev + 0.25 * tri + 0.15 * tok
    if is_prefix and base >= 0.5:
        base = min(1.0, base + 0.15)
    return round(base, 4)


# Back-compat module constants (mirror the config defaults).
MATCH_THRESHOLD = DEFAULT_CONFIG.match_threshold
HIGH_CONFIDENCE = DEFAULT_CONFIG.high_confidence


# ---------------------------------------------------------------------------
# Candidate-name extraction (unicode-aware; pluggable; optional model fallback)
# ---------------------------------------------------------------------------

def _is_name_token(tok: str) -> bool:
    """Language-agnostic 'looks like part of a proper name' test.

    * Cased scripts (Latin/Cyrillic/Greek/Armenian/…): token led by an uppercase or
      titlecase letter (category Lu/Lt) — e.g. "Zenith", "Газпром", "Ελλάς".
    * Uncased scripts (CJK, Hebrew, Arabic, Thai…): a run of 2+ letters with NO lowercase
      letter present (so we capture 株式 / טכנולוגיה without grabbing lowercase english words).
    """
    if not tok:
        return False
    cat0 = unicodedata.category(tok[0])
    if cat0 in ("Lu", "Lt"):
        return True
    if len(tok) >= 2 and all(unicodedata.category(c)[0] == "L" for c in tok):
        if not any(unicodedata.category(c) == "Ll" for c in tok):
            return True
    return False


def _default_extract(content: str, cfg: ResolutionConfig = DEFAULT_CONFIG) -> list[str]:
    """Extract candidate proper-name spans from text without any English/legal regex.

    Maximal runs of name-like tokens, optionally bridged by one short lowercase connector
    (handles "X of Y" / "X de Y" generically without hardcoding connector words).
    """
    # Tokenize but REMEMBER separators: a punctuation boundary breaks a name run, whitespace
    # does not. (Tokenizing with bare \w+ would merge "Поставщик: Роснефть" into one span.)
    text = unicodedata.normalize("NFKC", content)
    toks: list[tuple[str, bool]] = []   # (token, punctuation_break_before)
    prev_end: int | None = None
    for m in re.finditer(r"\w+", text, flags=re.UNICODE):
        gap = text[prev_end:m.start()] if prev_end is not None else " "
        toks.append((m.group(0), bool(gap) and not gap.isspace()))
        prev_end = m.end()

    spans: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        while cur and not _is_name_token(cur[-1]):
            cur.pop()   # never end a span on a bridged connector
        if cur:
            spans.append(" ".join(cur))
        cur.clear()

    for idx, (tok, brk) in enumerate(toks):
        if brk:
            flush()
        if _is_name_token(tok):
            cur.append(tok)
        elif (cur and not brk and cfg.connector_max_len
              and len(tok) <= cfg.connector_max_len and tok.islower()
              and idx + 1 < len(toks) and _is_name_token(toks[idx + 1][0])
              and not toks[idx + 1][1]):
            cur.append(tok)   # bridge one short lowercase connector ("X of Y")
        else:
            flush()
    flush()
    return [s for s in spans if s]


def _entity_strings(entry: Entry, cfg: ResolutionConfig) -> list[str]:
    """Raw candidate names from one entry (built-in extractor or a pluggable one)."""
    if not entry.content:
        return []
    extractor = cfg.extract or _default_extract
    out: list[str] = []
    seen: set[str] = set()
    for name in extractor(entry.content, cfg):
        name = name.strip()
        if not (cfg.min_name_len <= len(name) <= cfg.max_name_len):
            continue
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _corpus_generic_tokens(names: list[str], cfg: ResolutionConfig) -> frozenset[str]:
    """Tokens that are generic boilerplate by corpus document-frequency.

    Replaces the hardcoded ``_DEFINED_TERMS`` blocklist with a signal derived from the
    actual blackboard: a token shared across a large fraction of *distinct* candidate
    names is a defined term/boilerplate, not a distinguishing entity token.
    """
    if len(names) < cfg.min_corpus_for_generic:
        return frozenset(cfg.extra_generic_terms)
    # Occurrence frequency: fraction of candidate-name OCCURRENCES containing each token.
    # Boilerplate ("the", "exhibit", "section", month names) recurs heavily across the corpus;
    # real entity tokens ("hawthorne", "datadog") do not. Counting occurrences (not distinct
    # names) is what catches standalone-repeated boilerplate, whose distinct-name DF is 1.
    df: Counter[str] = Counter()
    for nm in names:
        for tok in set(_normalize_entity(nm, cfg).split()):
            if tok:
                df[tok] += 1
    n = len(names)
    generic = {t for t, c in df.items()
               if c >= cfg.min_generic_occurrences and c / n >= cfg.generic_doc_freq}
    return frozenset(generic | set(cfg.extra_generic_terms))


def _is_distinguishing(norm: str, generic: frozenset[str]) -> bool:
    """A name is a real entity candidate iff it has >=1 non-generic, non-trivial token."""
    tokens = norm.split()
    if not tokens or len(norm) < 3:
        return False
    return any(t not in generic for t in tokens)


def _strip_generic_edges(norm: str, generic: frozenset[str]) -> str:
    """Trim leading/trailing corpus-generic tokens from a name.

    Language-agnostic replacement for a hardcoded leading-stopword list: sentence-initial
    determiners ("The", "Our", "Der", "La", …) surface as generic by document-frequency
    and are stripped here, so "The Acme Robotics" matches "Acme Robotics". Interior tokens
    are preserved; a name that is entirely generic falls back to the full form (and is then
    rejected by ``_is_distinguishing``).
    """
    toks = norm.split()
    lo, hi = 0, len(toks)
    while lo < hi and toks[lo] in generic:
        lo += 1
    while hi > lo and toks[hi - 1] in generic:
        hi -= 1
    core = toks[lo:hi]
    return " ".join(core) if core else norm


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    clusters: list[list[dict]]
    match_pairs: list[MatchResult]
    entries_created: list[Entry]
    tokens_used: int
    generic_terms: frozenset[str] = field(default_factory=frozenset)


def _adjudicate_with_caller(
    caller: Any, cfg: ResolutionConfig, name_a: str, name_b: str,
) -> tuple[Optional[bool], int]:
    """Ask an optional ModelCaller whether two names are the same entity.

    Returns (same | None, tokens). None means 'could not decide' → caller falls back to
    surfacing a candidate. Never raises.
    """
    if caller is None or _call_model is None:
        return None, 0
    prompt = (
        "You reconcile entity names across documents in ANY language or domain. "
        "Are these two names the SAME real-world entity (e.g. a spelling/abbreviation/"
        "legal-form variant), or DIFFERENT entities that merely look similar?\n"
        f"A: {name_a!r}\nB: {name_b!r}\n"
        'Reply with strict JSON only: {"same": true|false, "confidence": 0.0-1.0}'
    )
    try:
        payload, tokens = _call_model(caller, prompt, max_tokens=cfg.caller_max_tokens)
    except Exception:
        return None, 0
    if isinstance(payload, dict) and "same" in payload:
        return bool(payload["same"]), int(tokens or 0)
    return None, int(tokens or 0)


def resolve_entities(
    blackboard: Blackboard,
    *,
    config: ResolutionConfig | None = None,
    caller: Any | None = None,
    threshold: float | None = None,
    max_entries: int | None = None,
) -> ResolutionResult:
    """Scan the blackboard for near-duplicate entity names and write resolution entries.

    Deterministic and zero-token by default. Supply ``caller`` (a ModelCaller) to let the
    model adjudicate ambiguous fuzzy pairs instead of surfacing them as review candidates.
    """
    cfg = config or DEFAULT_CONFIG
    if threshold is not None or max_entries is not None:
        cfg = ResolutionConfig(
            **{**cfg.__dict__,
               "match_threshold": threshold if threshold is not None else cfg.match_threshold,
               "max_entries": max_entries if max_entries is not None else cfg.max_entries})

    active = [e for e in blackboard.entries
              if e.status == "active" and e.type in ("observation", "analysis")]

    # 1. Gather raw candidate names, then derive the generic-term set from the corpus.
    per_entry: list[tuple[Entry, list[str]]] = []
    all_names: list[str] = []
    for entry in active[:cfg.max_entries]:
        names = _entity_strings(entry, cfg)
        if names:
            per_entry.append((entry, names))
            all_names.extend(names)

    generic = _corpus_generic_tokens(all_names, cfg)

    # 2. Keep only distinguishing names; record occurrences keyed by normalized form.
    occurrences: list[dict] = []
    seen_norm: dict[str, list[dict]] = {}
    for entry, names in per_entry:
        doc = entry.source.document if entry.source else ""
        for raw_name in names:
            norm = _strip_generic_edges(_normalize_entity(raw_name, cfg), generic)
            if len(norm) < cfg.min_name_len or not _is_distinguishing(norm, generic):
                continue
            occ = {"entry_id": entry.id, "raw": raw_name, "norm": norm, "doc": doc,
                   "content_snippet": entry.content[:200]}
            occurrences.append(occ)
            seen_norm.setdefault(norm, []).append(occ)

    if len(occurrences) < 2:
        return ResolutionResult([], [], [], 0, generic)

    # 3. Fuzzy match between distinct normalized names.
    unique_norms = list(seen_norm.items())
    tok_by_norm = {nm: set(nm.split()) for nm, _ in unique_norms}
    tri_by_norm = {nm: _trigrams(nm) for nm, _ in unique_norms}
    match_pairs: list[MatchResult] = []
    tokens_used = 0
    n = len(unique_norms)
    for i in range(n):
        norm_i, occs_i = unique_norms[i]
        tokens_i = tok_by_norm[norm_i]
        tri_i = tri_by_norm[norm_i]
        for j in range(i + 1, n):
            norm_j, occs_j = unique_norms[j]
            tokens_j = tok_by_norm[norm_j]
            # Cheap relatedness pre-filter: a shared exact token, OR — for single-token names
            # in ANY script, where token overlap structurally cannot fire ("Роснефть" vs
            # "Роснефти") — a shared trigram.
            if not (tokens_i & tokens_j):
                if not ((len(tokens_i) == 1 or len(tokens_j) == 1)
                        and (tri_i & tri_by_norm[norm_j])):
                    continue
            if _depluralize(norm_i) == _depluralize(norm_j):
                continue
            jw = _jaro_winkler(norm_i, norm_j)
            if jw < 0.5:
                continue
            lev = _levenshtein_ratio(norm_i, norm_j)
            tri_j = tri_by_norm[norm_j]
            tri_jacc = len(tri_i & tri_j) / len(tri_i | tri_j) if (tri_i | tri_j) else 0.0
            tok = _token_overlap(norm_i, norm_j)
            is_pre = _is_prefix_variant(norm_i, norm_j)
            composite = _composite_score(jw, lev, tri_jacc, tok, is_pre)
            if composite < cfg.match_threshold:
                continue

            occ_a, occ_b = occs_i[0], occs_j[0]
            entry_a = blackboard.find_entry(occ_a["entry_id"])
            entry_b = blackboard.find_entry(occ_b["entry_id"])
            if not entry_a or not entry_b:
                continue

            # Adjudicate the ambiguous band [match_threshold, high_confidence).
            adjudication = "deterministic"
            if composite >= cfg.high_confidence:
                confidence = min(0.98, composite + 0.05)
            else:
                same, tok_used = (None, 0)
                if caller is not None and cfg.use_caller_for_ambiguous:
                    same, tok_used = _adjudicate_with_caller(
                        caller, cfg, occ_a["raw"], occ_b["raw"])
                    tokens_used += tok_used
                if same is True:
                    adjudication = "model_confirmed"
                    confidence = min(0.95, composite + 0.1)
                elif same is False:
                    adjudication = "model_rejected"
                    continue  # model says different — drop the pair
                else:
                    adjudication = "candidate"
                    confidence = round(min(0.7, composite), 2)

            match_pairs.append(MatchResult(
                entry_a=entry_a, entry_b=entry_b,
                raw_name_a=occ_a["raw"], raw_name_b=occ_b["raw"],
                norm_a=norm_i, norm_b=norm_j,
                jaro_winkler=jw, levenshtein=lev, trigram_jaccard=tri_jacc,
                token_overlap=tok, is_prefix=is_pre, composite=composite,
                confidence=confidence, adjudication=adjudication,
            ))

    clusters = _cluster_matches(match_pairs, seen_norm)
    created = _create_resolution_entries(blackboard, clusters, match_pairs)
    return ResolutionResult(clusters, match_pairs, created, tokens_used, generic)


# ---------------------------------------------------------------------------
# Clustering (union-find)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry


def _cluster_matches(
    match_pairs: list[MatchResult],
    seen_norm: dict[str, list[dict]],
) -> list[list[dict]]:
    """Group occurrences into clusters via union-find over normalized forms.

    Only norms joined by a *surviving* match pair are unified (model-rejected pairs were
    already dropped). A cluster is returned only if it holds >1 distinct surface form.
    """
    uf = _UnionFind()
    for norm in seen_norm:
        uf.find(norm)
    for pair in match_pairs:
        uf.union(pair.norm_a, pair.norm_b)

    groups: dict[str, list[dict]] = {}
    for norm, occs in seen_norm.items():
        bucket = groups.setdefault(uf.find(norm), [])
        for occ in occs:
            if not any(o["entry_id"] == occ["entry_id"] and o["raw"] == occ["raw"]
                       for o in bucket):
                bucket.append(occ)

    return [
        members for members in groups.values()
        if len({m["raw"].strip().casefold() for m in members}) > 1
    ]


# ---------------------------------------------------------------------------
# Blackboard entry creation
# ---------------------------------------------------------------------------

def _canonical_name(cluster: list[dict]) -> str:
    freq = Counter(o["raw"].strip() for o in cluster if o["raw"].strip())
    if not freq:
        return ""
    return max(freq.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _create_resolution_entries(
    blackboard: Blackboard,
    clusters: list[list[dict]],
    match_pairs: list[MatchResult],
) -> list[Entry]:
    """Add one entity-resolution entry per cluster (exact = asserted, fuzzy = candidate)."""
    created: list[Entry] = []
    worker = WorkerRecord(
        worker_id="entity_resolution",
        description="deterministic_entity_clustering",
        iteration=blackboard.iteration,
    )

    for cluster in clusters:
        canonical = _canonical_name(cluster)
        variants = sorted(
            {o["raw"].strip() for o in cluster if o["raw"].strip()} - {canonical}
        )
        if not variants:
            continue

        distinct_norms = {c["norm"] for c in cluster}
        is_exact = len(distinct_norms) == 1

        best_composite = 0.0
        model_confirmed = False
        for pair in match_pairs:
            if pair.norm_a in distinct_norms and pair.norm_b in distinct_norms:
                best_composite = max(best_composite, pair.composite)
                if pair.adjudication == "model_confirmed":
                    model_confirmed = True

        variant_lines = []
        for v in variants:
            v_docs = sorted({o["doc"] for o in cluster
                             if o["raw"].strip() == v and o["doc"]})
            doc_tag = f" (from {', '.join(v_docs)})" if v_docs else ""
            variant_lines.append(f'- "{v}"{doc_tag}')

        if is_exact or model_confirmed:
            best_composite = max(best_composite, 0.95 if is_exact else best_composite)
            confidence = 0.9
            header = (
                f'ENTITY RESOLUTION: "{canonical}" appears under '
                f"{len(variants) + 1} surface form(s)"
                + (" (identical after normalization):" if is_exact
                   else " (confirmed same entity):")
            )
            note = ("These are the same entity written differently and should be "
                    "cross-referenced in all analysis.")
            kind = "exact" if is_exact else "confirmed"
        else:
            confidence = round(min(0.7, best_composite or 0.6), 2)
            header = (
                f'ENTITY RESOLUTION (CANDIDATE): "{canonical}" may be the same '
                f"entity as {len(variants)} similar name(s):"
            )
            note = ("Similar but NOT identical after normalization — review before "
                    "merging. String similarity cannot distinguish a true variant "
                    "(Petrochem/Petrochemical) from distinct entities that merely share "
                    "a prefix (Petrochem/Petroleum).")
            kind = "candidate"

        content = (header + "\n" + "\n".join(variant_lines)
                   + f'\nCanonical form: "{canonical}"\n' + note)

        entry = Entry(
            id=gen_entry_id(),
            type="analysis",
            content=content,
            source=EntrySource(
                document="; ".join(sorted({c["doc"] for c in cluster if c["doc"]})),
                section="cross_document",
                evidence=f"Entity resolution ({kind}): {len(variants) + 1} forms "
                         f"(composite={best_composite:.3f})",
            ),
            created_by=worker,
            confidence=confidence,
            tags=["entity_resolution", kind,
                  f"variant_count:{len(variants) + 1}",
                  f"composite:{best_composite:.3f}"],
            status="active",
        )
        blackboard.add_entry(entry)
        created.append(entry)

    return created


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_entity_resolution(blackboard: Blackboard, **kwargs) -> dict:
    """Run entity resolution and return a summary report. Zero tokens unless a caller is given."""
    result = resolve_entities(blackboard, **kwargs)
    return {
        "schema_version": 2,
        "clusters_found": len(result.clusters),
        "match_pairs": len(result.match_pairs),
        "entries_created": len(result.entries_created),
        "tokens_used": result.tokens_used,
        "generic_terms_inferred": sorted(result.generic_terms)[:50],
        "clusters": [
            {
                "canonical": _canonical_name(c),
                "variants": [
                    {"raw": m["raw"], "doc": m["doc"], "entry_id": m["entry_id"]}
                    for m in c
                ],
            }
            for c in result.clusters
        ],
        "matches": [
            {
                "name_a": m.raw_name_a, "name_b": m.raw_name_b,
                "composite": m.composite, "adjudication": m.adjudication,
                "jaro_winkler": round(m.jaro_winkler, 4),
                "levenshtein": round(m.levenshtein, 4),
                "trigram_jaccard": round(m.trigram_jaccard, 4),
                "token_overlap": round(m.token_overlap, 4),
                "is_prefix": m.is_prefix,
            }
            for m in result.match_pairs
        ],
    }
