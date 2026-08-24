"""State model for the target-closing loop.

Four primitives: Sources, Claims, Targets, Actions — plus an append-only
Event log. Everything the system knows or does is one of these.

Design rules:
- Target persisted status is only open/closed/waived/blocked. Everything
  else (needs_evidence, needs_analysis, ...) is COMPUTED from structure.
- Claims have a kind plus orthogonal flags — no linear readiness ladder.
- Code does bookkeeping (counts, flags, provenance); LLMs make every
  semantic judgment. Blockers computed here are structural facts only.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLAIM_KINDS = (
    "observation", "analysis", "calculation", "comparison", "issue",
    "recommendation", "gap", "uncertainty", "contradiction", "decision",
    "requirement",
)
DERIVED_KINDS = (
    "analysis", "calculation", "comparison", "issue",
    "recommendation", "decision",
)
ACTION_KINDS = ("read", "search", "bind", "analyze", "verify", "synthesize")
TARGET_STATUSES = ("open", "closed", "waived", "blocked")
MATERIALITY = ("critical", "high", "medium", "low")


def _materiality_rank(m: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(m, 1)


@dataclass
class Claim:
    id: str = ""
    kind: str = "observation"
    content: str = ""
    source_doc: str | None = None
    source_section: str | None = None
    evidence: str = ""
    source_span: tuple[int, int] | None = None
    support_refs: list[str] = field(default_factory=list)
    contradicts_refs: list[str] = field(default_factory=list)
    target_refs: list[str] = field(default_factory=list)
    confidence: float = 0.6
    verified: bool | None = None
    superseded: bool = False
    iteration: int = 0
    created_by: str = ""

    @property
    def is_derived(self) -> bool:
        return self.kind in DERIVED_KINDS

    @property
    def active(self) -> bool:
        return not self.superseded

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "content": self.content,
            "source_doc": self.source_doc, "source_section": self.source_section,
            "evidence": self.evidence,
            "source_span": list(self.source_span) if self.source_span is not None else None,
            "support_refs": self.support_refs,
            "contradicts_refs": self.contradicts_refs,
            "target_refs": self.target_refs,
            "confidence": self.confidence, "verified": self.verified,
            "superseded": self.superseded,
            "iteration": self.iteration, "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        span = data.get("source_span")
        source_span = None
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                source_span = (int(span[0]), int(span[1]))
            except (TypeError, ValueError):
                source_span = None
        try:
            confidence = float(data.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "observation")),
            content=str(data.get("content", "")),
            source_doc=data.get("source_doc"),
            source_section=data.get("source_section"),
            evidence=str(data.get("evidence", "")),
            source_span=source_span,
            support_refs=[str(r) for r in data.get("support_refs", [])],
            contradicts_refs=[str(r) for r in data.get("contradicts_refs", [])],
            target_refs=[str(r) for r in data.get("target_refs", [])],
            confidence=confidence,
            verified=data.get("verified"),
            superseded=bool(data.get("superseded", False)),
            iteration=int(data.get("iteration", 0)),
            created_by=str(data.get("created_by", "")),
        )


@dataclass
class Target:
    id: str = ""
    need: str = ""
    materiality: str = "medium"
    status: str = "open"
    reason: str = ""
    claim_refs: list[str] = field(default_factory=list)
    created_iteration: int = 0
    resolved_iteration: int | None = None
    proposed_by: str = "seed"
    # Claims that justified closure — closure is a defeasible verdict, not a
    # tombstone. Later evidence touching the basis flags a reopen candidate.
    closure_basis: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def rank(self) -> int:
        return _materiality_rank(self.materiality)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "need": self.need, "materiality": self.materiality,
            "status": self.status, "reason": self.reason,
            "claim_refs": self.claim_refs,
            "created_iteration": self.created_iteration,
            "resolved_iteration": self.resolved_iteration,
            "proposed_by": self.proposed_by,
            "closure_basis": self.closure_basis,
        }


@dataclass
class Obligation:
    """What the final answer owes — derived from the user's instruction
    (and corpus-discovered requirements), never from a task taxonomy.

    Coverage standard is a property of the instruction's language:
    "compare every provision" demands exhaustive; "main risks" demands
    summary. Targets answer questions; obligations bind the answer.
    """
    id: str = ""
    text: str = ""
    origin: str = "instruction"
    coverage: str = "material"
    mandatory: bool = True
    status: str = "open"
    reason: str = ""

    @property
    def set_valued(self) -> bool:
        # native-complete deliberately excluded: v5 showed unit machinery
        # strangles drafting (units force checklist-stitching where coherent
        # composition wins). Drafts are served by targets + requirement claims.
        return self.coverage in ("exhaustive", "material")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "origin": self.origin,
            "coverage": self.coverage, "mandatory": self.mandatory,
            "status": self.status, "reason": self.reason,
        }


@dataclass
class Unit:
    """A repeated item the answer must account for under an obligation —
    request category, provision, discrepancy, policy term, issue.

    Units are coverage rows, not investigation questions: they are never
    scheduled individually, never get controller cards, and are serviced
    by batch (bind/analysis) and preserved by packets.
    """
    id: str = ""
    name: str = ""
    obligation_ref: str = ""
    anchor: str = ""
    claim_refs: list[str] = field(default_factory=list)
    status: str = "discovered"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "obligation_ref": self.obligation_ref, "anchor": self.anchor,
            "claim_refs": self.claim_refs,
            "status": self.status, "reason": self.reason,
        }


@dataclass
class Source:
    """A document or web result that can ground claims.

    Wraps the lazy Document machinery — text is materialized on first read,
    never during triage (triage is metadata-only by design).
    """
    id: str = ""
    name: str = ""
    path: str = ""
    kind: str = "document"
    size_bytes: int = 0
    read_status: str = "unread"
    relevance: str = "unknown"
    relevance_reason: str = ""
    web_text: str = ""
    _doc: Any = field(default=None, repr=False)
    _section_index: Any = field(default=None, repr=False)

    @property
    def path_hint(self) -> str:
        """Directory portion of the path — structural signal for triage."""
        if not self.path:
            return ""
        norm = self.path.replace("\\", "/")
        parts = norm.split("/")
        return "/".join(parts[-4:-1]) if len(parts) > 1 else ""

    def text(self) -> str:
        if self.kind == "web":
            return self.web_text
        if self._doc is None:
            return ""
        return self._doc.text

    def section_index(self):
        if self._section_index is None:
            from ..swarm.section_index import build_section_index
            self._section_index = build_section_index(self.text())
        return self._section_index

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "path": self.path,
            "kind": self.kind, "size_bytes": self.size_bytes,
            "read_status": self.read_status,
            "relevance": self.relevance,
            "relevance_reason": self.relevance_reason,
        }


@dataclass
class Event:
    iteration: int = 0
    kind: str = ""
    summary: str = ""
    detail: dict = field(default_factory=dict)
    model: str = ""
    tokens: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        d = {
            "iteration": self.iteration, "kind": self.kind,
            "summary": self.summary, "detail": self.detail,
            "model": self.model, "tokens": self.tokens,
        }
        if self.tokens_in or self.tokens_out:
            d["tokens_in"] = self.tokens_in
            d["tokens_out"] = self.tokens_out
        return d


@dataclass
class Board:
    """The shared state: sources, claims, targets, events + bookkeeping.

    Token-tracking fields mirror swarm.Blackboard so runner.py cost
    accounting works unchanged.
    """
    instruction: str = ""
    metadata: dict = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    obligations: list[Obligation] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    iteration: int = 0
    stop_reason: str = ""
    total_tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cost_by_model: dict = field(default_factory=dict)
    token_budget: int = 3_000_000
    output_dir: str = ""

    def __post_init__(self):
        self._lock = threading.Lock()
        self._claim_counter = 0
        self._target_counter = 0
        self._obligation_counter = 0
        self._unit_counter = 0
        self._claim_index: dict[str, Claim] = {}
        self._target_index: dict[str, Target] = {}
        self._source_index: dict[str, Source] = {}
        self._obligation_index: dict[str, Obligation] = {}
        self._unit_index: dict[str, Unit] = {}
        self._unit_name_index: dict[tuple[str, str], str] = {}

    # --- IDs ---

    def next_claim_id(self) -> str:
        with self._lock:
            self._claim_counter += 1
            return f"c{self._claim_counter}"

    def next_target_id(self) -> str:
        with self._lock:
            self._target_counter += 1
            return f"t{self._target_counter}"

    # --- Mutation (thread-safe: executors run in parallel) ---

    def add_source(self, source: Source) -> None:
        with self._lock:
            self.sources.append(source)
            self._source_index[source.id] = source

    def add_claim(self, claim: Claim) -> bool:
        """Add a claim; returns False if an exact-content duplicate exists.

        Dedup is exact (normalized content key) — semantically-similar claims
        are left for bind/maintenance judgment, not code heuristics.
        """
        if not claim.id:
            claim.id = self.next_claim_id()
        content_key = " ".join(claim.content.lower().split())[:160]
        if claim.source_doc:
            loc = claim.source_span if claim.source_span is not None else (claim.source_section or "")
            key = f"{claim.kind}|{content_key}|{claim.source_doc}|{loc}"
        else:
            key = f"{claim.kind}|{content_key}"
        with self._lock:
            if not hasattr(self, "_content_index"):
                self._content_index: dict[str, str] = {}
            if key in self._content_index:
                return False
            self._content_index[key] = claim.id
            self.claims.append(claim)
            self._claim_index[claim.id] = claim
            for tid in claim.target_refs:
                t = self._target_index.get(tid)
                if t is not None and claim.id not in t.claim_refs:
                    t.claim_refs.append(claim.id)
        return True

    def add_target(self, target: Target) -> None:
        if not target.id:
            target.id = self.next_target_id()
        with self._lock:
            self.targets.append(target)
            self._target_index[target.id] = target

    def add_obligation(self, ob: Obligation) -> None:
        with self._lock:
            if not ob.id:
                self._obligation_counter += 1
                ob.id = f"o{self._obligation_counter}"
            self.obligations.append(ob)
            self._obligation_index[ob.id] = ob

    def add_unit(self, unit: Unit) -> Unit:
        """Add a unit, deduped by (obligation, normalized name).

        Returns the canonical unit (existing one on dedup hit)."""
        key = (unit.obligation_ref, " ".join(unit.name.lower().split())[:80])
        with self._lock:
            existing_id = self._unit_name_index.get(key)
            if existing_id is not None:
                canonical = self._unit_index[existing_id]
                for cid in unit.claim_refs:
                    if cid not in canonical.claim_refs:
                        canonical.claim_refs.append(cid)
                if canonical.status == "discovered" and canonical.claim_refs:
                    canonical.status = "evidenced"
                return canonical
            if not unit.id:
                self._unit_counter += 1
                unit.id = f"u{self._unit_counter}"
            if unit.claim_refs and unit.status == "discovered":
                unit.status = "evidenced"
            self.units.append(unit)
            self._unit_index[unit.id] = unit
            self._unit_name_index[key] = unit.id
            return unit

    def bind_claim_to_units(self, claim_id: str, unit_ids: list[str]) -> int:
        claim = self.find_claim(claim_id)
        if claim is None:
            return 0
        bound = 0
        with self._lock:
            for uid in unit_ids:
                unit = self._unit_index.get(uid)
                if unit is None:
                    continue
                if claim_id not in unit.claim_refs:
                    unit.claim_refs.append(claim_id)
                    bound += 1
                if unit.status == "discovered":
                    unit.status = "evidenced"
        return bound

    def find_obligation(self, ob_id: str) -> Obligation | None:
        return self._obligation_index.get(ob_id)

    def find_unit(self, unit_id: str) -> Unit | None:
        return self._unit_index.get(unit_id)

    def units_for(self, obligation_id: str) -> list[Unit]:
        return [u for u in self.units if u.obligation_ref == obligation_id]

    def open_mandatory_obligations(self) -> list[Obligation]:
        """Only set-valued obligations gate convergence — native-complete and
        summary substance is what targets already track."""
        return [
            o for o in self.obligations
            if o.mandatory and o.status == "open" and o.set_valued
        ]

    def obligation_card(self, ob: Obligation) -> dict:
        """Aggregate counts only — units never appear individually to the
        controller."""
        units = self.units_for(ob.id)
        active = [u for u in units if u.status != "waived"]
        return {
            "id": ob.id,
            "text": ob.text[:140],
            "coverage": ob.coverage,
            "mandatory": ob.mandatory,
            "status": ob.status,
            "units": len(active),
            "evidenced": sum(1 for u in active if u.status in ("evidenced", "analyzed")),
            "unevidenced": sum(1 for u in active if u.status == "discovered"),
        }

    def bind_claim(self, claim_id: str, target_ids: list[str]) -> bool:
        """Attach a claim to targets. Returns True if anything changed."""
        claim = self.find_claim(claim_id)
        if claim is None:
            return False
        changed = False
        with self._lock:
            for tid in target_ids:
                target = self._target_index.get(tid)
                if target is None:
                    continue
                if tid not in claim.target_refs:
                    claim.target_refs.append(tid)
                    changed = True
                if claim_id not in target.claim_refs:
                    target.claim_refs.append(claim_id)
                    changed = True
        return changed

    def resolve_target(self, target_id: str, status: str, reason: str) -> bool:
        if status not in ("closed", "waived", "blocked", "open"):
            return False
        target = self.find_target(target_id)
        if target is None:
            return False
        target.status = status
        target.reason = reason
        if status == "open":
            target.resolved_iteration = None
            target.closure_basis = []
        else:
            target.resolved_iteration = self.iteration
            if status == "closed":
                target.closure_basis = list(target.claim_refs)
        return True

    def reopen_candidates(self) -> list[dict]:
        """Closed targets whose closure basis later evidence has disturbed.

        Structural facts only: new claims bound after closure, or basis
        claims contradicted/superseded. Whether to actually reopen is a
        judgment call — the reframe pass decides.
        """
        out = []
        for t in self.targets:
            if t.status != "closed":
                continue
            new_claims = [c for c in t.claim_refs if c not in t.closure_basis]
            basis = [self._claim_index.get(cid) for cid in t.closure_basis]
            disturbed = [
                c.id for c in basis
                if c is not None and (c.superseded or c.contradicts_refs or any(
                    c.id in (o.contradicts_refs or [])
                    for o in self.claims if o.active
                ))
            ]
            if new_claims or disturbed:
                out.append({
                    "target_id": t.id, "need": t.need[:120],
                    "new_claims": len(new_claims),
                    "disturbed_basis": len(disturbed),
                })
        return out

    # --- Lookup ---

    def find_claim(self, claim_id: str) -> Claim | None:
        return self._claim_index.get(claim_id)

    def find_target(self, target_id: str) -> Target | None:
        return self._target_index.get(target_id)

    def find_source(self, source_id: str) -> Source | None:
        return self._source_index.get(source_id)

    # --- Token accounting (mirrors Blackboard.add_tokens) ---

    def add_tokens(self, tokens_in: int, tokens_out: int, model: str) -> None:
        with self._lock:
            total = tokens_in + tokens_out
            self.total_tokens_used += total
            self.tokens_input += tokens_in
            self.tokens_output += tokens_out
            if model:
                if model not in self.cost_by_model:
                    self.cost_by_model[model] = {
                        "input": 0, "output": 0, "total": 0, "calls": 0,
                    }
                self.cost_by_model[model]["input"] += tokens_in
                self.cost_by_model[model]["output"] += tokens_out
                self.cost_by_model[model]["total"] += total
                self.cost_by_model[model]["calls"] += 1

    def budget_used_pct(self) -> float:
        return round(self.total_tokens_used / max(self.token_budget, 1) * 100, 1)

    # --- Bookkeeping (structural facts only — no semantic judgment) ---

    def claims_for_target(self, target: Target) -> list[Claim]:
        return [
            c for c in (self._claim_index.get(cid) for cid in target.claim_refs)
            if c is not None and c.active
        ]

    def unbound_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.active and not c.target_refs]

    def target_blockers(self, target: Target) -> list[str]:
        """Computed structural blockers — why this target cannot close yet."""
        if not target.is_open:
            return []
        bound = self.claims_for_target(target)
        blockers = []
        raw = [c for c in bound if c.kind == "observation"]
        derived = [c for c in bound if c.is_derived]
        if not bound:
            blockers.append("needs_evidence")
        elif raw and not derived:
            blockers.append("needs_analysis")
        contradicted = [
            c for c in derived
            if c.contradicts_refs or any(
                c.id in (o.contradicts_refs or [])
                for o in bound if o.id != c.id
            )
        ]
        if contradicted:
            blockers.append("has_contradiction")
        unverified_material = [
            c for c in derived
            if target.rank >= 2 and c.verified is None and c.confidence < 0.55
        ]
        if unverified_material:
            blockers.append("needs_verification")
        return blockers

    def target_card(self, target: Target) -> dict:
        """Compact card the controller sees — counts and blockers, not the graph."""
        bound = self.claims_for_target(target)
        derived = [c for c in bound if c.is_derived]
        best = sorted(derived or bound, key=lambda c: -c.confidence)[:8]
        return {
            "id": target.id,
            "need": target.need,
            "materiality": target.materiality,
            "status": target.status,
            "blockers": self.target_blockers(target),
            "claims_bound": len(bound),
            "raw": sum(1 for c in bound if c.kind == "observation"),
            "derived": len(derived),
            "kinds": sorted({c.kind for c in derived}),
            "best_claims": [
                {"id": c.id, "kind": c.kind, "content": c.content}
                for c in best
            ],
        }

    def open_targets(self) -> list[Target]:
        return sorted(
            (t for t in self.targets if t.is_open),
            key=lambda t: -t.rank,
        )

    def resolved_targets(self) -> list[Target]:
        return [t for t in self.targets if not t.is_open]

    def material_open_targets(self) -> list[Target]:
        return [t for t in self.open_targets() if t.rank >= 2]

    # --- Events / observability ---

    def log(self, kind: str, summary: str, detail: dict | None = None,
            model: str = "", tokens: int = 0,
            tokens_in: int = 0, tokens_out: int = 0) -> None:
        ev = Event(
            iteration=self.iteration, kind=kind, summary=summary,
            detail=detail or {}, model=model, tokens=tokens,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
        with self._lock:
            self.events.append(ev)
        self._append_event_file(ev)

    def _append_event_file(self, ev: Event) -> None:
        if not self.output_dir:
            return
        try:
            d = Path(self.output_dir) / "loop"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")
        except OSError:
            pass

    def recent_events(self, iteration: int) -> list[Event]:
        return [e for e in self.events if e.iteration == iteration]

    # --- Snapshots ---

   def snapshot(self, label: str = "") -> None:
        if not self.output_dir:
            return
        d = Path(self.output_dir) / "loop"
        d.mkdir(parents=True, exist_ok=True)
        import re
        clean_label = re.sub(r"[^\w\-]", "", label) if label else ""
        suffix = f"_{clean_label}" if clean_label else ""
        path = (d / f"board_iter_{self.iteration}{suffix}.json").resolve()
        if not str(path).startswith(str(d.resolve())):
            return
        data = {
            "instruction": self.instruction,
            "iteration": self.iteration,
            "stop_reason": self.stop_reason,
            "sources": [s.to_dict() for s in self.sources],
            "claims": [c.to_dict() for c in self.claims],
            "targets": [
                {**t.to_dict(), "blockers": self.target_blockers(t)}
                for t in self.targets
            ],
            "obligations": [o.to_dict() for o in self.obligations],
            "units": [u.to_dict() for u in self.units],
            "total_tokens_used": self.total_tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "token_budget": self.token_budget,
            "budget_used_pct": self.budget_used_pct(),
            "cost_by_model": dict(self.cost_by_model),
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
