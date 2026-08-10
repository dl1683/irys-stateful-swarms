from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .entity_maintenance_store import EntityMaintenanceState
from .models import ModelCaller
from .worker_dispatch import parse_json_object


CONFIRMED_OUTCOMES = frozenset({"same_entity", "same_name_distinct_entity"})
_OUTCOMES = CONFIRMED_OUTCOMES | {"uncertain"}


@dataclass(frozen=True)
class SpecialistReviewSummary:
    model_calls: int = 0
    reviewed_candidate_ids: tuple[str, ...] = ()
    retry_candidate_ids: tuple[str, ...] = ()


def review_pending_candidates(
    state: EntityMaintenanceState,
    candidate_ids: Sequence[str],
    caller: ModelCaller,
) -> SpecialistReviewSummary:
    """Review eligible candidates once each and retain every result in the sidecar."""
    reviewed: list[str] = []
    retries: list[str] = []
    calls = 0
    for candidate_id in sorted(set(candidate_ids)):
        candidate = state.candidates.get(candidate_id)
        if not isinstance(candidate, dict) or candidate.get("status") not in {
            "pending_review", "review_retry",
        }:
            continue
        calls += 1
        try:
            result = caller.complete(_prompt(candidate, state.profiles), max_tokens=1024)
        except Exception as exc:
            candidate["status"] = "review_retry"
            candidate["last_error"] = f"{type(exc).__name__}: {exc}"
            state.runs.append({
                "stage": "entity_specialist_review",
                "candidate_id": candidate_id,
                "error": candidate["last_error"],
            })
            retries.append(candidate_id)
            continue

        payload = parse_json_object(result.text)
        decision = _validated_decision(payload, _source_quotes(candidate, state.profiles))
        if decision is None:
            decision = {
                "decision": "uncertain",
                "rationale": "The specialist response did not meet the source-citation contract.",
                "citations": [],
            }
        outcome = str(decision["decision"])
        decision_id = _decision_id(str(candidate.get("semantic_key", candidate_id)))
        source_card_ids = sorted({
            citation["source_card_id"] for citation in decision["citations"]
        })
        state.decisions[decision_id] = {
            "decision_id": decision_id,
            "candidate_id": candidate_id,
            "semantic_key": candidate.get("semantic_key", candidate_id),
            "outcome": outcome,
            "evidence_fingerprint": candidate.get("evidence_fingerprint", ""),
            "profile_ids": _strings(candidate.get("profile_ids")),
            "source_card_ids": source_card_ids,
            "rationale": decision["rationale"],
            "citations": decision["citations"],
            "conflicts": _strings(candidate.get("conflicts")),
        }
        candidate["status"] = "reviewed" if outcome in CONFIRMED_OUTCOMES else "uncertain"
        candidate.pop("last_error", None)
        reviewed.append(candidate_id)
    return SpecialistReviewSummary(calls, tuple(reviewed), tuple(retries))


def _prompt(candidate: Mapping[str, object], profiles: Mapping[str, object]) -> str:
    profile_ids = _strings(candidate.get("profile_ids"))
    relevant_profiles = [
        _compact_profile(profile_id, profiles.get(profile_id))
        for profile_id in profile_ids
        if isinstance(profiles.get(profile_id), Mapping)
    ]
    source_quotes = _source_quotes(candidate, profiles)
    return "\n".join((
        "Decide whether the source-grounded entity profiles identify the same entity.",
        "Return exactly this JSON object with no additional keys:",
        '{"decision":"same_entity | same_name_distinct_entity | uncertain",'
        '"rationale":"short explanation grounded in supplied cards",'
        '"citations":[{"source_card_id":"e1","quote":"exact supplied quote"}]}',
        "Candidate:",
        json.dumps({
            "score": candidate.get("score"),
            "evidence": candidate.get("evidence", []),
            "conflicts": candidate.get("conflicts", []),
        }, sort_keys=True, separators=(",", ":")),
        "Profiles:",
        json.dumps(relevant_profiles, sort_keys=True, separators=(",", ":")),
        "Direct worker cards:",
        json.dumps([
            {"source_card_id": card_id, "quote": quote}
            for card_id, quote in sorted(source_quotes.items())
        ], sort_keys=True, separators=(",", ":")),
    ))


def _compact_profile(profile_id: str, profile: Mapping[str, object]) -> dict[str, object]:
    facts = profile.get("facts", [])
    return {
        "profile_id": profile_id,
        "entity_type": profile.get("entity_type"),
        "primary_name": profile.get("primary_name"),
        "aliases": profile.get("aliases", []),
        "facts": [
            {
                key: fact[key]
                for key in ("field", "value", "source_card_id", "verified", "qualifiers")
                if key in fact
            }
            for fact in facts
            if isinstance(fact, Mapping)
        ],
    }


def _source_quotes(
    candidate: Mapping[str, object], profiles: Mapping[str, object],
) -> dict[str, str]:
    allowed = {
        card_id
        for group in candidate.get("source_card_groups", [])
        if isinstance(group, list)
        for card_id in group
        if isinstance(card_id, str)
    }
    quotes: dict[str, str] = {}
    for profile_id in _strings(candidate.get("profile_ids")):
        profile = profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            continue
        facts = profile.get("facts", [])
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, Mapping):
                continue
            card_id, quote = fact.get("source_card_id"), fact.get("quote")
            if (
                isinstance(card_id, str)
                and card_id in allowed
                and isinstance(quote, str)
                and quote.strip()
            ):
                quotes.setdefault(card_id, quote)
    return quotes


def _validated_decision(
    payload: object, source_quotes: Mapping[str, str],
) -> dict[str, object] | None:
    if not isinstance(payload, Mapping) or set(payload) != {"decision", "rationale", "citations"}:
        return None
    outcome, rationale, citations = payload.get("decision"), payload.get("rationale"), payload.get("citations")
    if outcome not in _OUTCOMES or not isinstance(rationale, str) or not rationale.strip():
        return None
    if not isinstance(citations, list) or not citations:
        return None
    validated: list[dict[str, str]] = []
    for citation in citations:
        if not isinstance(citation, Mapping) or set(citation) != {"source_card_id", "quote"}:
            return None
        card_id, quote = citation.get("source_card_id"), citation.get("quote")
        if (
            not isinstance(card_id, str)
            or not isinstance(quote, str)
            or not quote.strip()
            or card_id not in source_quotes
            or quote not in source_quotes[card_id]
        ):
            return None
        validated.append({"source_card_id": card_id, "quote": quote})
    return {
        "decision": outcome,
        "rationale": rationale.strip(),
        "citations": validated,
    }


def _decision_id(semantic_key: str) -> str:
    digest = hashlib.sha256(semantic_key.encode("utf-8")).hexdigest()[:16]
    return f"entity-decision-{digest}"


def _strings(value: object) -> list[str]:
    return sorted({item for item in value if isinstance(item, str)}) if isinstance(value, list) else []
