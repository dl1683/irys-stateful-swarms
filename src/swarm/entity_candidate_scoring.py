from __future__ import annotations

"""Persist conservative duplicate candidates for changed entity profiles."""

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations

from src.entity_resolution import (
    CompanyRecord,
    PersonAttribute,
    PersonRecord,
    resolve_companies,
    resolve_people,
)
from src.entity_resolution.person_normalization import government_id_components

from .entity_maintenance_store import EntityMaintenanceConfig, EntityMaintenanceState


@dataclass(frozen=True)
class CandidateRefreshSummary:
    new_candidate_ids: tuple[str, ...] = ()
    changed_candidate_ids: tuple[str, ...] = ()
    reused_candidate_ids: tuple[str, ...] = ()
    review_candidate_ids: tuple[str, ...] = ()


def refresh_candidates(
    state: EntityMaintenanceState,
    dirty_profile_ids: Collection[str],
    config: EntityMaintenanceConfig,
) -> CandidateRefreshSummary:
    """Score dirty profile neighbours without changing profile membership."""
    dirty = set(dirty_profile_ids)
    profiles = {
        profile_id: profile
        for profile_id, profile in state.profiles.items()
        if isinstance(profile, Mapping) and profile.get("entity_type") in {"company", "person"}
    }
    candidates = [
        *(_company_candidates(profiles, dirty, config)),
        *(_person_candidates(profiles, dirty, config)),
        *(_intra_profile_candidates(profiles, dirty)),
    ]
    new: list[str] = []
    changed: list[str] = []
    reused: list[str] = []
    review: list[str] = []
    for candidate in sorted(candidates, key=lambda item: str(item["semantic_key"])):
        matching_id = _matching_candidate_id(state.candidates, candidate)
        if matching_id is not None:
            reused.append(matching_id)
            continue
        candidate_id = str(candidate["candidate_id"])
        existing = state.candidates.get(candidate_id)
        if existing is None:
            new.append(candidate_id)
        else:
            changed.append(candidate_id)
        state.candidates[candidate_id] = candidate
        if candidate["status"] == "pending_review":
            review.append(candidate_id)
    return CandidateRefreshSummary(
        tuple(sorted(new)),
        tuple(sorted(changed)),
        tuple(sorted(reused)),
        tuple(sorted(review)),
    )


def _company_candidates(
    profiles: Mapping[str, Mapping[str, object]],
    dirty: set[str],
    config: EntityMaintenanceConfig,
) -> Iterable[dict[str, object]]:
    company_profiles = {
        profile_id: profile
        for profile_id, profile in profiles.items()
        if profile.get("entity_type") == "company"
    }
    records: list[CompanyRecord] = []
    record_profiles: dict[str, str] = {}
    for profile_id, profile in sorted(company_profiles.items()):
        metadata = _company_metadata(profile)
        for index, name in enumerate(_profile_names(profile)):
            record_id = f"{profile_id}#{index}"
            records.append(CompanyRecord(record_id, name, metadata=metadata))
            record_profiles[record_id] = profile_id

    by_pair: dict[tuple[str, str], list[object]] = defaultdict(list)
    result = resolve_companies(records)
    for resolver_candidate in (
        *result.auto_matches,
        *result.review_candidates,
        *result.rejected_candidates,
    ):
        profile_ids = tuple(sorted({
            record_profiles[resolver_candidate.left_record_id],
            record_profiles[resolver_candidate.right_record_id],
        }))
        if len(profile_ids) == 2 and dirty.intersection(profile_ids):
            by_pair[profile_ids].append(resolver_candidate)

    for profile_ids, resolver_candidates in sorted(by_pair.items()):
        winner = max(resolver_candidates, key=lambda item: item.score)
        evidence = sorted({note for item in resolver_candidates for note in item.evidence})
        conflicts = set(note for item in resolver_candidates for note in item.conflicts)
        conflicts.update(_company_verified_conflicts(
            company_profiles[profile_ids[0]], company_profiles[profile_ids[1]],
        ))
        yield _candidate(
            "company", profile_ids, company_profiles, winner.score, evidence, sorted(conflicts), config,
        )


def _person_candidates(
    profiles: Mapping[str, Mapping[str, object]],
    dirty: set[str],
    config: EntityMaintenanceConfig,
) -> Iterable[dict[str, object]]:
    person_profiles = {
        profile_id: profile
        for profile_id, profile in profiles.items()
        if profile.get("entity_type") == "person"
    }
    records = [_person_record(profile_id, profile) for profile_id, profile in sorted(person_profiles.items())]
    result = resolve_people(records)
    candidate_data: dict[tuple[str, str], tuple[float, set[str], set[str]]] = {}

    def add(profile_ids: tuple[str, str], score: float, evidence: Iterable[str], conflicts: Iterable[str]) -> None:
        if not dirty.intersection(profile_ids):
            return
        current = candidate_data.get(profile_ids)
        if current is None:
            candidate_data[profile_ids] = (score, set(evidence), set(conflicts))
            return
        candidate_data[profile_ids] = (
            max(score, current[0]), current[1] | set(evidence), current[2] | set(conflicts),
        )

    for match in result.auto_matches:
        add(tuple(sorted((match.left_record_id, match.right_record_id))), 1.0, ("same_verified_government_id",), ())
    for connection in result.uncertain_connections:
        member_ids = sorted({
            *connection.left.member_record_ids,
            *connection.right.member_record_ids,
        })
        evidence = (item.kind + ":" + item.relationship for item in connection.evidence)
        for profile_ids in combinations(member_ids, 2):
            add(profile_ids, connection.score, evidence, connection.conflicts)

    for profile_ids, (score, evidence, conflicts) in sorted(candidate_data.items()):
        augmented_conflicts = conflicts | set(_person_verified_conflicts(
            person_profiles[profile_ids[0]], person_profiles[profile_ids[1]],
        ))
        yield _candidate(
            "person", profile_ids, person_profiles, score, sorted(evidence), sorted(augmented_conflicts), config,
        )


def _intra_profile_candidates(
    profiles: Mapping[str, Mapping[str, object]], dirty: set[str],
) -> Iterable[dict[str, object]]:
    for profile_id, profile in sorted(profiles.items()):
        if profile_id not in dirty:
            continue
        conflicts, source_card_groups = _intra_profile_conflicts(profile)
        if not conflicts:
            continue
        yield _candidate(
            str(profile["entity_type"]), (profile_id,), profiles, 0.0, (), conflicts, None,
            source_card_groups=source_card_groups,
        )


def _candidate(
    entity_type: str,
    profile_ids: tuple[str, ...],
    profiles: Mapping[str, Mapping[str, object]],
    score: float,
    evidence: Iterable[str],
    conflicts: Iterable[str],
    config: EntityMaintenanceConfig | None,
    *,
    source_card_groups: list[list[str]] | None = None,
) -> dict[str, object]:
    evidence = sorted(set(evidence))
    conflicts = sorted(set(conflicts))
    score = round(max(0.0, min(1.0, score)), 6)
    semantic_key = entity_type + ":" + "|".join(profile_ids)
    hard_conflict = bool(set(conflicts) & {
        "verified_registration_number_conflict",
        "verified_tax_id_conflict",
        "verified_birth_date_conflict",
        "competing_verified_government_ids",
    })
    status = "pending_review" if hard_conflict or (
        config is not None
        and score >= config.duplicate_review_threshold
        and _independent_evidence_count(evidence) >= 2
    ) else "ignored"
    fingerprint_payload = {
        "profile_ids": profile_ids,
        "profiles": [
            (profile_id, profiles[profile_id].get("revision"), profiles[profile_id].get("fingerprint"))
            for profile_id in profile_ids
        ],
        "score": score,
        "evidence": evidence,
        "conflicts": conflicts,
    }
    evidence_fingerprint = "candidate_fp_" + _digest(fingerprint_payload)
    return {
        "candidate_id": "candidate_" + _digest(semantic_key),
        "semantic_key": semantic_key,
        "evidence_fingerprint": evidence_fingerprint,
        "profile_ids": list(profile_ids),
        "source_card_groups": source_card_groups or [
            sorted(_source_card_ids(profiles[profile_id])) for profile_id in profile_ids
        ],
        "entity_type": entity_type,
        "score": score,
        "evidence": evidence,
        "conflicts": conflicts,
        "status": status,
    }


def _company_metadata(profile: Mapping[str, object]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for field in ("registration_number", "tax_id", "domain", "website", "email_domain", "address"):
        values = _fact_values(profile, {field})
        if values:
            metadata[field] = values[0]
    return metadata


def _person_record(profile_id: str, profile: Mapping[str, object]) -> PersonRecord:
    attributes: list[PersonAttribute] = []
    for fact in _facts(profile):
        field, value = fact.get("field"), fact.get("value")
        if not isinstance(field, str) or not isinstance(value, str) or field == "name":
            continue
        qualifiers = fact.get("qualifiers", {})
        attributes.append(PersonAttribute(
            kind=field,
            value=value,
            source_document_id=_string(fact.get("source_card_id")),
            verified=_verified(fact),
            qualifiers=tuple(sorted(qualifiers.items())) if isinstance(qualifiers, Mapping) else (),
        ))
    return PersonRecord(
        profile_id,
        _profile_names(profile)[0],
        attributes=tuple(attributes),
    )


def _profile_names(profile: Mapping[str, object]) -> list[str]:
    names = [profile.get("primary_name"), *(profile.get("aliases", []) if isinstance(profile.get("aliases"), list) else [])]
    return sorted({name.strip() for name in names if isinstance(name, str) and name.strip()}, key=lambda name: (name.casefold(), name))


def _facts(profile: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    facts = profile.get("facts", [])
    return (fact for fact in facts if isinstance(fact, Mapping)) if isinstance(facts, list) else ()


def _fact_values(profile: Mapping[str, object], fields: set[str], *, verified: bool = False) -> list[str]:
    return sorted({
        value
        for fact in _facts(profile)
        if fact.get("field") in fields
        and (not verified or _verified(fact))
        and isinstance(value := fact.get("normalized_value"), str)
        and value
    })


def _company_verified_conflicts(left: Mapping[str, object], right: Mapping[str, object]) -> tuple[str, ...]:
    conflicts = []
    for field, label in (
        ("registration_number", "verified_registration_number_conflict"),
        ("tax_id", "verified_tax_id_conflict"),
    ):
        left_values = set(_fact_values(left, {field}, verified=True))
        right_values = set(_fact_values(right, {field}, verified=True))
        if left_values and right_values and left_values.isdisjoint(right_values):
            conflicts.append(label)
    return tuple(conflicts)


def _person_verified_conflicts(left: Mapping[str, object], right: Mapping[str, object]) -> tuple[str, ...]:
    conflicts = []
    left_birth_dates = set(_fact_values(left, {"birth_date"}, verified=True))
    right_birth_dates = set(_fact_values(right, {"birth_date"}, verified=True))
    if left_birth_dates and right_birth_dates and left_birth_dates.isdisjoint(right_birth_dates):
        conflicts.append("verified_birth_date_conflict")
    if _government_id_values_by_scheme(left) and _government_id_values_by_scheme(right):
        for scheme in set(_government_id_values_by_scheme(left)) & set(_government_id_values_by_scheme(right)):
            if _government_id_values_by_scheme(left)[scheme].isdisjoint(_government_id_values_by_scheme(right)[scheme]):
                conflicts.append("competing_verified_government_ids")
                break
    return tuple(conflicts)


def _intra_profile_conflicts(profile: Mapping[str, object]) -> tuple[list[str], list[list[str]]]:
    fields = ("registration_number", "tax_id") if profile.get("entity_type") == "company" else ("birth_date",)
    groups: list[list[str]] = []
    conflicts: list[str] = []
    for field in fields:
        values = _verified_fact_cards(profile, field)
        if len(values) > 1:
            conflicts.append({
                "registration_number": "verified_registration_number_conflict",
                "tax_id": "verified_tax_id_conflict",
                "birth_date": "verified_birth_date_conflict",
            }[field])
            groups.extend(cards for _, cards in sorted(values.items()))
    if profile.get("entity_type") == "person":
        by_scheme: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for fact in _facts(profile):
            if fact.get("field") != "government_id" or not _verified(fact):
                continue
            attribute = PersonAttribute(
                "government_id", str(fact.get("value", "")),
                qualifiers=tuple(sorted(fact.get("qualifiers", {}).items())) if isinstance(fact.get("qualifiers"), Mapping) else (),
                verified=True,
            )
            if components := government_id_components(attribute):
                by_scheme[components[:2]][components[2]].add(_string(fact.get("source_card_id")) or "")
        if any(len(values) > 1 for values in by_scheme.values()):
            conflicts.append("competing_verified_government_ids")
            groups.extend(
                sorted(cards) for values in by_scheme.values() if len(values) > 1 for cards in values.values()
            )
    return sorted(set(conflicts)), [list(group) for group in sorted({tuple(group) for group in groups})]


def _verified_fact_cards(profile: Mapping[str, object], field: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for fact in _facts(profile):
        value = fact.get("normalized_value")
        if fact.get("field") == field and _verified(fact) and isinstance(value, str) and value:
            card_id = _string(fact.get("source_card_id"))
            if card_id:
                values[value].add(card_id)
    return values


def _government_id_values_by_scheme(profile: Mapping[str, object]) -> dict[tuple[str, str], set[str]]:
    by_scheme: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fact in _facts(profile):
        if fact.get("field") != "government_id" or not _verified(fact):
            continue
        qualifiers = fact.get("qualifiers", {})
        attribute = PersonAttribute(
            "government_id", str(fact.get("value", "")),
            qualifiers=tuple(sorted(qualifiers.items())) if isinstance(qualifiers, Mapping) else (),
            verified=True,
        )
        if components := government_id_components(attribute):
            by_scheme[components[:2]].add(components[2])
    return by_scheme


def _independent_evidence_count(evidence: Iterable[str]) -> int:
    categories = set()
    for item in evidence:
        if item.startswith(("exact_", "sorted_", "shared_", "token_", "rarity_", "fuzzy_", "acronym_", "related_", "name:")):
            categories.add("name")
        elif item.startswith("same_"):
            categories.add(item)
        elif ":" in item:
            categories.add(item.split(":", 1)[0])
    return len(categories)


def _source_card_ids(profile: Mapping[str, object]) -> list[str]:
    source_card_ids = profile.get("source_card_ids", [])
    return [card_id for card_id in source_card_ids if isinstance(card_id, str)] if isinstance(source_card_ids, list) else []


def _matching_candidate_id(
    candidates: Mapping[str, Mapping[str, object]], candidate: Mapping[str, object],
) -> str | None:
    for candidate_id, existing in candidates.items():
        if (
            existing.get("semantic_key") == candidate["semantic_key"]
            and existing.get("evidence_fingerprint") == candidate["evidence_fingerprint"]
        ):
            return candidate_id
    return None


def _verified(fact: Mapping[str, object]) -> bool:
    return fact.get("verified") is True or fact.get("status") == "verified"


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
