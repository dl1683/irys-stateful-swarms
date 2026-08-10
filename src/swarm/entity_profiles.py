from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .blackboard import Blackboard
from .entity_annotations import ENTITY_TYPES
from .entity_maintenance_store import EntityMaintenanceState
from .entity_mention_repair import eligible_direct_entries
from .models import Entry, WorkerRecord


@dataclass(frozen=True)
class ProfileRefreshSummary:
    dirty_profile_ids: tuple[str, ...] = ()
    created_profile_ids: tuple[str, ...] = ()
    updated_profile_ids: tuple[str, ...] = ()


def refresh_entity_profiles(
    state: EntityMaintenanceState,
    entries: Iterable[Entry],
    min_card_count: int,
) -> ProfileRefreshSummary:
    """Refresh source-linked profiles from active direct-document cards only."""
    groups: dict[str, list[tuple[Entry, Mapping[str, object], Mapping[str, object]]]] = defaultdict(list)
    for entry in sorted(eligible_direct_entries(entries), key=lambda item: item.id):
        if entry.type == "entity_information":
            continue
        for index, annotation in enumerate(entry.entities):
            if not isinstance(annotation, Mapping):
                continue
            provenance = _provenance(entry, index)
            if provenance.get("method") not in {"worker", "deterministic_repair"}:
                continue
            entity_type = annotation.get("entity_type")
            name = annotation.get("name")
            if entity_type not in ENTITY_TYPES or not isinstance(name, str) or not name.strip():
                continue
            profile_id = _profile_id(entity_type, name)
            repaired_profile_id = provenance.get("profile_id")
            if (
                provenance.get("method") == "deterministic_repair"
                and isinstance(repaired_profile_id, str)
                and repaired_profile_id.startswith(f"{entity_type}:")
            ):
                profile_id = repaired_profile_id
            groups[profile_id].append((entry, annotation, provenance))

    created: list[str] = []
    updated: list[str] = []
    for profile_id in sorted(groups):
        mentions = groups[profile_id]
        source_card_ids = {entry.id for entry, _, _ in mentions if entry.id}
        existing = state.profiles.get(profile_id)
        if existing is None and len(source_card_ids) < min_card_count:
            continue
        payload = _profile_payload(profile_id, mentions)
        fingerprint = _profile_fingerprint(payload)
        if existing is not None and existing.get("fingerprint") == fingerprint:
            continue
        revision = 1 if existing is None else _revision(existing) + 1
        payload["revision"] = revision
        payload["fingerprint"] = fingerprint
        state.profiles[profile_id] = payload
        (created if existing is None else updated).append(profile_id)

    dirty = tuple(sorted((*created, *updated)))
    return ProfileRefreshSummary(dirty, tuple(created), tuple(updated))


def project_entity_information_cards(
    blackboard: Blackboard,
    state: EntityMaintenanceState,
    summary: ProfileRefreshSummary,
) -> tuple[str, ...]:
    """Project each changed profile to its single concise current context card."""
    projected: list[str] = []
    for profile_id in sorted(summary.dirty_profile_ids):
        profile = state.profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            continue
        card_id = f"entity-info-{profile_id.replace(':', '-')}"
        content = json.dumps(_card_payload(profile), sort_keys=True, separators=(",", ":"))
        card = blackboard.find_entry(card_id)
        if card is None:
            blackboard.add_entry(Entry(
                id=card_id,
                type="entity_information",
                content=content,
                created_by=WorkerRecord(
                    "entity_maintenance", "entity_information", blackboard.iteration,
                ),
                confidence=1.0,
                source=None,
            ))
        else:
            card.type = "entity_information"
            card.content = content
            card.source = None
            card.created_by = WorkerRecord(
                "entity_maintenance", "entity_information", blackboard.iteration,
            )
            card.confidence = 1.0
            card.status = "active"
        projected.append(card_id)
    return tuple(projected)


def _profile_payload(
    profile_id: str,
    mentions: list[tuple[Entry, Mapping[str, object], Mapping[str, object]]],
) -> dict[str, object]:
    entity_type, _ = profile_id.split(":", 1)
    names = sorted({str(annotation["name"]).strip() for _, annotation, _ in mentions}, key=_name_sort_key)
    primary_name = names[0]
    aliases = set(names[1:])
    facts: dict[tuple[str, str, str], dict[str, object]] = {}

    for entry, annotation, _ in mentions:
        name = str(annotation["name"]).strip()
        _add_fact(facts, entry, "name", name)
        attributes = annotation.get("attributes", [])
        if not isinstance(attributes, list):
            continue
        for attribute in attributes:
            if not isinstance(attribute, Mapping):
                continue
            field, value = attribute.get("kind"), attribute.get("value")
            if not isinstance(field, str) or not field.strip() or not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            _add_fact(
                facts, entry, field.strip(), value,
                verified=attribute.get("verified") is True,
                qualifiers=_qualifiers(attribute.get("qualifiers")),
            )
            if field == "alias" and value != primary_name:
                aliases.add(value)

    return {
        "profile_id": profile_id,
        "entity_type": entity_type,
        "primary_name": primary_name,
        "aliases": sorted(alias for alias in aliases if alias),
        "source_card_ids": sorted({entry.id for entry, _, _ in mentions if entry.id}),
        "facts": [facts[key] for key in sorted(facts)],
    }


def _add_fact(
    facts: dict[tuple[str, str, str], dict[str, object]],
    entry: Entry,
    field: str,
    value: str,
    *,
    verified: bool = False,
    qualifiers: dict[str, str] | None = None,
) -> None:
    normalized_value = _normalized_value(value)
    key = (field, normalized_value, entry.id)
    if key in facts:
        return
    quote = entry.source.evidence if entry.source else ""
    facts[key] = {
        "field": field,
        "value": value,
        "normalized_value": normalized_value,
        "source_card_id": entry.id,
        "source_document": entry.source.document if entry.source else None,
        "source_section": entry.source.section if entry.source else None,
        "quote": quote,
        "provenance_quality": "quoted_direct" if value in quote else "direct_worker_context",
        "status": "observed",
        "verified": verified,
        "qualifiers": qualifiers or {},
    }


def _card_payload(profile: Mapping[str, object]) -> dict[str, object]:
    facts_by_field: dict[str, list[str]] = {}
    for fact in profile.get("facts", []):
        if not isinstance(fact, Mapping):
            continue
        field, value = fact.get("field"), fact.get("value")
        if isinstance(field, str) and isinstance(value, str):
            facts_by_field.setdefault(field, []).append(value)
    return {
        "profile_id": profile.get("profile_id", ""),
        "primary_name": profile.get("primary_name", ""),
        "aliases": profile.get("aliases", []),
        "facts": {
            field: sorted(set(values), key=_name_sort_key)
            for field, values in sorted(facts_by_field.items())
        },
        "revision": profile.get("revision", 1),
    }


def _qualifiers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in sorted(value.items())
        if isinstance(key, str) and key and isinstance(item, str) and item
    }


def _provenance(entry: Entry, index: int) -> Mapping[str, object]:
    if index < len(entry.entity_annotation_provenance):
        provenance = entry.entity_annotation_provenance[index]
        if isinstance(provenance, Mapping):
            return provenance
    return {}


def _profile_id(entity_type: str, name: str) -> str:
    normalized_name = "-".join(re.findall(r"[\w]+", name.casefold()))
    return f"{entity_type}:{normalized_name}"


def _normalized_value(value: str) -> str:
    return "".join(re.findall(r"\w+", value.casefold()))


def _name_sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)


def _profile_fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "profile_fp_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _revision(profile: Mapping[str, object]) -> int:
    revision = profile.get("revision", 0)
    return revision if isinstance(revision, int) and revision >= 1 else 0
