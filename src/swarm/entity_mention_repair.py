from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .entity_annotations import ENTITY_TYPES
from .models import Entry


@dataclass(frozen=True)
class RepairSummary:
    changed_card_ids: tuple[str, ...] = ()
    repaired_mentions: int = 0
    untyped_mentions: int = 0


def entity_profile_id(entity_type: str, name: str) -> str:
    """Return the one profile-ID format used by workers and deterministic repair."""
    normalized_name = "-".join(re.findall(r"[\w]+", name.casefold()))
    return f"{entity_type}:{normalized_name}"


def eligible_direct_entries(entries: Iterable[Entry]) -> tuple[Entry, ...]:
    return tuple(
        entry for entry in entries
        if entry.status == "active" and entry.direct_document_context
    )


def build_entity_catalogue(
    entries: Iterable[Entry], profiles: Mapping[str, Mapping[str, object]]
) -> Mapping[str, Mapping[str, object]]:
    """Collect deterministic literal-match candidates from direct worker cards and profiles."""
    candidates: dict[str, tuple[tuple[int, str, str], dict[str, object]]] = {}

    def add_candidate(
        name: object,
        entity_type: object,
        profile_id: object,
        *,
        source_rank: int,
    ) -> None:
        if not isinstance(name, str):
            return
        normalized = _normalized_name(name)
        if not normalized:
            return
        safe_type = _safe_entity_type(entity_type)
        canonical_profile_id = entity_profile_id(safe_type or "untyped", name)
        if isinstance(profile_id, str) and profile_id:
            _, _, profile_name = profile_id.partition(":")
            canonical_profile_id = entity_profile_id(
                safe_type or "untyped", profile_name or name,
            )
        key = entity_profile_id(safe_type or "untyped", name)
        record = {
            "entity_type": safe_type,
            "normalized_name": normalized,
            "profile_id": canonical_profile_id,
        }
        priority = (source_rank, canonical_profile_id, name)
        current = candidates.get(key)
        if current is None or priority < current[0]:
            candidates[key] = (priority, record)

    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        entity_type = profile.get("entity_type")
        primary_name = profile.get("primary_name", profile.get("name"))
        add_candidate(primary_name, entity_type, profile_id, source_rank=0)
        aliases = profile.get("aliases", ())
        if isinstance(aliases, Iterable) and not isinstance(aliases, (str, bytes, Mapping)):
            for alias in aliases:
                add_candidate(alias, entity_type, profile_id, source_rank=0)

    for entry in sorted(eligible_direct_entries(entries), key=lambda entry: entry.id):
        for index, annotation in enumerate(entry.entities):
            if not isinstance(annotation, Mapping):
                continue
            provenance = (
                entry.entity_annotation_provenance[index]
                if index < len(entry.entity_annotation_provenance)
                and isinstance(entry.entity_annotation_provenance[index], Mapping)
                else {}
            )
            if provenance.get("method") not in ("worker", "deterministic_repair"):
                continue
            add_candidate(
                annotation.get("name"),
                annotation.get("entity_type"),
                provenance.get("profile_id"),
                source_rank=1,
            )

    return {
        key: record
        for key, (_, record) in sorted(candidates.items())
    }


def repair_entity_mentions(
    entries: Iterable[Entry],
    catalogue: Mapping[str, Mapping[str, object]],
    max_mentions_per_card: int,
    run_id: str,
) -> RepairSummary:
    """Append only missing, literal entity mentions on active direct-document cards."""
    candidates = _catalogue_candidates(catalogue)
    changed_card_ids: list[str] = []
    repaired_mentions = 0
    untyped_mentions = 0

    for entry in sorted(eligible_direct_entries(entries), key=lambda entry: entry.id):
        if max_mentions_per_card <= len(entry.entities):
            continue
        text = f"{entry.content}\n{entry.source.evidence if entry.source else ''}"
        existing_names = {
            _normalized_name(annotation.get("name"))
            for annotation in entry.entities
            if isinstance(annotation, Mapping) and isinstance(annotation.get("name"), str)
        }
        changed = False

        for candidate in candidates:
            if max_mentions_per_card <= len(entry.entities):
                break
            normalized = candidate["normalized_name"]
            if normalized in existing_names:
                continue
            matched = _literal_match(text, normalized)
            if matched is None:
                continue

            observed_text = _normalize_whitespace(matched.group("mention"))
            entity_type = _safe_entity_type(candidate["entity_type"])
            if entity_type is not None:
                entry.entities.append({
                    "entity_type": entity_type,
                    "name": observed_text,
                    "attributes": [],
                })
                entry.entity_annotation_provenance.append({
                    "method": "deterministic_repair",
                    "matched_text": observed_text,
                    "profile_id": candidate["profile_id"],
                    "run_id": run_id,
                })
                existing_names.add(normalized)
                repaired_mentions += 1
                changed = True
                continue

            rejection = f"untyped_repaired_mention:{normalized}"
            if rejection not in entry.entity_annotation_rejections:
                entry.entity_annotation_rejections.append(rejection)
                untyped_mentions += 1
                changed = True

        if changed:
            changed_card_ids.append(entry.id)

    return RepairSummary(
        changed_card_ids=tuple(changed_card_ids),
        repaired_mentions=repaired_mentions,
        untyped_mentions=untyped_mentions,
    )


def _catalogue_candidates(
    catalogue: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for key, record in catalogue.items():
        if not isinstance(record, Mapping):
            continue
        normalized = record.get("normalized_name")
        if isinstance(normalized, str) and normalized:
            entity_type = record.get("entity_type")
            profile_id = record.get("profile_id", key)
            records.append({
                "entity_type": _safe_entity_type(entity_type),
                "normalized_name": _normalized_name(normalized),
                "profile_id": profile_id if isinstance(profile_id, str) else str(key),
            })
            continue

        records.extend(build_entity_catalogue((), {str(key): record}).values())

    unique: dict[tuple[object, str], dict[str, object]] = {}
    for record in records:
        identity = (record["entity_type"], record["normalized_name"])
        current = unique.get(identity)
        if current is None or str(record["profile_id"]) < str(current["profile_id"]):
            unique[identity] = record
    return tuple(sorted(
        unique.values(),
        key=lambda record: (
            -len(str(record["normalized_name"])),
            str(record["profile_id"]),
            str(record["entity_type"]),
        ),
    ))


def _normalized_name(value: str) -> str:
    return _normalize_whitespace(value).casefold()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _literal_match(text: str, normalized_name: str) -> re.Match[str] | None:
    tokens = normalized_name.split(" ")
    literal = r"\s+".join(re.escape(token) for token in tokens)
    return re.search(rf"(?<!\w)(?P<mention>{literal})(?!\w)", text, re.IGNORECASE)


def _safe_entity_type(value: object) -> str | None:
    return value if isinstance(value, str) and value in ENTITY_TYPES else None
