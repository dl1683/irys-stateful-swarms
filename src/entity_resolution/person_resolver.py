from __future__ import annotations

"""Conservative person aggregation and connection screening."""

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
from itertools import combinations

from .person_models import (
    PersonAttribute,
    PersonAutoMatch,
    PersonEvidence,
    PersonProfile,
    PersonRecord,
    PersonResolutionResult,
    UncertainConnection,
)
from .person_normalization import (
    canonicalize_person_attribute_kind,
    government_id_components,
    government_id_key,
    normalize_person_attribute,
    normalize_person_name,
    person_name_similarity,
)


MAX_CONTEXT_BLOCK_SIZE = 500
CONNECTION_THRESHOLD = 0.30


class _UnionFind:
    def __init__(self, record_ids: Iterable[str]) -> None:
        self.parent = {record_id: record_id for record_id in record_ids}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            winner, loser = sorted((left_root, right_root))
            self.parent[loser] = winner


def _stable_id(prefix: str, values: Iterable[str]) -> str:
    digest = hashlib.sha1(
        json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _attribute_sort_key(attribute: PersonAttribute) -> tuple[object, ...]:
    return (
        canonicalize_person_attribute_kind(attribute.kind),
        normalize_person_attribute(attribute),
        attribute.value,
        attribute.source_document_id or "",
        attribute.effective_from or "",
        attribute.effective_to or "",
        attribute.verified,
        attribute.notes or "",
        attribute.qualifiers,
    )


def _verified_ids(record: PersonRecord) -> set[str]:
    return {key for attribute in record.attributes if (key := government_id_key(attribute))}


def _verified_values(entity: PersonRecord | PersonProfile, kind: str) -> set[str]:
    kind = canonicalize_person_attribute_kind(kind)
    return {
        normalize_person_attribute(attribute)
        for attribute in entity.attributes
        if canonicalize_person_attribute_kind(attribute.kind) == kind
        and attribute.verified
        and normalize_person_attribute(attribute)
    }


def _entity_names(entity: PersonRecord | PersonProfile) -> tuple[str, ...]:
    primary_names = entity.names if isinstance(entity, PersonProfile) else (entity.name,)
    aliases = tuple(
        attribute.value
        for attribute in entity.attributes
        if canonicalize_person_attribute_kind(attribute.kind) == "alias"
    )
    return primary_names + aliases


def _names_compatible(left: PersonRecord | PersonProfile, right: PersonRecord | PersonProfile) -> bool:
    left_names = _entity_names(left)
    right_names = _entity_names(right)
    return any(person_name_similarity(left_name, right_name) >= 0.82 for left_name in left_names for right_name in right_names)


def _verified_identity_conflicts(
    entities: Iterable[PersonRecord | PersonProfile],
) -> tuple[str, ...]:
    entities = tuple(entities)
    conflicts: list[str] = []
    verified_birth_dates = {
        value
        for entity in entities
        for value in _verified_values(entity, "birth_date")
    }
    if len(verified_birth_dates) > 1:
        conflicts.append("verified_birth_date_conflict")

    identifiers_by_scheme: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entity in entities:
        for attribute in entity.attributes:
            components = government_id_components(attribute)
            if components is None:
                continue
            issuer, identifier_type, identifier = components
            identifiers_by_scheme[(issuer, identifier_type)].add(identifier)
    if any(len(identifiers) > 1 for identifiers in identifiers_by_scheme.values()):
        conflicts.append("competing_verified_government_ids")
    return tuple(conflicts)


def _pair_conflicts(left: PersonRecord, right: PersonRecord) -> tuple[str, ...]:
    conflicts = list(_verified_identity_conflicts((left, right)))
    if set(_verified_ids(left)) & set(_verified_ids(right)) and not _names_compatible(left, right):
        conflicts.append("incompatible_names_on_shared_government_id")
    return tuple(conflicts)


def _group_conflicts(records: Iterable[PersonRecord]) -> tuple[str, ...]:
    records = tuple(records)
    conflicts = list(_verified_identity_conflicts(records))
    if any(not _names_compatible(left, right) for left, right in combinations(records, 2)):
        conflicts.append("incompatible_names_on_shared_government_id")
    return tuple(conflicts)


def _profile(records: list[PersonRecord], confirmed_by: Iterable[str]) -> PersonProfile:
    member_ids = tuple(sorted(record.record_id for record in records))
    names = tuple(sorted({record.name for record in records}, key=lambda value: (value.casefold(), value)))
    attributes = tuple(sorted({attribute for record in records for attribute in record.attributes}, key=_attribute_sort_key))
    document_ids = tuple(sorted({record.document_id for record in records if record.document_id}))
    snippets = tuple(sorted({snippet for record in records for snippet in record.snippets}))
    canonical_name = sorted(names, key=lambda value: (-len(normalize_person_name(value).tokens), -len(value), value.casefold(), value))[0]
    return PersonProfile(
        profile_id=_stable_id("person_profile", member_ids),
        canonical_name=canonical_name,
        names=names,
        member_record_ids=member_ids,
        document_ids=document_ids,
        snippets=snippets,
        attributes=attributes,
        confirmed_by=tuple(sorted(set(confirmed_by))),
    )


def _connection(
    left: PersonProfile,
    right: PersonProfile,
    score: float,
    evidence: Iterable[PersonEvidence],
    conflicts: Iterable[str],
    warnings: Iterable[str],
) -> UncertainConnection:
    ordered_left, ordered_right = sorted((left, right), key=lambda profile: profile.profile_id)
    return UncertainConnection(
        connection_id=_stable_id("person_connection", (ordered_left.profile_id, ordered_right.profile_id)),
        status="uncertain_connection",
        human_or_llm_review_required=True,
        left=ordered_left,
        right=ordered_right,
        score=max(0.0, min(1.0, score)),
        evidence=tuple(evidence),
        conflicts=tuple(sorted(set(conflicts))),
        warnings=tuple(sorted(set(warnings))),
    )


def _profile_values(profile: PersonProfile, kind: str) -> tuple[str, ...]:
    kind = canonicalize_person_attribute_kind(kind)
    return tuple(sorted({
        normalize_person_attribute(attribute)
        for attribute in profile.attributes
        if canonicalize_person_attribute_kind(attribute.kind) == kind and normalize_person_attribute(attribute)
    }))


def _shared_values(left: PersonProfile, right: PersonProfile, kind: str) -> tuple[str, ...]:
    return tuple(sorted(set(_profile_values(left, kind)) & set(_profile_values(right, kind))))


def _profile_government_ids(profile: PersonProfile) -> tuple[str, ...]:
    return tuple(sorted({
        identifier
        for attribute in profile.attributes
        if (identifier := government_id_key(attribute))
    }))


def _identity_warning_needed(
    shared_government_ids: tuple[str, ...],
    conflicts: Iterable[str],
) -> bool:
    conflict_set = set(conflicts)
    return bool(shared_government_ids and conflict_set) or (
        "competing_verified_government_ids" in conflict_set
    )


def _name_evidence(left: PersonProfile, right: PersonProfile) -> PersonEvidence | None:
    pairs = [
        (person_name_similarity(left_name, right_name), left_name, right_name)
        for left_name in left.names
        for right_name in right.names
    ]
    similarity, left_name, right_name = max(pairs)
    left_forms = normalize_person_name(left_name)
    right_forms = normalize_person_name(right_name)
    if left_forms.normalized == right_forms.normalized:
        relationship, contribution, detail = "exact_match", 0.55, "Exact normalized name; common names remain ambiguous"
    elif left_forms.sorted_tokens == right_forms.sorted_tokens:
        relationship, contribution, detail = "reordered_match", 0.50, "Same normalized name tokens in a different order"
    elif similarity >= 0.82:
        relationship, contribution, detail = "similar", 0.35, f"Name similarity {similarity:.3f}"
    else:
        return None
    return PersonEvidence("name", relationship, (left_name,), (right_name,), contribution, detail)


ATTRIBUTE_WEIGHTS = {
    "birth_date": 0.30,
    "email": 0.35,
    "phone": 0.35,
    "residence": 0.18,
    "birthplace": 0.15,
    "workplace": 0.15,
    "profession": 0.08,
    "nationality": 0.05,
    "gender": 0.02,
    "family_member": 0.25,
    "alias": 0.45,
}


def _family_reference_evidence(left: PersonProfile, right: PersonProfile) -> PersonEvidence | None:
    left_family = set(_profile_values(left, "family_member"))
    right_family = set(_profile_values(right, "family_member"))
    left_names = {normalize_person_name(name).normalized for name in left.names}
    right_names = {normalize_person_name(name).normalized for name in right.names}
    left_references = tuple(sorted(left_family & right_names))
    right_references = tuple(sorted(right_family & left_names))
    if not left_references and not right_references:
        return None
    return PersonEvidence(
        kind="family_member",
        relationship="declared_reference",
        left_values=left_references,
        right_values=right_references,
        score_contribution=ATTRIBUTE_WEIGHTS["family_member"],
        detail="At least one profile names the other person as a family member; the relationship itself is not inferred",
    )


def _attribute_evidence(left: PersonProfile, right: PersonProfile) -> tuple[PersonEvidence, ...]:
    evidence: list[PersonEvidence] = []
    for kind, contribution in ATTRIBUTE_WEIGHTS.items():
        if kind == "family_member":
            continue
        shared = _shared_values(left, right, kind)
        if shared:
            evidence.append(PersonEvidence(
                kind=kind,
                relationship="exact_normalized_match",
                left_values=shared,
                right_values=shared,
                score_contribution=contribution,
                detail=f"Shared normalized {kind} value",
            ))
    if family_reference := _family_reference_evidence(left, right):
        evidence.append(family_reference)
    shared_government_ids = tuple(sorted(set(_profile_government_ids(left)) & set(_profile_government_ids(right))))
    if shared_government_ids:
        evidence.append(PersonEvidence(
            kind="government_id",
            relationship="exact_verified_match",
            left_values=shared_government_ids,
            right_values=shared_government_ids,
            score_contribution=0.0,
            detail="Same verified, issuer-qualified government identifier; conflict prevents automatic aggregation",
        ))
    return tuple(evidence)


def _blocking_keys(profile: PersonProfile) -> set[str]:
    keys: set[str] = set()
    for name in profile.names:
        forms = normalize_person_name(name)
        if forms.normalized:
            keys.add("name:" + forms.normalized)
            keys.add("sorted_name:" + forms.sorted_tokens)
        for token in forms.tokens:
            if len(token) >= 3:
                keys.add("name_token:" + token)
                keys.add("name_prefix:" + token[:4])
    for kind in (*ATTRIBUTE_WEIGHTS, "government_id"):
        for value in _profile_values(profile, kind):
            keys.add(f"{kind}:{value}")
    return keys


def generate_person_candidate_pairs(profiles: Iterable[PersonProfile]) -> list[tuple[str, str]]:
    blocks: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        for key in _blocking_keys(profile):
            blocks[key].append(profile.profile_id)
    pairs: set[tuple[str, str]] = set()
    for key, profile_ids in blocks.items():
        unique = sorted(set(profile_ids))
        block_kind = key.split(":", 1)[0]
        if len(unique) > MAX_CONTEXT_BLOCK_SIZE and block_kind not in {"government_id", "email", "phone"}:
            continue
        for index, left_id in enumerate(unique):
            for right_id in unique[index + 1:]:
                pairs.add((left_id, right_id))
    return sorted(pairs)


def _screen_profile_pair(left: PersonProfile, right: PersonProfile) -> UncertainConnection | None:
    evidence = list(_attribute_evidence(left, right))
    if name_item := _name_evidence(left, right):
        evidence.insert(0, name_item)
    context_items = [item for item in evidence if item.kind not in {"name", "nationality", "gender", "profession"}]
    score = sum(item.score_contribution for item in evidence)
    has_name_signal = any(item.kind == "name" for item in evidence)
    has_two_context_signals = len(context_items) >= 2
    if score < CONNECTION_THRESHOLD or not (has_name_signal or has_two_context_signals):
        return None

    conflicts = list(_verified_identity_conflicts((left, right)))
    shared_government_ids = tuple(sorted(set(_profile_government_ids(left)) & set(_profile_government_ids(right))))

    warnings = ["similarities_may_reflect_family_household_marriage_colleagues_or_coincidence"]
    if _identity_warning_needed(shared_government_ids, conflicts):
        warnings.append("possible_identity_misuse_or_data_error")
    return _connection(left, right, score, evidence, conflicts, warnings)


def resolve_people(records: Iterable[PersonRecord]) -> PersonResolutionResult:
    ordered_records = sorted(records, key=lambda record: record.record_id)
    if len({record.record_id for record in ordered_records}) != len(ordered_records):
        raise ValueError("record_id values must be unique")
    records_by_id = {record.record_id: record for record in ordered_records}
    id_blocks: dict[str, list[str]] = defaultdict(list)
    for record in ordered_records:
        for identifier in _verified_ids(record):
            id_blocks[identifier].append(record.record_id)

    proposed_pairs = sorted({
        (left_id, right_id, identifier)
        for identifier, record_ids in id_blocks.items()
        for index, left_id in enumerate(sorted(set(record_ids)))
        for right_id in sorted(set(record_ids))[index + 1:]
    })
    union_find = _UnionFind(records_by_id)
    auto_matches: list[PersonAutoMatch] = []
    rejected_id_pairs: list[tuple[str, str, str, tuple[str, ...]]] = []
    for left_id, right_id, identifier in proposed_pairs:
        left, right = records_by_id[left_id], records_by_id[right_id]
        left_root, right_root = union_find.find(left_id), union_find.find(right_id)
        prospective_records = [
            record
            for record in ordered_records
            if union_find.find(record.record_id) in {left_root, right_root}
        ]
        conflicts = tuple(dict.fromkeys((*_pair_conflicts(left, right), *_group_conflicts(prospective_records))))
        if conflicts:
            rejected_id_pairs.append((left_id, right_id, identifier, conflicts))
            continue
        union_find.union(left_id, right_id)
        auto_matches.append(PersonAutoMatch(left_id, right_id, (identifier,)))

    groups: dict[str, list[PersonRecord]] = defaultdict(list)
    for record in ordered_records:
        groups[union_find.find(record.record_id)].append(record)
    profiles = []
    for group in groups.values():
        confirmed_by = (
            (identifier for record in group for identifier in _verified_ids(record))
            if len(group) > 1
            else ()
        )
        profiles.append(_profile(group, confirmed_by))
    profiles.sort(key=lambda profile: profile.profile_id)
    confirmed_profiles = tuple(profile for profile in profiles if len(profile.member_record_ids) > 1)

    profile_by_record_id = {
        record_id: profile
        for profile in profiles
        for record_id in profile.member_record_ids
    }
    profiles_by_id = {profile.profile_id: profile for profile in profiles}
    rejected_pairs_by_profile_pair: dict[tuple[str, str], list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for left_id, right_id, identifier, conflicts in rejected_id_pairs:
        left_profile = profile_by_record_id[left_id]
        right_profile = profile_by_record_id[right_id]
        if left_profile.profile_id == right_profile.profile_id:
            continue
        profile_pair = tuple(sorted((left_profile.profile_id, right_profile.profile_id)))
        rejected_pairs_by_profile_pair[profile_pair].append((identifier, conflicts))

    forced_connections = []
    for left_profile_id, right_profile_id in sorted(rejected_pairs_by_profile_pair):
        rejected_items = sorted(rejected_pairs_by_profile_pair[(left_profile_id, right_profile_id)])
        forced_connections.append(
            _connection(
                profiles_by_id[left_profile_id],
                profiles_by_id[right_profile_id],
                1.0,
                tuple(
                    PersonEvidence(
                        "government_id",
                        "exact_match",
                        (identifier,),
                        (identifier,),
                        1.0,
                        "Same verified government identifier",
                    )
                    for identifier, _ in rejected_items
                ),
                tuple(sorted({
                    conflict
                    for _, item_conflicts in rejected_items
                    for conflict in item_conflicts
                })),
                ("possible_identity_misuse_or_data_error",),
            )
        )
    connections_by_id = {connection.connection_id: connection for connection in forced_connections}
    forced_connection_pairs = {
        tuple(sorted((connection.left.profile_id, connection.right.profile_id)))
        for connection in forced_connections
    }
    screened_out: list[tuple[str, str]] = []
    for left_id, right_id in generate_person_candidate_pairs(profiles):
        if (left_id, right_id) in forced_connection_pairs:
            continue
        connection = _screen_profile_pair(profiles_by_id[left_id], profiles_by_id[right_id])
        if connection is None:
            screened_out.append((left_id, right_id))
        else:
            connections_by_id[connection.connection_id] = connection

    return PersonResolutionResult(
        confirmed_profiles=confirmed_profiles,
        auto_matches=tuple(sorted(auto_matches, key=lambda match: (match.left_record_id, match.right_record_id))),
        uncertain_connections=tuple(sorted(connections_by_id.values(), key=lambda item: item.connection_id)),
        screened_out_profile_pairs=tuple(sorted(set(screened_out))),
    )
