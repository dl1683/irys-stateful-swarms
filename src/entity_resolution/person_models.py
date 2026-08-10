from __future__ import annotations

"""Immutable contracts for conservative person connection screening."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonAttribute:
    kind: str
    value: str
    source_document_id: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    verified: bool = False
    notes: str | None = None
    qualifiers: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "source_document_id": self.source_document_id,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "verified": self.verified,
            "notes": self.notes,
            "qualifiers": dict(self.qualifiers),
        }


@dataclass(frozen=True)
class PersonRecord:
    record_id: str
    name: str
    document_id: str | None = None
    snippets: tuple[str, ...] = ()
    attributes: tuple[PersonAttribute, ...] = ()


@dataclass(frozen=True)
class PersonNameForms:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    sorted_tokens: str


@dataclass(frozen=True)
class PersonEvidence:
    kind: str
    relationship: str
    left_values: tuple[str, ...]
    right_values: tuple[str, ...]
    score_contribution: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relationship": self.relationship,
            "left_values": list(self.left_values),
            "right_values": list(self.right_values),
            "score_contribution": round(self.score_contribution, 6),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PersonProfile:
    profile_id: str
    canonical_name: str
    names: tuple[str, ...]
    member_record_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    snippets: tuple[str, ...]
    attributes: tuple[PersonAttribute, ...]
    confirmed_by: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "canonical_name": self.canonical_name,
            "names": list(self.names),
            "member_record_ids": list(self.member_record_ids),
            "document_ids": list(self.document_ids),
            "snippets": list(self.snippets),
            "attributes": [attribute.to_dict() for attribute in self.attributes],
            "confirmed_by": list(self.confirmed_by),
        }


@dataclass(frozen=True)
class PersonAutoMatch:
    left_record_id: str
    right_record_id: str
    matching_verified_government_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_record_id": self.left_record_id,
            "right_record_id": self.right_record_id,
            "matching_verified_government_ids": list(self.matching_verified_government_ids),
        }


@dataclass(frozen=True)
class UncertainConnection:
    connection_id: str
    status: str
    human_or_llm_review_required: bool
    left: PersonProfile
    right: PersonProfile
    score: float
    evidence: tuple[PersonEvidence, ...]
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "status": self.status,
            "human_or_llm_review_required": self.human_or_llm_review_required,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "score": round(self.score, 6),
            "evidence": [item.to_dict() for item in self.evidence],
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PersonResolutionResult:
    confirmed_profiles: tuple[PersonProfile, ...] = ()
    auto_matches: tuple[PersonAutoMatch, ...] = ()
    uncertain_connections: tuple[UncertainConnection, ...] = ()
    screened_out_profile_pairs: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_profiles": [profile.to_dict() for profile in self.confirmed_profiles],
            "auto_matches": [match.to_dict() for match in self.auto_matches],
            "uncertain_connections": [connection.to_dict() for connection in self.uncertain_connections],
            "screened_out_profile_pairs": [list(pair) for pair in self.screened_out_profile_pairs],
        }
