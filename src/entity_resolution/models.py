from __future__ import annotations

"""Small immutable data contracts used by the resolver pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompanyRecord:
    record_id: str
    name: str
    document_id: str | None = None
    snippets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompanyNameForms:
    raw: str
    normalized: str
    core: str
    tokens: tuple[str, ...]
    sorted_tokens: str


@dataclass(frozen=True)
class ResolverConfig:
    auto_merge_threshold: float = 0.92
    review_threshold: float = 0.68


@dataclass(frozen=True)
class CompanyCandidate:
    left_record_id: str
    right_record_id: str
    score: float
    tier: str
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_record_id": self.left_record_id,
            "right_record_id": self.right_record_id,
            "score": round(self.score, 6),
            "tier": self.tier,
            "evidence": list(self.evidence),
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class DuplicateNameReport:
    report_id: str
    status: str
    user_review_required: bool
    names: tuple[str, ...]
    record_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "status": self.status,
            "user_review_required": self.user_review_required,
            "names": list(self.names),
            "record_ids": list(self.record_ids),
            "document_ids": list(self.document_ids),
            "evidence": list(self.evidence),
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class CompanyCluster:
    cluster_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    member_record_ids: tuple[str, ...]
    supporting_metadata: dict[str, Any] = field(default_factory=dict)
    evidence_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "member_record_ids": list(self.member_record_ids),
            "supporting_metadata": self.supporting_metadata,
            "evidence_notes": list(self.evidence_notes),
        }


@dataclass(frozen=True)
class ResolutionResult:
    clusters: tuple[CompanyCluster, ...] = ()
    auto_matches: tuple[CompanyCandidate, ...] = ()
    duplicate_name_reports: tuple[DuplicateNameReport, ...] = ()
    review_candidates: tuple[CompanyCandidate, ...] = ()
    rejected_candidates: tuple[CompanyCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "auto_matches": [candidate.to_dict() for candidate in self.auto_matches],
            "duplicate_name_reports": [report.to_dict() for report in self.duplicate_name_reports],
            "review_candidates": [candidate.to_dict() for candidate in self.review_candidates],
            "rejected_candidates": [candidate.to_dict() for candidate in self.rejected_candidates],
        }
