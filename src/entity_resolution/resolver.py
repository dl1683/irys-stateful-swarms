from __future__ import annotations

"""Candidate blocking and deterministic union-find clustering."""

import hashlib
from dataclasses import replace
from collections import defaultdict
from collections.abc import Iterable

from .metadata import record_address, record_domain, record_registration
from .models import CompanyCandidate, CompanyCluster, CompanyRecord, DuplicateNameReport, ResolutionResult, ResolverConfig
from .normalization import company_initials, normalize_company_name
from .scoring import score_candidate
from .token_stats import WEAK_WORDS, token_weights


# ponytail: cap noisy blocks; exact/core blocks still provide the normal path.
MAX_NOISY_BLOCK_SIZE = 500


def _add_blocks(blocks: dict[str, list[str]], key: str, record_id: str) -> None:
    if key:
        blocks[key].append(record_id)


def generate_candidate_pairs(records: Iterable[CompanyRecord]) -> list[tuple[str, str]]:
    records_by_id = {record.record_id: record for record in records}
    blocks: dict[str, list[str]] = defaultdict(list)
    for record in records_by_id.values():
        forms = normalize_company_name(record.name)
        if forms.core:
            _add_blocks(blocks, "core:" + forms.core, record.record_id)
        if forms.sorted_tokens:
            _add_blocks(blocks, "sorted:" + forms.sorted_tokens, record.record_id)
        if forms.tokens and forms.tokens[0] not in WEAK_WORDS:
            _add_blocks(blocks, "first:" + forms.tokens[0], record.record_id)
        distinctive = [token for token in forms.tokens if token not in WEAK_WORDS]
        for token in distinctive:
            _add_blocks(blocks, "token:" + token, record.record_id)
        if initials := company_initials(forms.tokens):
            _add_blocks(blocks, "acronym:" + initials, record.record_id)
        domain = record_domain(record)
        if domain:
            _add_blocks(blocks, "domain:" + domain, record.record_id)
        registration = record_registration(record)
        if registration:
            _add_blocks(blocks, "registration:" + registration, record.record_id)
        address = record_address(record)
        if address:
            _add_blocks(blocks, "address:" + address, record.record_id)

    pairs: set[tuple[str, str]] = set()
    for block_key, record_ids in blocks.items():
        unique = sorted(set(record_ids))
        if len(unique) < 2:
            continue
        if block_key.split(":", 1)[0] in {"first", "token", "domain", "address", "acronym"} and len(unique) > MAX_NOISY_BLOCK_SIZE:
            continue
        for index, left_id in enumerate(unique):
            for right_id in unique[index + 1 :]:
                pairs.add((left_id, right_id))
    return sorted(pairs)


class _UnionFind:
    def __init__(self, ids: Iterable[str]) -> None:
        self.parent = {record_id: record_id for record_id in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        winner, loser = sorted((left_root, right_root))
        self.parent[loser] = winner


def _cluster_id(member_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha1("|".join(member_ids).encode("utf-8")).hexdigest()[:12]
    return "cluster_" + digest


def _canonical_name(records: list[CompanyRecord]) -> str:
    return sorted((record.name for record in records), key=lambda value: (len(normalize_company_name(value).core), value.lower(), value))[0]


def _supporting_metadata(records: list[CompanyRecord]) -> dict[str, object]:
    domains = sorted({domain for record in records if (domain := record_domain(record))})
    registrations = sorted({reg for record in records if (reg := record_registration(record))})
    output: dict[str, object] = {}
    if domains:
        output["domains"] = domains
    if registrations:
        output["registration_numbers"] = registrations
    return output


def _report_id(status: str, record_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha1("|".join(record_ids).encode("utf-8")).hexdigest()[:12]
    kind = "auto" if status == "auto_merged" else "review"
    return f"duplicate_{kind}_{digest}"


def _report(
    status: str,
    record_ids: tuple[str, ...],
    records_by_id: dict[str, CompanyRecord],
    evidence: Iterable[str],
    conflicts: Iterable[str],
) -> DuplicateNameReport:
    names = tuple(sorted(
        {records_by_id[record_id].name for record_id in record_ids},
        key=lambda value: (value.lower(), value),
    ))
    document_ids = tuple(sorted({
        document_id
        for record_id in record_ids
        if (document_id := records_by_id[record_id].document_id)
    }))
    return DuplicateNameReport(
        report_id=_report_id(status, record_ids),
        status=status,
        user_review_required=status == "review_required",
        names=names,
        record_ids=record_ids,
        document_ids=document_ids,
        evidence=tuple(sorted(set(evidence))),
        conflicts=tuple(sorted(set(conflicts))),
    )


def _duplicate_name_reports(
    records_by_id: dict[str, CompanyRecord],
    clusters: Iterable[CompanyCluster],
    review_candidates: Iterable[CompanyCandidate],
) -> tuple[DuplicateNameReport, ...]:
    clusters = tuple(clusters)
    review_candidates = tuple(review_candidates)
    reports = [
        _report(
            "auto_merged",
            cluster.member_record_ids,
            records_by_id,
            cluster.evidence_notes,
            (),
        )
        for cluster in clusters
    ]

    review_ids = {
        record_id
        for candidate in review_candidates
        for record_id in (candidate.left_record_id, candidate.right_record_id)
    }
    review_groups = _UnionFind(records_by_id)
    for candidate in review_candidates:
        review_groups.union(candidate.left_record_id, candidate.right_record_id)
    # Include confirmed aliases in any review component they touch without
    # turning the review candidate itself into an automatic cluster member.
    for cluster in clusters:
        if review_ids.intersection(cluster.member_record_ids):
            first, *rest = cluster.member_record_ids
            for record_id in rest:
                review_groups.union(first, record_id)

    grouped: dict[str, set[str]] = defaultdict(set)
    for record_id in review_ids:
        grouped[review_groups.find(record_id)].add(record_id)
    for cluster in clusters:
        roots = {review_groups.find(record_id) for record_id in cluster.member_record_ids}
        for root in roots.intersection(grouped):
            grouped[root].update(cluster.member_record_ids)

    for record_ids in grouped.values():
        ordered_ids = tuple(sorted(record_ids))
        candidates = [
            candidate
            for candidate in review_candidates
            if candidate.left_record_id in record_ids
            and candidate.right_record_id in record_ids
        ]
        cluster_evidence = [
            note
            for cluster in clusters
            if set(cluster.member_record_ids).issubset(record_ids)
            for note in cluster.evidence_notes
        ]
        reports.append(_report(
            "review_required",
            ordered_ids,
            records_by_id,
            [*cluster_evidence, *(note for candidate in candidates for note in candidate.evidence)],
            (conflict for candidate in candidates for conflict in candidate.conflicts),
        ))

    return tuple(sorted(reports, key=lambda report: (report.status, report.record_ids, report.report_id)))


def resolve_companies(records: Iterable[CompanyRecord], config: ResolverConfig | None = None) -> ResolutionResult:
    config = config or ResolverConfig()
    sorted_records = sorted(records, key=lambda record: record.record_id)
    if len({record.record_id for record in sorted_records}) != len(sorted_records):
        raise ValueError("record_id values must be unique")
    records_by_id = {record.record_id: record for record in sorted_records}
    weights = token_weights(sorted_records)
    candidate_pairs = generate_candidate_pairs(sorted_records)
    candidates: list[CompanyCandidate] = []
    for left_id, right_id in candidate_pairs:
        candidates.append(score_candidate(records_by_id[left_id], records_by_id[right_id], config, weights))

    proposed_auto = sorted((candidate for candidate in candidates if candidate.tier == "auto_merge"), key=lambda c: (c.left_record_id, c.right_record_id))
    review_candidates = [candidate for candidate in candidates if candidate.tier == "review"]
    rejected_candidates = [candidate for candidate in candidates if candidate.tier == "reject"]
    applied_auto: list[CompanyCandidate] = []

    union_find = _UnionFind(records_by_id)
    for candidate in proposed_auto:
        left_root = union_find.find(candidate.left_record_id)
        right_root = union_find.find(candidate.right_record_id)
        left_regs = {
            reg
            for record_id, record in records_by_id.items()
            if union_find.find(record_id) == left_root and (reg := record_registration(record))
        }
        right_regs = {
            reg
            for record_id, record in records_by_id.items()
            if union_find.find(record_id) == right_root and (reg := record_registration(record))
        }
        if left_regs and right_regs and left_regs.isdisjoint(right_regs):
            review_candidates.append(
                replace(
                    candidate,
                    tier="review",
                    conflicts=tuple(dict.fromkeys((*candidate.conflicts, "cluster_registration_conflict"))),
                )
            )
            continue
        union_find.union(candidate.left_record_id, candidate.right_record_id)
        applied_auto.append(candidate)

    auto_matches = tuple(applied_auto)
    review_matches = tuple(sorted(review_candidates, key=lambda c: (c.left_record_id, c.right_record_id)))
    rejected_matches = tuple(sorted(rejected_candidates, key=lambda c: (c.left_record_id, c.right_record_id)))

    grouped: dict[str, list[CompanyRecord]] = defaultdict(list)
    for record in sorted_records:
        grouped[union_find.find(record.record_id)].append(record)

    clusters: list[CompanyCluster] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        member_ids = tuple(sorted(record.record_id for record in members))
        aliases = tuple(sorted({record.name for record in members}, key=lambda value: (value.lower(), value)))
        member_pairs = [
            candidate
            for candidate in auto_matches
            if candidate.left_record_id in member_ids and candidate.right_record_id in member_ids
        ]
        # Review candidates are intentionally excluded: only automatic pairs
        # are allowed to create a cluster without human confirmation.
        notes = tuple(sorted({note for candidate in member_pairs for note in candidate.evidence}))
        clusters.append(
            CompanyCluster(
                cluster_id=_cluster_id(member_ids),
                canonical_name=_canonical_name(members),
                aliases=aliases,
                member_record_ids=member_ids,
                supporting_metadata=_supporting_metadata(members),
                evidence_notes=notes,
            )
        )

    clusters.sort(key=lambda cluster: (cluster.member_record_ids, cluster.cluster_id))
    duplicate_name_reports = _duplicate_name_reports(
        records_by_id,
        clusters,
        review_matches,
    )
    return ResolutionResult(
        clusters=tuple(clusters),
        auto_matches=auto_matches,
        duplicate_name_reports=duplicate_name_reports,
        review_candidates=review_matches,
        rejected_candidates=rejected_matches,
    )
