from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from .blackboard import Blackboard
from .entity_candidate_scoring import refresh_candidates
from .entity_maintenance_store import EntityMaintenanceConfig, EntityMaintenanceState
from .entity_mention_repair import RepairSummary, build_entity_catalogue, eligible_direct_entries, repair_entity_mentions
from .entity_profiles import refresh_entity_profiles
from .entity_specialist_review import CONFIRMED_OUTCOMES, review_pending_candidates
from .models import Entry, ModelCaller, WorkerRecord


@dataclass(frozen=True)
class EntityMaintenanceRunSummary:
    trigger: Literal["periodic", "final"]
    processed_card_ids: tuple[str, ...] = ()
    repaired_mentions: int = 0
    dirty_profile_ids: tuple[str, ...] = ()
    review_candidate_ids: tuple[str, ...] = ()
    specialist_calls: int = 0
    projected_decision_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def maintenance_is_due(
    iteration: int,
    config: EntityMaintenanceConfig,
    *,
    final: bool = False,
) -> bool:
    interval = config.entity_resolution_interval_iterations
    return final or (interval > 0 and iteration > 0 and iteration % interval == 0)


def run_entity_maintenance(
    blackboard: Blackboard,
    caller: ModelCaller,
    config: EntityMaintenanceConfig,
    *,
    trigger: Literal["periodic", "final"],
) -> EntityMaintenanceRunSummary:
    state = blackboard.entity_maintenance_state
    dirty_entries = [
        entry for entry in eligible_direct_entries(blackboard.entries) if state.card_is_dirty(entry)
    ]
    all_direct_entries = eligible_direct_entries(blackboard.entries)
    if dirty_entries:
        catalogue = build_entity_catalogue(all_direct_entries, state.profiles)
        repair = repair_entity_mentions(
            dirty_entries,
            catalogue,
            config.entity_repair_max_mentions_per_card,
            f"entity-maintenance-{blackboard.iteration}-{trigger}",
        )
    else:
        repair = RepairSummary()
    profiles = refresh_entity_profiles(
        state, all_direct_entries, config.entity_profile_min_card_count,
    )
    _retire_profile_dependencies(blackboard, state, profiles.retired_profile_ids)
    candidates = refresh_candidates(state, set(profiles.dirty_profile_ids), config)
    if dirty_entries:
        state.mark_cards_processed(dirty_entries)
    review_ids = tuple(sorted(set(candidates.review_candidate_ids) | {
        candidate_id
        for candidate_id, candidate in state.candidates.items()
        if isinstance(candidate, Mapping) and candidate.get("status") == "review_retry"
    }))
    reviews = review_pending_candidates(state, review_ids, caller)
    projected = project_confirmed_decisions(blackboard, state)
    summary = EntityMaintenanceRunSummary(
        trigger=trigger,
        processed_card_ids=tuple(entry.id for entry in dirty_entries),
        repaired_mentions=repair.repaired_mentions,
        dirty_profile_ids=profiles.dirty_profile_ids,
        review_candidate_ids=review_ids,
        specialist_calls=reviews.model_calls,
        projected_decision_ids=projected,
    )
    state.runs.append(summary.to_dict())
    if blackboard.output_dir:
        state.write(blackboard.output_dir)
    return summary


def project_confirmed_decisions(
    blackboard: Blackboard,
    state: EntityMaintenanceState,
) -> tuple[str, ...]:
    """Expose only confirmed sidecar decisions as stable resolution cards."""
    projected: list[str] = []
    for decision_id, decision in sorted(state.decisions.items()):
        if not isinstance(decision, Mapping) or decision.get("outcome") not in CONFIRMED_OUTCOMES:
            continue
        semantic_key = decision.get("semantic_key")
        fingerprint = decision.get("evidence_fingerprint")
        if not isinstance(semantic_key, str) or not semantic_key or not isinstance(fingerprint, str):
            continue
        card_id = f"duplicate-resolution-{semantic_key}"
        payload = {
            "outcome": decision["outcome"],
            "profile_ids": _strings(decision.get("profile_ids")),
            "decision_id": str(decision.get("decision_id", decision_id)),
            "evidence_fingerprint": fingerprint,
            "source_card_ids": _strings(decision.get("source_card_ids")),
            "rationale": str(decision.get("rationale", "")),
            "conflicts": _strings(decision.get("conflicts")),
        }
        existing = blackboard.find_entry(card_id)
        if existing is not None and _fingerprint(existing) == fingerprint:
            continue
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if existing is None:
            blackboard.add_entry(_resolution_entry(blackboard, card_id, content))
        else:
            existing.type = "duplicate_name_resolution"
            existing.content = content
            existing.source = None
            existing.created_by = WorkerRecord(
                "entity_maintenance", "duplicate_name_resolution", blackboard.iteration,
            )
            existing.confidence = 1.0
            existing.status = "active"
        projected.append(decision_id)
    return tuple(projected)


def _resolution_entry(blackboard: Blackboard, card_id: str, content: str) -> Entry:
    return Entry(
        id=card_id,
        type="duplicate_name_resolution",
        content=content,
        created_by=WorkerRecord(
            "entity_maintenance", "duplicate_name_resolution", blackboard.iteration,
        ),
        confidence=1.0,
        source=None,
    )


def _fingerprint(entry: Entry) -> str | None:
    try:
        payload = json.loads(entry.content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload.get("evidence_fingerprint") if isinstance(payload, dict) else None


def _strings(value: object) -> list[str]:
    return sorted({item for item in value if isinstance(item, str)}) if isinstance(value, list) else []


def _retire_profile_dependencies(
    blackboard: Blackboard,
    state: EntityMaintenanceState,
    retired_profile_ids: tuple[str, ...],
) -> None:
    retired = set(retired_profile_ids)
    if not retired:
        return
    for records in (state.candidates, state.decisions):
        for record_id, record in list(records.items()):
            if isinstance(record, Mapping) and retired.intersection(_strings(record.get("profile_ids"))):
                del records[record_id]
    for entry in blackboard.entries:
        if entry.type != "duplicate_name_resolution":
            continue
        try:
            payload = json.loads(entry.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and retired.intersection(_strings(payload.get("profile_ids"))):
            entry.status = "superseded"
