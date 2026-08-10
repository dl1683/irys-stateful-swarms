from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .models import Entry


@dataclass(frozen=True)
class EntityMaintenanceConfig:
    entity_resolution_interval_iterations: int = 3
    entity_profile_min_card_count: int = 2
    duplicate_review_threshold: float = 0.85
    entity_repair_max_mentions_per_card: int = 20

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> EntityMaintenanceConfig:
        return cls(
            entity_resolution_interval_iterations=metadata.get(
                "entity_resolution_interval_iterations", 3
            ),
            entity_profile_min_card_count=metadata.get("entity_profile_min_card_count", 2),
            duplicate_review_threshold=metadata.get("duplicate_review_threshold", 0.85),
            entity_repair_max_mentions_per_card=metadata.get(
                "entity_repair_max_mentions_per_card", 20
            ),
        )


@dataclass
class EntityMaintenanceState:
    schema_version: int = 1
    processed_card_fingerprints: dict[str, str] = field(default_factory=dict)
    profiles: dict[str, dict[str, object]] = field(default_factory=dict)
    candidates: dict[str, dict[str, object]] = field(default_factory=dict)
    decisions: dict[str, dict[str, object]] = field(default_factory=dict)
    runs: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def load(cls, output_dir: str) -> EntityMaintenanceState:
        path = _state_path(output_dir)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Entity maintenance state must be a JSON object")
        return cls(
            schema_version=data.get("schema_version", 1),
            processed_card_fingerprints=data.get("processed_card_fingerprints", {}),
            profiles=data.get("profiles", {}),
            candidates=data.get("candidates", {}),
            decisions=data.get("decisions", {}),
            runs=data.get("runs", []),
        )

    def write(self, output_dir: str) -> Path:
        path = _state_path(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "processed_card_fingerprints": self.processed_card_fingerprints,
                "profiles": self.profiles,
                "candidates": self.candidates,
                "decisions": self.decisions,
                "runs": self.runs,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(path)
        return path

    def card_is_dirty(self, entry: Entry) -> bool:
        return self.processed_card_fingerprints.get(entry.id) != self.fingerprint(entry)

    def mark_cards_processed(self, entries: Iterable[Entry]) -> None:
        for entry in entries:
            self.processed_card_fingerprints[entry.id] = self.fingerprint(entry)

    def fingerprint(self, payload: object) -> str:
        if isinstance(payload, Entry):
            payload = {
                "id": payload.id,
                "content": payload.content,
                "source": (
                    {
                        "document": payload.source.document,
                        "section": payload.source.section,
                        "evidence": payload.source.evidence,
                    }
                    if payload.source else None
                ),
                "status": payload.status,
                "direct_document_context": payload.direct_document_context,
                "entities": payload.entities,
                "entity_annotation_provenance": payload.entity_annotation_provenance,
            }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _state_path(output_dir: str) -> Path:
    return Path(output_dir) / "swarm" / "entity_resolution" / "state.json"
