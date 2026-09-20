"""Run a four-call entity-overview checkpoint on a finite Harvey LAB fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import ingest_file
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    NameCatalogue,
    build_name_discovery_prompt,
    normalize_name,
    overview_inventory,
    parse_discovered_names,
    record_discovered_names,
    prepare_overview_tasks,
)
from src.swarm.models import Entry, EntrySource
from src.swarm.orchestrator import run_orchestrator
from src.swarm.worker_dispatch import call_model, execute_workers_parallel
from src.providers.gemini import GeminiCaller


class RecordingCaller:
    """Save the exact checkpoint inputs, outputs, and usage without logging credentials."""

    def __init__(self, caller):
        self.caller = caller
        self.calls: list[dict] = []

    def complete(self, prompt, *, max_tokens=8192, temperature=0.05, json_mode=True):
        result = self.caller.complete(
            prompt, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
        )
        self.calls.append({
            "prompt": prompt,
            "response": result.text,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "tokens_total": result.tokens_total,
            "model": result.model,
            "provider_attempts": result.provider_attempts,
        })
        return result


def _fixture_cards(task_dir: Path) -> list[Entry]:
    """Use the configured sanctions documents, retaining only a finite Crestmoor fixture."""
    selected = [
        ("commercial-invoice-draft.docx", "Crestmoor Trading"),
        ("kyc-crestmoor-trading.docx", "Crestmoor Trading"),
        ("sentinel-screening-results.docx", "Crestmoor Trading"),
        ("kyc-zenith-petrochemical.docx", "Crestmoor Trading"),
    ]
    cards = []
    for index, (filename, needle) in enumerate(selected, start=1):
        document = ingest_file(task_dir / "documents" / filename)
        paragraph = next(
            (line.strip() for line in document.text.splitlines() if needle.casefold() in line.casefold()),
            "",
        )
        if not paragraph:
            raise ValueError(f"fixture name not found in {filename}")
        cards.append(Entry(
            id=f"fixture_{index}", content=paragraph,
            source=EntrySource(document=filename, section="fixture excerpt", evidence=paragraph),
            confidence=0.9,
        ))
    return cards


def _snapshot_cards(snapshot: Path, entry_ids: list[str]) -> list[Entry]:
    """Reuse a finite set of original cards without replaying a full swarm run."""
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    by_id = {entry.get("id"): entry for entry in saved.get("entries", [])}
    cards = []
    for entry_id in entry_ids:
        entry = by_id.get(entry_id)
        if not entry:
            raise ValueError(f"snapshot does not contain entry {entry_id}")
        source = entry.get("source")
        cards.append(Entry(
            id=entry_id, content=str(entry.get("content", "")),
            type=str(entry.get("type", "observation")), status=str(entry.get("status", "active")),
            source=EntrySource(**source) if isinstance(source, dict) else None,
            confidence=float(entry.get("confidence", 0.5)),
        ))
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--entity", default="crestmoor trading ag")
    parser.add_argument("--blackboard", type=Path,
                        help="Saved blackboard for a two-call finite replay.")
    parser.add_argument("--entry-ids",
                        help="Comma-separated original entry IDs to replay with --blackboard.")
    parser.add_argument("--variant", action="append", default=[],
                        help="Candidate spelling for the replay retrieval group; repeat as needed.")
    parser.add_argument("--refresh-artifact", type=Path,
                        help="Prior live artifact; makes one refresh call and skips downstream work.")
    args = parser.parse_args()

    if bool(args.blackboard) != bool(args.entry_ids):
        parser.error("--blackboard and --entry-ids must be used together")
    if not args.blackboard and not args.task_dir:
        parser.error("--task-dir is required unless replaying --blackboard entries")
    if args.refresh_artifact and not args.blackboard:
        parser.error("--refresh-artifact requires --blackboard replay")

    os.environ["GEMINI_API_KEY"] = args.api_key_file.read_text(encoding="utf-8").strip()
    os.environ["GEMINI_REQUESTS_PER_MINUTE"] = "10"
    args.output.mkdir(parents=True, exist_ok=True)

    replay = bool(args.blackboard)
    refresh_only = bool(args.refresh_artifact)
    cards = (_snapshot_cards(args.blackboard, args.entry_ids.split(",")) if replay
             else _fixture_cards(args.task_dir))
    caller = RecordingCaller(GeminiCaller(model="gemini-3.1-flash-lite"))

    report = {
        "model": "gemini-3.1-flash-lite",
        "requests_per_minute": 10,
        "fixture_card_ids": [entry.id for entry in cards], "replay": replay, "refresh_only": refresh_only,
    }
    try:
        if replay:
            entity_id = normalize_name(args.entity)
            variants = args.variant or [args.entity]
            state = {"variants": {entity_id: variants}}
            if refresh_only:
                prior = json.loads(args.refresh_artifact.read_text(encoding="utf-8"))
                state["overviews"] = {entity_id: {
                    "entry_id": "prior-live-overview",
                    "input_ids": prior.get("matched_card_ids", []),
                    "structured": prior.get("structured_overview", {}),
                }}
            names = variants
        else:
            names_payload, name_tokens = call_model(caller, build_name_discovery_prompt(cards))
            names = parse_discovered_names(names_payload)
            state = {"discovered_card_ids": [entry.id for entry in cards]}
            record_discovered_names(state, names_payload, cards)
            catalogue = NameCatalogue({key: set(value) for key, value in state["variants"].items()})
            catalogue.add([args.entity])
            entity_id = catalogue.key_for(args.entity)
            state["variants"] = {key: sorted(value) for key, value in catalogue.variants.items()}
        board = Blackboard(
            task_instruction="Assess the sanctions compliance entity record.",
            entries=cards, entity_overview_state=state,
        )
        if replay:
            inventory, orchestration = [], {"replay": True}
            overview_tasks = [{
                "description": "Build the structured entity overview from these candidate cards.",
                "expected_output_type": "entity_overview", "entity_overview_id": entity_id,
                "entity_variants": variants, "reads_from_blackboard": [],
                "reads_from_documents": [], "priority": "high",
                "entity_refresh": refresh_only,
            }]
        else:
            inventory = overview_inventory(state, cards)
            orchestration, orchestration_tokens = run_orchestrator(
                board, caller, entity_overviews=inventory,
                override=(
                    "Return exactly one entity_overview worker for retrieval ID "
                    f"'{entity_id}', with no other workers."
                ),
            )
            overview_tasks = [
                task for task in prepare_overview_tasks(
                    orchestration.get("workers", []), inventory, state, iteration=1,
                )
                if task.get("expected_output_type") == "entity_overview"
            ]
        if len(overview_tasks) != 1:
            raise ValueError("orchestrator did not schedule exactly one usable entity overview")
        overview_output = execute_workers_parallel(overview_tasks, board, caller)[0]
        overview_entries = overview_output.entries
        matched_ids = overview_output.task["entity_input_ids"]
        overview_tokens = overview_output.tokens_used
        board.add_entries_batch(overview_entries)
        overview = next(entry for entry in overview_entries if entry.type == "entity_overview")
        worker_output = [] if refresh_only else execute_workers_parallel([{
            "description": "Use the entity overview to identify supported sanctions-compliance follow-up work.",
            "reads_from_blackboard": [overview.id], "reads_from_documents": [],
            "expected_output_type": "analysis", "priority": "high",
        }], board, caller)
        report.update({
            "discovered_names": names, "entity_id": entity_id,
            "inventory": inventory, "orchestration": orchestration,
            "matched_card_ids": matched_ids,
            "overview_entries": [entry.to_dict() for entry in overview_entries],
            "structured_overview": overview_output.task.get("entity_structured_overview", {}),
            "downstream_entries": [entry.to_dict() for entry in worker_output[0].entries] if worker_output else [],
            "tokens": {"overview": overview_tokens, "downstream": worker_output[0].tokens_used if worker_output else 0},
        })
        if not replay:
            report["tokens"].update({"name_discovery": name_tokens, "orchestration": orchestration_tokens})
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["calls"] = caller.calls
        (args.output / "entity_overview_live.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )

    print(json.dumps({
        "saved": str(args.output / "entity_overview_live.json"),
        "calls": len(caller.calls), "matched_card_ids": matched_ids,
        "overview_chars": len(overview.content),
    }))


if __name__ == "__main__":
    main()
