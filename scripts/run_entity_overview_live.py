"""Run a four-call entity-overview checkpoint on a finite Harvey LAB fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import ingest_file
from src.swarm import _run_due_deltas
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    NameCatalogue,
    automatic_overview_tasks,
    build_name_discovery_prompt,
    normalize_name,
    overview_inventory,
    parse_discovered_names,
    record_discovered_names,
    prepare_overview_tasks,
)
from src.swarm.models import Entry, EntrySource, WorkerRecord
from src.swarm.orchestrator import run_orchestrator
from src.swarm.synthesis_packet import build_synthesis_packet, project_overview_statements
from src.swarm.worker_dispatch import call_model, execute_workers_parallel


class RecordingCaller:
    """Save the exact checkpoint inputs, outputs, and usage without logging credentials."""

    def __init__(self, caller, max_calls: int = 4):
        self.caller = caller
        self.max_calls = max_calls
        self.logical_calls = 0
        self.calls: list[dict] = []

    def complete(self, prompt, *, max_tokens=8192, temperature=0.05, json_mode=True):
        self.logical_calls += 1
        if self.logical_calls > self.max_calls:
            raise RuntimeError(f"logical model-call cap reached ({self.max_calls} calls)")
        try:
            result = self.caller.complete(
                prompt, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
            )
        except Exception as exc:
            self.calls.append({
                "prompt": prompt, "error": str(exc),
                "requested_output_tokens": max_tokens,
                "provider_requests_total": getattr(self.caller, "provider_request_count", None),
            })
            raise
        self.calls.append({
            "prompt": prompt,
            "response": result.text,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "tokens_total": result.tokens_total,
            "model": result.model,
            "provider_attempts": result.provider_attempts,
            "provider_requests_total": getattr(self.caller, "provider_request_count", None),
        })
        return result


def _separate_screening_rows(packet: list[dict]) -> list[dict]:
    """Accept candidate and sanctions-list profile labels as distinct people."""
    return [row for row in packet if any(term in
            str(row.get("overview_group_label", "")).casefold()
            for term in ("candidate", "sanctioned entity"))]


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


def _saved_entry(raw: dict) -> Entry:
    source = raw.get("source")
    worker = raw.get("created_by") or {}
    return Entry(
        id=str(raw.get("id", "")), type=str(raw.get("type", "observation")),
        content=str(raw.get("content", "")),
        source=EntrySource(**source) if isinstance(source, dict) else None,
        created_by=WorkerRecord(
            str(worker.get("worker_id", "")), str(worker.get("description", "")),
            int(worker.get("iteration", 0)),
        ),
        confidence=float(raw.get("confidence", 0.5)), status=str(raw.get("status", "active")),
        supports_entries=list(raw.get("supports", [])),
        supersedes_entries=list(raw.get("supersedes", [])),
    )


def _freshness_fixture(snapshot: Path) -> tuple[list[Entry], dict, int, list[dict], list[dict]]:
    """Load the saved run, show global fairness, then bound live work to two people."""
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    entries = [_saved_entry(raw) for raw in saved.get("entries", [])]
    state = json.loads(json.dumps(saved.get("entity_overview_state", {})))
    state["pending"] = []
    state["failed"] = []
    state["initial_queue"] = []
    # Historical snapshots predate direct-read bookkeeping. These inspected cards
    # carry quoted document evidence; keep the explicit list visible in the artifact.
    direct_ids = [
        entry.id for entry in entries
        if entry.type == "observation" and entry.source and entry.source.document
    ] + ["e392"]
    state["direct_source_entry_ids"] = list(dict.fromkeys(direct_ids))
    iteration = int(saved.get("iteration", 0))

    full_inventory = overview_inventory(state, entries, iteration)
    selection_state = json.loads(json.dumps(state))
    full_selection = automatic_overview_tasks(full_inventory, selection_state, iteration)

    personal_ids = ["nikolai v petrov", "dmitri k volkov"]
    missing = [entity_id for entity_id in personal_ids if entity_id not in state.get("overviews", {})]
    if missing:
        raise ValueError(f"snapshot is missing personal overview state: {', '.join(missing)}")
    compact_state = {
        "variants": {entity_id: state["variants"][entity_id] for entity_id in personal_ids},
        "overviews": {entity_id: state["overviews"][entity_id] for entity_id in personal_ids},
        "direct_source_entry_ids": state["direct_source_entry_ids"],
        "pending": [], "failed": [], "initial_queue": [],
    }
    relevant_ids = set()
    full_by_id = {item["entity_id"]: item for item in full_inventory}
    for entity_id in personal_ids:
        relevant_ids.update(compact_state["overviews"][entity_id].get("input_ids", []))
        relevant_ids.update(full_by_id[entity_id].get("source_card_ids", []))
    compact_entries = [
        entry for entry in entries
        if entry.id in relevant_ids and entry.status == "active" and entry.type != "entity_overview"
    ]
    compact_inventory = overview_inventory(compact_state, compact_entries, iteration)
    compact_tasks = automatic_overview_tasks(compact_inventory, compact_state, iteration)
    if {task["entity_overview_id"] for task in compact_tasks} != set(personal_ids):
        raise ValueError("bounded personal fixture did not select both refreshes")
    return compact_entries, compact_state, iteration, full_selection, compact_tasks


def _run_freshness_checkpoint(args, caller: RecordingCaller | None) -> dict:
    cards, state, iteration, full_selection, overview_tasks = _freshness_fixture(args.blackboard)
    report = {
        "model": args.model, "requests_per_minute": 10,
        "snapshot": str(args.blackboard), "historical_snapshot_immutable": True,
        "iteration": iteration, "fixture_card_ids": [entry.id for entry in cards],
        "historical_direct_evidence_override": ["e392"],
        "full_selection": full_selection, "bounded_personal_selection": overview_tasks,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        report["calls"] = []
        return report

    board = Blackboard(
        task_instruction="Extract current personal residency and sanctions-screening distinctions.",
        entries=cards, iteration=iteration, entity_overview_state=state,
    )
    # Sequential calls make provider failures attributable and keep the hard cap auditable.
    outputs = [execute_workers_parallel([task], board, caller)[0] for task in overview_tasks]
    failures = [
        {"entity_id": output.task.get("entity_overview_id"),
         "error": output.task.get("entity_error", "no usable overview")}
        for output in outputs if not output.entries
    ]
    if failures:
        report.update({
            "error": "one or more bounded personal overview refreshes failed",
            "refresh_failures": failures, "calls": caller.calls,
            "provider_requests_total": getattr(caller.caller, "provider_request_count", None),
        })
        return report
    for output in outputs:
        board.add_entries_batch(output.entries)
        entity_id = output.task["entity_overview_id"]
        prior = board.entity_overview_state["overviews"][entity_id]
        board.entity_overview_state["overviews"][entity_id] = {
            **prior, "entry_id": next(e.id for e in output.entries if e.type == "entity_overview"),
            "input_ids": output.task["entity_input_ids"],
            "structured": output.task["entity_structured_overview"],
            "last_success_iteration": iteration,
        }
    overview_ids = [
        board.entity_overview_state["overviews"][entity_id]["entry_id"]
        for entity_id in ("nikolai v petrov", "dmitri k volkov")
    ]
    [worker] = execute_workers_parallel([{
        "description": "State each person's supported residency and keep screening candidates distinct.",
        "reads_from_blackboard": overview_ids, "reads_from_documents": [],
        "expected_output_type": "analysis", "priority": "high",
    }], board, caller)
    board.add_entries_batch(worker.entries)
    packet = build_synthesis_packet(project_overview_statements(board), board)
    packet_text = "\n".join(
        f"- [{row.get('overview_group_label')}] {row['summary']} (supports: {', '.join(row['entry_ids'])})"
        for row in packet
    )
    draft_payload, draft_tokens = call_model(caller, f"""Write a concise evidence-grounded paragraph.
Keep Nikolai V. Petrov and Dmitri K. Volkov separate from their screening candidates.
Retain each subject label and state supported residency facts. Do not infer missing facts.

LABELLED PACKET ROWS:
{packet_text}""", json_mode=False)
    report.update({
        "overview_outputs": [e.to_dict() for output in outputs for e in output.entries],
        "downstream_entries": [e.to_dict() for e in worker.entries],
        "packet": packet, "draft": draft_payload.get("text", ""),
        "draft_tokens": draft_tokens, "calls": caller.calls,
        "provider_requests_total": getattr(caller.caller, "provider_request_count", None),
    })
    return report


def _run_final_freshness_checkpoint(args, caller: RecordingCaller | None) -> dict:
    """Replay one late source finding through the real final pass, below routine threshold."""
    saved = json.loads(args.blackboard.read_text(encoding="utf-8"))
    by_id = {raw["id"]: raw for raw in saved["entries"]}
    entity_id, late_id = "nikolai v petrov", "e392"
    saved_state = saved["entity_overview_state"]
    prior = saved_state["overviews"][entity_id]
    selected_ids = list(dict.fromkeys(prior["input_ids"] + [late_id, prior["entry_id"]]))
    entries = [_saved_entry(by_id[entry_id]) for entry_id in selected_ids]
    state = {
        "variants": {entity_id: saved_state["variants"][entity_id]},
        "overviews": {entity_id: json.loads(json.dumps(prior))},
        # This historical direct read predates the explicit direct-source ledger.
        "direct_source_entry_ids": [late_id], "pending": [], "failed": [],
    }
    iteration = int(saved["iteration"])
    before = overview_inventory(state, entries, iteration)
    if len(before) != 1 or before[0]["new_card_count"] != 1 or before[0]["suggestion"]:
        raise ValueError("late-source fixture is not a single below-threshold update")
    report = {
        "model": args.model, "snapshot": str(args.blackboard),
        "historical_snapshot_immutable": True, "entity_id": entity_id,
        "late_source_id": late_id, "fixture_card_ids": selected_ids,
        "inventory_before": before, "dry_run": args.dry_run, "calls": [],
    }
    if args.dry_run:
        return report
    board = Blackboard(
        task_instruction="Report current personal residency and screening distinctions.",
        entries=entries, iteration=iteration, entity_overview_state=state,
    )
    _run_due_deltas(board, caller, final=True)
    packet = build_synthesis_packet(project_overview_statements(board), board)
    new_id = board.entity_overview_state["overviews"][entity_id]["entry_id"]
    subject_rows = [row for row in packet if late_id in row["entry_ids"]
                    and "petrov" in str(row.get("overview_group_label", "")).casefold()]
    candidate_rows = _separate_screening_rows(packet)
    report.update({
        "inventory_after": overview_inventory(board.entity_overview_state, board.entries, iteration),
        "final_candidates": state.get("final_candidates", []),
        "final_attempted": state.get("final_attempted", []),
        "freshness_failures": state.get("freshness_failures", []),
        "prior_overview_id": prior["entry_id"], "current_overview_id": new_id,
        "overview_outputs": [entry.to_dict() for entry in board.entries
                             if entry.type == "entity_overview" and entry.id == new_id],
        "packet": packet, "subject_rows_from_late_source": subject_rows,
        "candidate_rows": candidate_rows, "calls": caller.calls,
        "provider_requests_total": getattr(caller.caller, "provider_request_count", None),
    })
    if new_id == prior["entry_id"] or not subject_rows or not candidate_rows:
        report["error"] = "final pass did not publish a labelled subject fact and separate candidate"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Gemini model ID for live calls and artifact provenance.")
    parser.add_argument("--entity", default="crestmoor trading ag")
    parser.add_argument("--blackboard", type=Path,
                        help="Saved blackboard for a two-call finite replay.")
    parser.add_argument("--entry-ids",
                        help="Comma-separated original entry IDs to replay with --blackboard.")
    parser.add_argument("--variant", action="append", default=[],
                        help="Candidate spelling for the replay retrieval group; repeat as needed.")
    parser.add_argument("--refresh-artifact", type=Path,
                        help="Prior live artifact; makes one refresh call and skips downstream work.")
    parser.add_argument("--freshness-checkpoint", action="store_true",
                        help="Use saved September evidence for the bounded four-call checkpoint.")
    parser.add_argument("--final-freshness-checkpoint", action="store_true",
                        help="Replay one late source fact through the final pass (one model call).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and save selections without making provider calls.")
    args = parser.parse_args()

    if args.freshness_checkpoint and args.final_freshness_checkpoint:
        parser.error("choose only one freshness checkpoint")
    if not (args.freshness_checkpoint or args.final_freshness_checkpoint) and bool(args.blackboard) != bool(args.entry_ids):
        parser.error("--blackboard and --entry-ids must be used together")
    if (args.freshness_checkpoint or args.final_freshness_checkpoint) and not args.blackboard:
        parser.error("freshness checkpoints require --blackboard")
    if not (args.freshness_checkpoint or args.final_freshness_checkpoint) and not args.blackboard and not args.task_dir:
        parser.error("--task-dir is required unless replaying --blackboard entries")
    if args.refresh_artifact and not args.blackboard:
        parser.error("--refresh-artifact requires --blackboard replay")

    if not args.dry_run:
        if not args.api_key_file:
            parser.error("--api-key-file is required for live calls")
        os.environ["GEMINI_API_KEY"] = args.api_key_file.read_text(encoding="utf-8").strip()
        os.environ["GEMINI_REQUESTS_PER_MINUTE"] = "10"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty attempt directory")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.freshness_checkpoint or args.final_freshness_checkpoint:
        os.environ["GEMINI_RETRY_DELAY_SECONDS"] = "20"
        from src.providers.gemini import GeminiCaller
        caller = None if args.dry_run else RecordingCaller(
            GeminiCaller(model=args.model), max_calls=1 if args.final_freshness_checkpoint else 4,
        )
        report = {}
        try:
            report = (_run_final_freshness_checkpoint(args, caller)
                      if args.final_freshness_checkpoint else _run_freshness_checkpoint(args, caller))
        except Exception as exc:
            report = {
                "model": args.model, "snapshot": str(args.blackboard),
                "dry_run": args.dry_run, "error": str(exc),
                "calls": caller.calls if caller else [],
            }
            raise
        finally:
            (args.output / "entity_overview_freshness.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
        if report.get("error"):
            raise RuntimeError(report["error"])
        print(json.dumps({
            "saved": str(args.output / "entity_overview_freshness.json"),
            "calls": len(report["calls"]),
            "selection": (report.get("final_candidates", [item["entity_id"] for item in
                         report.get("inventory_before", [])]) if args.final_freshness_checkpoint else [
                task["entity_overview_id"] for task in report["bounded_personal_selection"]
            ]),
        }))
        return

    replay = bool(args.blackboard)
    refresh_only = bool(args.refresh_artifact)
    cards = (_snapshot_cards(args.blackboard, args.entry_ids.split(",")) if replay
             else _fixture_cards(args.task_dir))
    from src.providers.gemini import GeminiCaller
    caller = RecordingCaller(GeminiCaller(model=args.model))

    report = {
        "model": args.model,
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
