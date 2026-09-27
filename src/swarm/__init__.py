from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .blackboard import Blackboard
from .analysis import run_direct_analysis
from .artifact_commitments import build_artifact_commitments
from .blackboard_maintenance import (
    blackboard_maintenance_enabled,
    run_blackboard_maintenance,
)
from .convergence import check_convergence, supervisor_review
from .seed import generate_seed, seed_to_signals
from .domain_lens import (
    generate_domain_lens, lens_to_entries, lens_to_signals, format_lens_guidance,
)
from .curation import curate_entries
from .debt_sensors import debt_sensors_enabled, run_debt_sensors
from .derived_work import (
    calculation_debt_enabled,
    run_calculation_debt_detection,
)
from .obligations import build_synthesis_obligations
from .models import (
    Document, DocumentStatus, Entry, EntrySource, EpistemicStatus, ModelCaller, Task, WorkerRecord,
    gen_entry_id,
)
from .orchestrator import run_orchestrator
from .section_index import build_section_index
from .signals import prioritize_signals
from .source_claims import (
    source_claim_verification_enabled,
    verify_source_claims,
)
from .source_custody import enforce_source_custody
from .state_conversion import (
    coverage_report_to_entries, run_plan_coverage_review,
    run_plan_coverage_state_repair, run_state_conversion_review,
)
from .artifact_contracts import build_artifact_contracts, contracts_to_signals
from .synthesis_packet import (
    build_synthesis_packet,
    consolidate_items,
    filter_evidence_entries,
    write_synthesis_packet_report,
)
from .synthesis import (
    shadow_judge_audit,
    shadow_judge_audit_enabled,
    synthesize_deliverable,
    synthesize_file_deliverables,
)
from .survival_trace import write_pending_survival_trace
from .worker_dispatch import (
    call_model, execute_workers_parallel, parse_worker_output,
    passes_quality_gate,
)
from .entity_overview import (
    NameCatalogue, apply_delta, attach_relevant_overviews, automatic_overview_tasks,
    build_delta_prompt, build_overview_prompt,
    discover_pending_names, entry_state,
    overview_inventory, prepare_overview_tasks, render_pointer_card,
    select_delta_tasks, sync_overview_references,
    select_identity_reviews, identity_cards, build_identity_review_prompt,
    _validated_identity_outcome, apply_identity_merge, validate_pointer_cards,
)


_STRUCTURED_SINGLE_FILE_EXTENSIONS = {
    ".xlsx", ".xls", ".csv", ".pptx", ".ppt",
}
_STRUCTURED_SINGLE_FILE_HINTS = (
    "matrix", "workbook", "tracker", "register", "deck", "presentation",
)
_DOCX_FILE_SCOPED_HINTS = ("redline", "redlined", "rider")
_DOCX_MARKUP_EXCLUSIONS = ("analysis", "memo", "report", "cover")


def _record_entity_discovery_failure(state: dict, iteration: int, error: Exception) -> None:
    """Leave retryable discovery failures visible in the next snapshot/report."""
    state.setdefault("discovery_failures", []).append({
        "iteration": iteration,
        "error_type": type(error).__name__,
        "error": str(error)[:300],
    })


def _prepare_entity_work(blackboard: Blackboard, workers: list[dict], iteration: int,
                         inventory: list[dict]) -> list[dict]:
    """Attach current cards without holding up ordinary workers."""
    state = blackboard.entity_overview_state
    requested = prepare_overview_tasks(workers, inventory, state, iteration)
    return attach_relevant_overviews(requested, blackboard.entries, state)


def _run_initial_overviews(blackboard: Blackboard, caller: ModelCaller,
                           smart_caller: ModelCaller | None) -> None:
    """Run eligible creation after readers commit, outside their worker batch."""
    state = blackboard.entity_overview_state
    registered = {doc.name for doc in blackboard.documents}
    while True:
        inventory = overview_inventory(state, blackboard.entries, blackboard.iteration, registered)
        tasks = automatic_overview_tasks(inventory, state, blackboard.iteration)
        if not tasks:
            break
        remaining = blackboard.token_budget - blackboard.total_tokens_used
        eligible = []
        for task in tasks:
            group_id = task["entity_overview_id"]
            catalogue = NameCatalogue({group_id: set(task["entity_variants"])})
            matched = [(entry, reason) for entry, reason in catalogue.matched_entries(
                blackboard.entries, group_id) if entry.id in task["entity_source_ids"]]
            estimate = len(build_overview_prompt(group_id, task["entity_variants"], matched).encode("utf-8")) + 8192
            if estimate <= remaining:
                eligible.append(task)
                remaining -= estimate
            else:
                state.setdefault("budget_limited", []).append(group_id)
                for field in ("pending", "initial_queue"):
                    state[field] = [key for key in state.get(field, [])
                                    if key not in task["entity_overview_group_ids"]]
        if not eligible:
            break
        outputs = execute_workers_parallel(eligible, blackboard, caller, smart_caller=smart_caller)
        _commit_entity_worker_outputs(blackboard, outputs, caller, blackboard.iteration)


def _sync_entity_state(blackboard: Blackboard) -> None:
    sync_overview_references(blackboard.entity_overview_state, blackboard.entries,
                             {doc.name for doc in blackboard.documents}, blackboard.iteration)


def _run_due_deltas(blackboard: Blackboard, caller: ModelCaller, *, final: bool = False) -> None:
    """One bounded wave at each due point, including the final reading checkpoint."""
    state = blackboard.entity_overview_state
    if final and state.get("final_delta_done"):
        return
    _sync_entity_state(blackboard)
    inventory = overview_inventory(state, blackboard.entries, blackboard.iteration,
                                   {doc.name for doc in blackboard.documents})
    tasks = select_delta_tasks(state, inventory, blackboard.entries)
    if final:
        state["final_delta_done"] = True
    if not tasks:
        return
    remaining = blackboard.token_budget - blackboard.total_tokens_used
    eligible = []
    for task in tasks:
        record = state["overviews"][task["entity_group_id"]]
        card = next(card for card in record["cards"] if card["id"] == task["entity_card_id"])
        sources = blackboard.get_entries_by_ids(task["entity_delta_source_ids"])
        # UTF-8 bytes conservatively bound input tokens; delta output is capped at 2048.
        estimate = len(build_delta_prompt(card, sources, task["entity_hints"]).encode("utf-8")) + 2048
        if estimate <= remaining:
            eligible.append(task)
            remaining -= estimate
        else:
            state.setdefault("budget_limited", []).append(task["entity_card_id"])
            state.setdefault("delta_backlog", []).append(task["entity_card_id"])
    tasks = eligible
    if not tasks:
        return
    outputs = execute_workers_parallel(tasks, blackboard, caller)
    by_id = {entry.id: entry for entry in blackboard.entries if entry.status == "active"
             and entry.source and entry.source.document in {doc.name for doc in blackboard.documents}}
    for output in outputs:
        blackboard.add_tokens(output.tokens_used, output.tokens_input, output.tokens_output, output.model)
        usage = state.setdefault("usage", {})
        usage["delta_calls"] = usage.get("delta_calls", 0) + bool(output.tokens_used)
        usage["delta_tokens"] = usage.get("delta_tokens", 0) + output.tokens_used
        task = output.task
        record = state["overviews"][task["entity_group_id"]]
        index = next(i for i, card in enumerate(record["cards"])
                     if card["id"] == task["entity_card_id"])
        card = record["cards"][index]
        error = task.get("entity_error", "")
        if not error:
            try:
                revised = apply_delta(card, task["entity_delta_payload"],
                                      set(task["entity_delta_source_ids"]), by_id)
                revised["semantic_input_ids"] = inventory[
                    next(i for i, item in enumerate(inventory)
                         if item["entity_id"] == task["entity_group_id"])
                ]["source_card_ids"]
                card.clear()
                card.update(revised)
                entry = by_id.get(card["id"])
                if entry is None:
                    entry = next((e for e in blackboard.entries if e.id == card["id"]), None)
                if entry and revised["identity_source_ids"]:
                    entry.status = "active"
                    entry.supports_entries = revised["identity_source_ids"][:]
                    entry.content = render_pointer_card(revised)
                state["overview_review_requests"] = [r for r in state.get("overview_review_requests", [])
                                                      if r["overview_id"] != card["id"]]
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        if error:
            card["failed_delta_signature"] = task["entity_signature"]
            state.setdefault("delta_errors", []).append({
                "iteration": blackboard.iteration, "entity_id": card["id"],
                "job_kind": "delta", "reason": error[:300],
                "affected_card_ids": task["entity_delta_source_ids"],
            })
        state.setdefault("jobs", []).append({
            "iteration": blackboard.iteration, "kind": "delta", "entity_id": card["id"],
            "model": output.model, "input_tokens": output.tokens_input,
            "output_tokens": output.tokens_output, "provider_attempts": task.get("provider_attempts"),
            "failed": bool(error),
        })


def _run_split_rebuild(blackboard: Blackboard, card_id: str, caller: ModelCaller) -> tuple[int, int, int, str]:
    """Rebuild only a model-confirmed conflated card, preserving the old card."""
    from .entity_overview import build_overview_prompt
    from .worker_dispatch import get_last_call_usage
    state = blackboard.entity_overview_state
    card = identity_cards(state)[card_id]
    originals = blackboard.get_entries_by_ids(card["identity_source_ids"] + card["candidate_source_ids"])
    matched = [(entry, "confirmed conflation") for entry in originals if entry.status == "active"]
    prompt = build_overview_prompt(card["name"], [card["name"]], matched)
    if len(prompt.encode("utf-8")) + 8192 > blackboard.token_budget - blackboard.total_tokens_used:
        raise ValueError("split rebuild exceeds remaining token budget")
    payload, tokens = call_model(caller, prompt,
                                 max_tokens=8192)
    cards = validate_pointer_cards(payload.get("cards"), {entry.id for entry, _ in matched})
    if len(cards) < 2 or set.intersection(*(set(c["identity_source_ids"]) for c in cards)):
        raise ValueError("split rebuild needs separate source-backed identities")
    state.setdefault("identity_history", []).append({"kind": "split", "old_id": card_id,
                                                       "before": deepcopy(card)})
    replacements = []
    for rebuilt in cards:
        rebuilt["id"] = gen_entry_id()
        rebuilt["split_from"] = card_id
        replacements.append(rebuilt)
        blackboard.add_entries_batch([Entry(id=rebuilt["id"], type="entity_overview",
            content=render_pointer_card(rebuilt), supports_entries=rebuilt["identity_source_ids"][:],
            epistemic=EpistemicStatus("inference", "unknown", ""),
            created_by=WorkerRecord("identity_rebuild", "confirmed split", blackboard.iteration),
            status="active")])
    for record in state.get("overviews", {}).values():
        if any(c["id"] == card_id for c in record.get("cards", [])):
            record["cards"] = [c for c in record["cards"] if c["id"] != card_id] + replacements
            record["card_ids"] = [c["id"] for c in record["cards"]]
            record["entry_id"] = record["card_ids"][0]
    next(entry for entry in blackboard.entries if entry.id == card_id).status = "inactive"
    _, model, t_in, t_out = get_last_call_usage()
    return tokens, t_in, t_out, model


def _run_identity_reviews(blackboard: Blackboard, caller: ModelCaller,
                          smart_caller: ModelCaller | None) -> None:
    """Review at most three candidate pairs after all document reading."""
    state = blackboard.entity_overview_state
    _sync_entity_state(blackboard)
    proposals = select_identity_reviews(state, blackboard.entries)
    _sync_entity_state(blackboard)
    if not proposals:
        return
    remaining = blackboard.token_budget - blackboard.total_tokens_used
    selected = []
    for task in proposals:
        cards = identity_cards(state)
        originals = blackboard.get_entries_by_ids(task["source_ids"])
        estimate = len(build_identity_review_prompt(cards[task["pair"][0]], cards[task["pair"][1]],
                                                    originals).encode("utf-8")) + 2048
        if len(selected) < 3 and estimate <= remaining:
            selected.append(task)
            remaining -= estimate
    state["identity_review_backlog"] = [task["pair"] for task in proposals if task not in selected]
    if not selected:
        return
    outputs = execute_workers_parallel(selected, blackboard, caller, smart_caller=smart_caller)
    for output in outputs:
        task = output.task
        pair = task["pair"]
        key = "|".join(pair)
        blackboard.add_tokens(output.tokens_used, output.tokens_input, output.tokens_output, output.model)
        usage = state.setdefault("usage", {})
        usage["identity_review_calls"] = usage.get("identity_review_calls", 0) + bool(output.tokens_used)
        usage["identity_review_tokens"] = usage.get("identity_review_tokens", 0) + output.tokens_used
        cards = identity_cards(state)
        original = {e.id: e for e in blackboard.get_entries_by_ids(task["source_ids"])
                    if e.status == "active" and e.type != "entity_overview"}
        outcome = "unresolved"
        error = task.get("entity_error", "")
        refs = []
        if not error:
            try:
                payload = task["identity_payload"]
                outcome, refs = _validated_identity_outcome(payload, cards[pair[0]], cards[pair[1]], original)
                if outcome == "same":
                    target = apply_identity_merge(state, blackboard.entries, *pair, refs)
                    finding = Entry(id=gen_entry_id(), type="analysis",
                        content=(f"Identity review: {cards[pair[0]]['name']} ({pair[0]}) and "
                                 f"{cards[pair[1]]['name']} ({pair[1]}) identify the same entity; "
                                 f"use {target} for lookup."),
                        supports_entries=refs, epistemic=EpistemicStatus("inference", "unknown", ""),
                        created_by=WorkerRecord("identity_review", "source-grounded duplicate review",
                                                blackboard.iteration), status="active")
                    blackboard.add_entries_batch([finding])
                    state["identity_history"][-1]["finding_id"] = finding.id
                elif outcome == "distinct":
                    finding = Entry(id=gen_entry_id(), type="analysis",
                        content=(f"Identity review: {cards[pair[0]]['name']} ({pair[0]}) and "
                                 f"{cards[pair[1]]['name']} ({pair[1]}) identify distinct entities."),
                        supports_entries=refs, epistemic=EpistemicStatus("inference", "unknown", ""),
                        created_by=WorkerRecord("identity_review", "source-grounded distinction",
                                                blackboard.iteration), status="active")
                    blackboard.add_entries_batch([finding])
                    for card, other_id in ((cards[pair[0]], pair[1]), (cards[pair[1]], pair[0])):
                        pointers = card.setdefault("possibly_same_as", [])
                        pointers[:] = [p for p in pointers if p["overview_id"] != other_id]
                        pointers.append({"overview_id": other_id, "source_ids": refs,
                                         "status": "distinct"})
                elif outcome == "split":
                    split_id = payload["split_overview_id"]
                    from .worker_dispatch import get_last_call_usage, set_last_call_usage
                    set_last_call_usage(None)
                    rebuild_error = ""
                    try:
                        tokens, t_in, t_out, model = _run_split_rebuild(
                            blackboard, split_id, smart_caller or caller)
                    except (ValueError, RuntimeError) as exc:
                        _, model, t_in, t_out = get_last_call_usage()
                        tokens = t_in + t_out
                        rebuild_error = str(exc)
                    blackboard.add_tokens(tokens, t_in, t_out, model)
                    usage["identity_rebuild_calls"] = usage.get("identity_rebuild_calls", 0) + bool(tokens)
                    usage["identity_rebuild_tokens"] = usage.get("identity_rebuild_tokens", 0) + tokens
                    state.setdefault("jobs", []).append({"iteration": blackboard.iteration,
                        "kind": "identity_rebuild", "entity_id": split_id, "model": model,
                        "input_tokens": t_in, "output_tokens": t_out,
                        "failed": bool(rebuild_error), "error": rebuild_error[:300]})
                    if rebuild_error:
                        raise ValueError(rebuild_error)
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                error = str(exc)
        state.setdefault("identity_reviews", {})[key] = {
            "signature": task["signature"], "outcome": outcome if not error else "unresolved",
            "source_ids": refs, "error": error[:300]}
        if task.get("entity_caller_fallback"):
            state.setdefault("caller_fallbacks", []).append({"iteration": blackboard.iteration,
                "entity_id": key, "reason": "smart caller unavailable"})
        state.setdefault("jobs", []).append({"iteration": blackboard.iteration,
            "kind": "identity_review", "entity_id": key, "model": output.model,
            "input_tokens": output.tokens_input, "output_tokens": output.tokens_output,
            "provider_attempts": task.get("provider_attempts"), "failed": bool(error),
            "outcome": outcome if not error else "unresolved"})
    _sync_entity_state(blackboard)


def _after_document_round(blackboard: Blackboard, outputs: list,
                          caller: ModelCaller) -> None:
    """Initial reading and analysis-only rounds do not advance the delta clock."""
    registered = {doc.name for doc in blackboard.documents}
    if not any(doc_name in registered for output in outputs
               for doc_name, _ in output.sections_read):
        return
    state = blackboard.entity_overview_state
    state["document_rounds"] = state.get("document_rounds", 0) + 1
    if state["document_rounds"] % 3 == 0:
        _run_due_deltas(blackboard, caller)


def _commit_entity_worker_outputs(blackboard: Blackboard, outputs: list,
                                  caller: ModelCaller, iteration: int) -> int:
    """Publish validated overview state and provenance after a worker batch."""
    state = blackboard.entity_overview_state
    new_entries = []
    for output in outputs:
        blackboard.add_tokens(output.tokens_used, output.tokens_input, output.tokens_output, output.model)
        new_entries.extend(entry for entry in output.entries if passes_quality_gate(entry))
        for doc_name, sec_name in output.sections_read:
            for document in blackboard.documents:
                if document.name == doc_name:
                    document.mark_section_read(sec_name)
                    if " (part " in sec_name:
                        document.mark_section_read(sec_name.split(" (part ")[0])
    blackboard.add_entries_batch(new_entries)

    for output in outputs:
        if output.task.get("expected_output_type") != "entity_overview":
            continue
        usage = state.setdefault("usage", {})
        usage["initial_calls"] = usage.get("initial_calls", 0) + bool(output.tokens_used)
        usage["initial_tokens"] = usage.get("initial_tokens", 0) + output.tokens_used
        if output.task.get("entity_caller_fallback"):
            state.setdefault("caller_fallbacks", []).append({
                "iteration": iteration, "entity_id": output.task.get("entity_overview_id"),
                "reason": "smart caller unavailable",
            })
        group_ids = output.task.get("entity_overview_group_ids") or [output.task.get("entity_overview_id", "")]
        for field in ("pending", "initial_queue"):
            queue = state.setdefault(field, [])
            queue[:] = [key for key in queue if key not in group_ids]
        cards = [entry for entry in output.entries if entry in new_entries
                 and entry.type == "entity_overview" and entry.status == "active"]
        if not cards or len(cards) != len(output.task.get("entity_cards", [])):
            for group_id in group_ids:
                if group_id not in state.setdefault("failed", []):
                    state["failed"].append(group_id)
            state.setdefault("freshness_failures", []).append({
                "iteration": iteration, "group_ids": group_ids,
                "error": output.task.get("entity_error", "unusable overview output")[:300],
            })
            state.setdefault("jobs", []).append({
                "iteration": iteration, "kind": "initial", "entity_id": group_ids[0],
                "model": output.model, "input_tokens": output.tokens_input,
                "output_tokens": output.tokens_output,
                "provider_attempts": output.task.get("provider_attempts"), "failed": True,
            })
            continue
        for group_id in group_ids:
            state.setdefault("overviews", {})[group_id] = {
                "entry_id": cards[0].id,
                "card_ids": [card.id for card in cards],
                "input_ids": output.task.get("entity_input_ids", []),
                "cards": [dict(card, id=entry.id,
                               semantic_input_ids=output.task.get("entity_input_ids", [])) for card, entry in
                          zip(output.task["entity_cards"], cards)],
                "source_states": {
                    entry.id: entry_state(entry) for entry in blackboard.entries
                    if entry.id in output.task.get("entity_input_ids", [])
                    and entry.source and entry.source.document
                },
                "last_success_iteration": iteration,
            }
        state.setdefault("jobs", []).append({
            "iteration": iteration, "kind": "initial", "entity_id": cards[0].id,
            "model": output.model, "input_tokens": output.tokens_input,
            "output_tokens": output.tokens_output,
            "provider_attempts": output.task.get("provider_attempts"), "failed": False,
        })
        usage = state.setdefault("usage", {})
        usage["overview_tokens"] = usage.get("overview_tokens", 0) + output.tokens_used
        usage["overview_calls"] = usage.get("overview_calls", 0) + 1

    overview_ids = {card["id"] for record in state.get("overviews", {}).values()
                    for card in record.get("cards", [])}
    for output in outputs:
        consumed = [entry_id for entry_id in output.task.get("reads_from_blackboard", [])
                    if entry_id in overview_ids]
        if consumed:
            state.setdefault("recipients", []).append({
                "iteration": iteration, "worker_id": output.worker_id,
                "overview_ids": consumed,
                "referenced_overview_ids": [entry_id for entry_id in consumed
                                            if any(entry_id in entry.supports_entries
                                                   for entry in output.entries)],
            })
        attached = {item["overview_id"] for item in output.task.get("entity_overview_attachments", [])}
        for request in output.task.get("overview_review_requests", []):
            if not isinstance(request, dict) or request.get("overview_id") not in attached:
                continue
            reason = request.get("reason", "")
            if not isinstance(reason, str) or not reason.strip():
                continue
            kind = ("identity_review_requests" if any(word in reason.casefold()
                    for word in ("identity", "duplicate", "same person", "same entity"))
                    else "overview_review_requests")
            item = {"overview_id": request["overview_id"], "reason": reason.strip()[:250]}
            queue = state.setdefault(kind, [])
            if item not in queue:
                queue.append(item)

    direct_entries = [entry for output in outputs if output.sections_read
                      for entry in output.entries if entry in new_entries
                      and entry.type != "entity_overview" and entry.source and entry.source.document]
    if direct_entries:
        direct_ids = state.setdefault("direct_source_entry_ids", [])
        direct_ids[:] = list(dict.fromkeys(direct_ids + [entry.id for entry in direct_entries]))
        try:
            _, tokens = discover_pending_names(state, direct_entries, caller)
            blackboard.add_tokens_from_last_call(tokens)
            if tokens:
                usage = state.setdefault("usage", {})
                usage["name_discovery_tokens"] = usage.get("name_discovery_tokens", 0) + tokens
                usage["name_discovery_calls"] = usage.get("name_discovery_calls", 0) + 1
        except (ValueError, RuntimeError) as error:
            _record_entity_discovery_failure(state, iteration, error)
    _sync_entity_state(blackboard)
    return len(new_entries)


def _explicit_output_filenames(deliverables_map: dict) -> list[str]:
    filenames = []
    if not isinstance(deliverables_map, dict):
        return filenames
    for filename in deliverables_map.values():
        if isinstance(filename, str) and filename not in filenames:
            filenames.append(filename)
    return filenames


def _should_use_file_scoped_synthesis(deliverables_map: dict) -> bool:
    filenames = _explicit_output_filenames(deliverables_map)
    if len(filenames) > 1:
        return True
    if len(filenames) != 1:
        return False

    filename = filenames[0].lower()
    _, ext = os.path.splitext(filename)
    if ext in _STRUCTURED_SINGLE_FILE_EXTENSIONS:
        return True
    if ext == ".docx":
        stem = os.path.splitext(os.path.basename(filename))[0]
        if any(hint in stem for hint in _DOCX_FILE_SCOPED_HINTS):
            return True
        if "markup" in stem and not any(
            excluded in stem for excluded in _DOCX_MARKUP_EXCLUSIONS
        ):
            return True
        return False
    return any(hint in filename for hint in _STRUCTURED_SINGLE_FILE_HINTS)


def _restore_checkpoint(task: Task, path: Path) -> tuple[Blackboard, dict, list[int], bool]:
    board, seed_plan, counts, loop_ended, saved_sources = Blackboard.load_checkpoint(path)
    if board.task_instruction != task.instruction:
        raise ValueError("checkpoint task instruction differs from current task")
    # The checkpoint location anchors a relative output path across CLI working directories.
    if Path(path).resolve().parents[2] != Path(task.output_dir).resolve():
        raise ValueError("checkpoint output directory differs from current run")
    board.output_dir = task.output_dir
    fresh = _build_doc_statuses(task.documents)
    if Blackboard.source_fingerprints(fresh) != saved_sources:
        raise ValueError("checkpoint source files differ from current task")
    if len(board.documents) != len(fresh):
        raise ValueError("checkpoint document state is incomplete")
    for saved, current in zip(board.documents, fresh):
        if (saved.id, saved.name, saved.source_path) != (
                current.id, current.name, current.source_path):
            raise ValueError("checkpoint document identities differ from current task")
        # A document read before interruption must have text and index available again.
        if current._lazy_doc is not None and saved._checkpoint_was_loaded:
            current.materialize()
        saved.text = current.text
        saved.section_index = current.section_index
        saved._lazy_doc = current._lazy_doc
        if hasattr(current, "structured"):
            saved.structured = current.structured
    return board, seed_plan, counts, loop_ended


def run_swarm(task: Task, caller: ModelCaller, *,
              synthesis_caller: ModelCaller | None = None,
              smart_caller: ModelCaller | None = None,
              reviewer_caller: ModelCaller | None = None,
              token_budget: int | None = None,
              max_iterations: int | None = None,
              min_iterations: int | None = None,
              resume_checkpoint: Path | None = None) -> tuple[str | dict[str, str], Blackboard]:
    budget = token_budget or int(os.getenv("SWARM_TOKEN_BUDGET", "3000000"))
    max_iter = max_iterations or int(os.getenv("SWARM_MAX_ITERATIONS", "15"))
    min_iter = min_iterations or int(os.getenv("SWARM_MIN_ITERATIONS", "2"))
    synth_caller = synthesis_caller or caller
    review_caller = reviewer_caller

    if resume_checkpoint is not None:
        # Validation and source reattachment happen before any provider call.
        blackboard, seed_plan, _entries_per_iter, loop_ended = _restore_checkpoint(
            task, resume_checkpoint,
        )
        if not loop_ended and blackboard.iteration >= max_iter:
            raise ValueError("checkpoint iteration is beyond the configured maximum")
        start_iteration = max_iter + 1 if loop_ended else blackboard.iteration + 1
    else:
        blackboard = Blackboard(
            task_instruction=task.instruction,
            documents=_build_doc_statuses(task.documents),
            token_budget=budget,
            started_at=datetime.now(timezone.utc).isoformat(),
            output_dir=task.output_dir,
        )
        seed_plan = {}
        _entries_per_iter = []
        start_iteration = 1

    domain_lens = {}
    if resume_checkpoint is None:
        # Phase 2: Structural profiling (skip unloaded docs in large corpora)
        for doc in blackboard.documents:
            if not doc.text:
                continue
            profile, tokens = _run_structural_profile(doc, task, caller)
            doc.structural_profile = profile
            blackboard.add_tokens_from_last_call(tokens)

        # Phase 3: seed task decomposition and analytical planning
        if review_caller is not None:
            seed_plan, seed_tokens = generate_seed(blackboard, review_caller)
            blackboard.add_tokens_from_last_call(seed_tokens)
            seed_to_signals(seed_plan, blackboard)
            # Put the analytical framework on the blackboard as a strategy entry
            framework = seed_plan.get("analytical_framework", "")
            if framework:
                from .models import WorkerRecord
                blackboard.add_entry(Entry(
                    id=gen_entry_id(), type="strategy", content=framework,
                    created_by=WorkerRecord("seed_planner", "analytical_framework", 0),
                    confidence=0.9, status="active",
                ))
            for criterion in seed_plan.get("completeness_criteria", []):
                if isinstance(criterion, str) and criterion.strip():
                    blackboard.add_entry(Entry(
                        id=gen_entry_id(), type="strategy",
                        content=f"COMPLETENESS CRITERION: {criterion}",
                        created_by=WorkerRecord("seed_planner", "completeness_criteria", 0),
                        confidence=0.9, status="active",
                    ))
            blackboard.save_snapshot("seed")

        # Phase 3a: Domain Lens — DISABLED (W12: unclear value, adds tokens without
        # measurable improvement; seed plan + extraction depth check are sufficient)

        # Phase 3b: If no documents but web search is enabled, add a research signal
        from .web_search import web_search_enabled
        if not blackboard.documents and web_search_enabled():
            from .models import Signal, gen_signal_id
            blackboard.add_signal(Signal(
                id=gen_signal_id(), type="question",
                content=(
                    "No source documents provided. Use web search to find "
                    "information needed to answer the task. Break the question "
                    "into specific search queries."
                ),
                origin_entry="bootstrap", priority="critical",
                status="open", iteration_created=0,
            ))

        # Phase 4: Initial reading (parallel per section)
        entries, tokens = _execute_initial_reading(blackboard, task, caller, seed_plan, domain_lens)
        blackboard.add_entries_batch(entries)
        blackboard.add_tokens(tokens)
        blackboard.entity_overview_state["direct_source_entry_ids"] = [
            entry.id for entry in entries if entry.source and entry.source.document
        ]
        try:
            _, name_tokens = discover_pending_names(blackboard.entity_overview_state, entries, caller)
            blackboard.add_tokens_from_last_call(name_tokens)
            if name_tokens:
                usage = blackboard.entity_overview_state.setdefault("usage", {})
                usage["name_discovery_tokens"] = usage.get("name_discovery_tokens", 0) + name_tokens
                usage["name_discovery_calls"] = usage.get("name_discovery_calls", 0) + 1
        except (ValueError, RuntimeError) as error:
            # Keep unprocessed cards pending; later eligible direct work can retry discovery.
            _record_entity_discovery_failure(blackboard.entity_overview_state, 0, error)
        _run_initial_overviews(blackboard, caller, smart_caller)

        # Phase 4: Extraction depth check — auto re-extract under-covered documents
        for doc in blackboard.documents:
            if not doc.structural_profile:
                continue
            expected = doc.structural_profile.get("numbered_items", 0)
            if not isinstance(expected, (int, float)) or expected <= 0:
                continue
            actual = len([
                e for e in blackboard.entries
                if e.source and e.source.document == doc.name
                and e.status == "active"
                and e.type in ("observation", "analysis", "calculation")
            ])
            if actual < expected * 0.5:
                from .models import Signal, gen_signal_id
                blackboard.add_signal(Signal(
                    id=gen_signal_id(), type="convergence_gap",
                    content=(
                        f"Document '{doc.name}' has ~{int(expected)} enumerable items "
                        f"but only {actual} extracted. Need targeted re-extraction."
                    ),
                    origin_entry="extraction_depth_check", priority="critical",
                    status="open", iteration_created=0,
                ))

        # Phase 5: Prioritize initial signals
        unp = [s for s in blackboard.signals if s.status == "open"]
        if unp:
            blackboard.add_tokens(prioritize_signals(blackboard, unp, review_caller or caller))

    # Phase 6: Swarm loop
    # W13 revert: flash-lite (caller) for orchestrator+workers. W12 showed
    # flash causes iteration stalling (80% stall rate) and 43-63% fewer
    # analysis entries. Flash-lite produces tighter, more focused dispatches.
    # Flash stays for reviewer/synthesis/signal-prioritization only.
    loop_caller = caller
    _premium_caller_env = os.getenv("SWARM_FABLE_MODEL", "")
    _premium_caller = None
    if _premium_caller_env:
        from ..providers.anthropic import AnthropicCaller
        _premium_caller = AnthropicCaller(model=_premium_caller_env)

    for iteration in range(start_iteration, max_iter + 1):
        blackboard.iteration = iteration
        blackboard.expire_old_signals()
        if iteration == 1 or iteration % 4 == 0:
            blackboard.save_snapshot(f"pre_{iteration}")

        # Iteration cycling: 2 flash, 1 Opus (iterations 3, 6, 9, 12, 15)
        iter_caller = loop_caller
        if _premium_caller is not None and iteration % 3 == 0:
            iter_caller = _premium_caller

        # Analysis mode shift: after iteration 8, if obs:analysis ratio > 3:1,
        # force the orchestrator to stop extracting and start reasoning
        analysis_mode_override = ""
        if iteration > 8:
            _active = [e for e in blackboard.entries if e.status == "active"]
            _obs = sum(1 for e in _active if e.type == "observation")
            _ana = sum(1 for e in _active if e.type in ("analysis", "calculation"))
            _ratio = _obs / max(_ana, 1)
            if _ratio > 3.0:
                analysis_mode_override = (
                    f"ANALYSIS MODE: obs:analysis ratio is {_ratio:.1f}:1 "
                    f"({_obs} observations, {_ana} analyses/calculations). "
                    f"You are over-extracting and under-reasoning. "
                    f"Dispatch ONLY analysis, calculation, and strategy workers. "
                    f"NO observation workers. Focus on: "
                    f"(1) What conclusions are MISSING from existing observations? "
                    f"(2) What calculations should be performed from extracted numbers? "
                    f"(3) What cross-document comparisons need to be made? "
                    f"(4) What issues should be flagged based on findings so far?"
                )

        # Diminishing returns: if last 2 iterations each added ≤3 entries,
        # nudge the orchestrator toward convergence
        _diminishing = ""
        if len(_entries_per_iter) >= 2 and all(c <= 3 for c in _entries_per_iter[-2:]):
            _diminishing = (
                f"DIMINISHING RETURNS: last 2 iterations added only "
                f"{_entries_per_iter[-2]} and {_entries_per_iter[-1]} entries. "
                f"Consider converging if analysis entries exist."
            )
        _combined_override = " ".join(
            p for p in [analysis_mode_override, _diminishing] if p
        )

        inventory = overview_inventory(blackboard.entity_overview_state, blackboard.entries, iteration)
        orch, orch_tokens = run_orchestrator(
            blackboard, iter_caller, override=_combined_override,
            entity_overviews=inventory,
        )
        blackboard.add_tokens_from_last_call(orch_tokens)

        if orch.get("action") == "converge" and iteration >= min_iter:
            converged, conv_tokens = check_convergence(blackboard, orch, iter_caller)
            blackboard.add_tokens_from_last_call(conv_tokens)
            if converged:
                blackboard.save_snapshot("converged")
                blackboard.save_snapshot(f"post_{iteration}")
                blackboard.save_checkpoint(seed_plan, _entries_per_iter, loop_ended=True)
                break
            _reject = "Convergence rejected. Address the gaps identified."
            if analysis_mode_override:
                _reject += " " + analysis_mode_override
            orch, t = run_orchestrator(
                blackboard, iter_caller, override=_reject, entity_overviews=inventory,
            )
            blackboard.add_tokens_from_last_call(t)
            if orch.get("action") == "converge":
                converged2, t2 = check_convergence(blackboard, orch, iter_caller)
                blackboard.add_tokens_from_last_call(t2)
                if converged2:
                    blackboard.save_snapshot("converged_retry")
                    blackboard.save_snapshot(f"post_{iteration}")
                    blackboard.save_checkpoint(seed_plan, _entries_per_iter, loop_ended=True)
                    break
                _force = "You MUST produce workers. Do NOT converge. Find remaining gaps."
                if analysis_mode_override:
                    _force += " " + analysis_mode_override
                orch, t3 = run_orchestrator(
                    blackboard, iter_caller, override=_force, entity_overviews=inventory,
                )
                blackboard.add_tokens_from_last_call(t3)

        tasks_list = _prepare_entity_work(blackboard, orch.get("workers", []), iteration, inventory)
        if not tasks_list:
            _entries_per_iter.append(0)
            blackboard.save_snapshot(f"post_{iteration}")
            blackboard.save_checkpoint(seed_plan, _entries_per_iter,
                                       loop_ended=iteration == max_iter)
            continue

        if analysis_mode_override:
            _analytical = [
                t for t in tasks_list
                if t.get("expected_output_type", "observation") != "observation"
            ]
            _critical_reads = [
                t for t in tasks_list
                if t.get("expected_output_type", "observation") == "observation"
                and any(
                    s.id in t.get("addresses_signals", [])
                    for s in blackboard.signals
                    if s.status == "open" and s.priority == "critical"
                )
            ]
            if _analytical or _critical_reads:
                tasks_list = _analytical + _critical_reads
            else:
                from .convergence import analytical_steering
                _steering_tasks, _steer_tokens = analytical_steering(
                    blackboard, iter_caller,
                )
                blackboard.add_tokens_from_last_call(_steer_tokens)
                if _steering_tasks:
                    tasks_list = _steering_tasks
                else:
                    _obs_tasks = [
                        t for t in tasks_list
                        if t.get("expected_output_type", "observation") == "observation"
                    ]
                    tasks_list = _obs_tasks[:1]

        if not tasks_list:
            _entries_per_iter.append(0)
            blackboard.save_snapshot(f"post_{iteration}")
            blackboard.save_checkpoint(seed_plan, _entries_per_iter,
                                       loop_ended=iteration == max_iter)
            continue

        outputs = execute_workers_parallel(tasks_list, blackboard, iter_caller)

        _entries_per_iter.append(_commit_entity_worker_outputs(blackboard, outputs, iter_caller, iteration))
        _run_initial_overviews(blackboard, caller, smart_caller)
        _after_document_round(blackboard, outputs, caller)

        new_sigs = [
            s for s in blackboard.signals
            if s.status == "open" and s.iteration_created == iteration
        ]
        if new_sigs:
            blackboard.add_tokens(prioritize_signals(blackboard, new_sigs, review_caller or caller))

        if blackboard.budget_used_pct() >= 85:
            blackboard.save_snapshot("budget_exhausted")
            blackboard.save_snapshot(f"post_{iteration}")
            blackboard.save_checkpoint(seed_plan, _entries_per_iter, loop_ended=True)
            break

        blackboard.save_snapshot(f"post_{iteration}")
        blackboard.save_checkpoint(seed_plan, _entries_per_iter,
                                   loop_ended=iteration == max_iter)

    # Phase after extraction: direct analysis
    if review_caller is not None:
        analysis_entries, analysis_tokens = run_direct_analysis(
            blackboard, seed_plan, review_caller,
        )
        blackboard.add_entries_batch(analysis_entries)
        blackboard.add_tokens_from_last_call(analysis_tokens)
        blackboard.save_snapshot("post_analysis")

    # Phase 6: Supervisor review (smarter model, only if available)
    if review_caller is not None:
        for review_round in range(2):
            approved, gaps, rev_tokens = supervisor_review(blackboard, review_caller)
            blackboard.add_tokens_from_last_call(rev_tokens)
            if approved:
                blackboard.save_snapshot(f"supervisor_approved_{review_round}")
                break
            # Supervisor found gaps — add as critical signals and re-enter swarm
            from .models import Signal, gen_signal_id
            for gap in gaps:
                blackboard.add_signal(Signal(
                    id=gen_signal_id(), type="convergence_gap", content=gap,
                    origin_entry="supervisor_review", priority="critical",
                    status="open", iteration_created=blackboard.iteration,
                ))
            # Run a few more iterations to address supervisor's gaps
            for extra_iter in range(1, 4):
                blackboard.iteration += 1
                if blackboard.budget_used_pct() >= 90:
                    break
                inventory = overview_inventory(
                    blackboard.entity_overview_state, blackboard.entries, blackboard.iteration,
                )
                orch, t = run_orchestrator(blackboard, loop_caller, entity_overviews=inventory)
                blackboard.add_tokens_from_last_call(t)
                if orch.get("action") == "converge":
                    break
                tasks_list = _prepare_entity_work(
                    blackboard, orch.get("workers", []), blackboard.iteration, inventory,
                )
                if not tasks_list:
                    continue
                outputs = execute_workers_parallel(tasks_list, blackboard, loop_caller)
                _commit_entity_worker_outputs(blackboard, outputs, loop_caller, blackboard.iteration)
                _run_initial_overviews(blackboard, caller, smart_caller)
                _after_document_round(blackboard, outputs, caller)
            blackboard.save_snapshot(f"post_supervisor_{review_round}")

    _run_due_deltas(blackboard, caller, final=True)
    enforce_source_custody(blackboard, "pre_state_conversion")
    _sync_entity_state(blackboard)
    _run_identity_reviews(blackboard, caller, smart_caller)

    # Phase 7a: State conversion review — convert observations into analytical state
    if review_caller is not None:
        sc_entries, sc_report, sc_tokens = run_state_conversion_review(
            blackboard, seed_plan, review_caller,
        )
        blackboard.add_entries_batch(sc_entries)
        blackboard.add_tokens_from_last_call(sc_tokens)

        # Phase 7b: Plan coverage review — adversarial seed/criteria coverage check
        cov_report, cov_tokens = run_plan_coverage_review(
            blackboard, seed_plan, review_caller,
            domain_lens=domain_lens,
        )
        blackboard.add_tokens_from_last_call(cov_tokens)

        # Materialize plan coverage as substantive blackboard entries
        active_now = [e for e in blackboard.entries if e.status == "active"]
        coverage_entries = coverage_report_to_entries(
            seed_plan, cov_report, blackboard.iteration,
            active_entries=active_now,
            domain_lens=domain_lens,
        )
        blackboard.add_entries_batch(coverage_entries)
        blackboard.save_snapshot("post_state_conversion")

        # Phase 7c: bounded pre-obligation state repair from high/critical
        # coverage gaps. This strengthens state before obligations rather than
        # patching final output.
        repair_entries, repair_report, repair_tokens = run_plan_coverage_state_repair(
            blackboard, coverage_entries, review_caller,
        )
        blackboard.add_entries_batch(repair_entries)
        if repair_tokens:
            blackboard.add_tokens_from_last_call(repair_tokens)

        # Persist reports for diagnostics
        if blackboard.output_dir:
            swarm_dir = os.path.join(blackboard.output_dir, "swarm")
            os.makedirs(swarm_dir, exist_ok=True)
            with open(os.path.join(swarm_dir, "state_conversion_review.json"), "w", encoding="utf-8") as f:
                json.dump(sc_report, f, indent=2)
            with open(os.path.join(swarm_dir, "plan_coverage_review.json"), "w", encoding="utf-8") as f:
                json.dump(cov_report, f, indent=2)
            with open(os.path.join(swarm_dir, "plan_coverage_repair.json"), "w", encoding="utf-8") as f:
                json.dump(repair_report, f, indent=2)

        blackboard.save_snapshot("post_state_repair")

    custody_report = enforce_source_custody(blackboard, "post_state_repair")
    _sync_entity_state(blackboard)

    if review_caller is not None and blackboard_maintenance_enabled():
        _, maintenance_tokens = run_blackboard_maintenance(
            blackboard, seed_plan, review_caller,
        )
        if maintenance_tokens:
            blackboard.add_tokens_from_last_call(maintenance_tokens)
        custody_report = enforce_source_custody(blackboard, "post_blackboard_maintenance")
        _sync_entity_state(blackboard)

    debt_sensor_report = None
    if review_caller is not None and debt_sensors_enabled():
        debt_sensor_report, debt_sensor_tokens = run_debt_sensors(
            blackboard, seed_plan, review_caller,
        )
        if debt_sensor_tokens:
            blackboard.add_tokens_from_last_call(debt_sensor_tokens)
        custody_report = enforce_source_custody(blackboard, "post_debt_sensors")
        _sync_entity_state(blackboard)

    derived_work_report = None
    if review_caller is not None and calculation_debt_enabled():
        derived_work_report, calc_debt_tokens = run_calculation_debt_detection(
            blackboard, seed_plan, review_caller,
        )
        if calc_debt_tokens:
            blackboard.add_tokens_from_last_call(calc_debt_tokens)
        custody_report = enforce_source_custody(blackboard, "post_calculation_debt_detection")
        _sync_entity_state(blackboard)

    _sync_entity_state(blackboard)
    blackboard.save_snapshot("post_processing")

    # Phase 7b: Synthesis Readiness Gate
    active_entries = [e for e in blackboard.entries if e.status == "active"]
    sourced_active = [
        e for e in active_entries
        if e.source and e.source.document
        and e.type in ("observation", "analysis", "calculation")
    ]
    total_docs = len(blackboard.documents)
    read_docs = sum(1 for d in blackboard.documents if d.read_status != "unread")
    quarantined = sum(1 for e in blackboard.entries if e.status == "source_quarantined")
    evidentiary_entries = sum(
        1 for e in blackboard.entries if e.status in ("active", "source_quarantined")
    )
    quarantine_rate = quarantined / evidentiary_entries if evidentiary_entries > 0 else 0.0

    synthesis_blocked = False
    block_reasons = []
    if total_docs > 0 and read_docs == 0:
        block_reasons.append(f"zero documents read out of {total_docs}")
        synthesis_blocked = True
    if not sourced_active and total_docs > 0:
        block_reasons.append("zero source-grounded active entries")
        synthesis_blocked = True
    if quarantine_rate > 0.8 and evidentiary_entries > 20:
        block_reasons.append(f"quarantine rate {quarantine_rate:.0%} exceeds 80% threshold")
        synthesis_blocked = True

    if synthesis_blocked:
        blackboard.save_snapshot("synthesis_blocked")
        deliverable = (
            f"# Synthesis Blocked — Insufficient Source Evidence\n\n"
            f"The system determined that synthesis cannot proceed reliably:\n"
            + "\n".join(f"- {r}" for r in block_reasons)
            + f"\n\n**Active entries:** {len(active_entries)}\n"
            f"**Source-grounded active:** {len(sourced_active)}\n"
            f"**Documents read:** {read_docs}/{total_docs}\n"
            f"**Quarantine rate:** {quarantine_rate:.0%}\n"
        )
        return deliverable, blackboard

    # Phase 7c: build synthesis obligations
    obligations = []
    if review_caller is not None:
        obligations, obl_tokens = build_synthesis_obligations(
            blackboard, seed_plan, review_caller,
        )
        blackboard.add_tokens_from_last_call(obl_tokens)
        blackboard.save_snapshot("post_obligations")

    # Phase 7d: artifact-native structural contracts
    deliverables_map_pre = task.metadata.get("deliverables", {})
    if deliverables_map_pre:
        contract_caller = review_caller or caller
        contract_items, contract_tokens = build_artifact_contracts(
            blackboard, deliverables_map_pre, contract_caller,
        )
        blackboard.add_tokens_from_last_call(contract_tokens)
        contracts_to_signals(contract_items, blackboard)
        obligations.extend(contract_items)

    # Phase 8: Curate + Combine obligations + Synthesize
    must_include, cur_tokens = curate_entries(blackboard, caller)
    blackboard.add_tokens_from_last_call(cur_tokens)

    # Merge obligations into must_include (obligations first, deduped)
    if obligations:
        seen = set()
        combined = []
        for o in obligations:
            key = (o.get("summary", "") if isinstance(o, dict) else str(o))[:60].lower().strip()
            if key not in seen:
                combined.append(o)
                seen.add(key)
        for m in must_include:
            key = (m.get("summary", "") if isinstance(m, dict) else str(m))[:60].lower().strip()
            if key not in seen:
                combined.append(m)
                seen.add(key)
        must_include = combined

    deliverables_map = task.metadata.get("deliverables", {})
    artifact_commitments = build_artifact_commitments(blackboard, deliverables_map)
    if artifact_commitments:
        seen = {
            (m.get("entry_id", ""), m.get("target_file", ""))
            for m in must_include if isinstance(m, dict)
        }
        for commitment in artifact_commitments:
            key = (commitment.get("entry_id", ""), commitment.get("target_file", ""))
            if key not in seen:
                must_include.insert(0, commitment)
                seen.add(key)

    # Phase 8b: consolidate near-duplicate items, then build synthesis packet
    must_include = consolidate_items(must_include)
    synthesis_packet = build_synthesis_packet(must_include, blackboard)
    write_synthesis_packet_report(synthesis_packet, blackboard.output_dir)

    if derived_work_report or debt_sensor_report:
        write_pending_survival_trace(
            blackboard.output_dir,
            derived_work_report,
            must_include,
            debt_sensor_report,
        )

    if _should_use_file_scoped_synthesis(deliverables_map):
        deliverable, synth_tokens = synthesize_file_deliverables(
            blackboard, synthesis_packet, deliverables_map, [], synth_caller,
        )
    else:
        deliverable, synth_tokens = synthesize_deliverable(
            blackboard, synthesis_packet, synth_caller,
        )
    blackboard.add_tokens_from_last_call(synth_tokens)

    # Shadow judge audit — DISABLED (W12: late-stage patching with unclear ROI)

    if source_claim_verification_enabled():
        deliverable, claim_tokens, _ = verify_source_claims(
            deliverable, blackboard, synth_caller,
        )
        blackboard.add_tokens_from_last_call(claim_tokens)

    blackboard.save_snapshot("final")

    return deliverable, blackboard


PROFILE_THRESHOLD = 50


_MAX_INITIAL_MATERIALIZE = 30

def _materialize_selected_docs(blackboard: Blackboard, seed_plan: dict | None) -> None:
    """Load text for documents the seed planner selected.

    For small corpora (<=PROFILE_THRESHOLD), load everything.
    For large corpora, use a multi-strategy approach:
      1. extraction_focus refs from seed plan (exact + fuzzy match)
      2. Task instruction keywords matched against directory paths
      3. Read-request signals matched against document paths
    """
    lazy_docs = [(ds, ds.source_path)
                 for ds in blackboard.documents if not ds.is_loaded]
    if not lazy_docs:
        return

    if len(lazy_docs) <= PROFILE_THRESHOLD:
        for ds, _ in lazy_docs:
            ds.materialize()
        return

    must_load: set[str] = set()
    should_load: set[str] = set()

    if seed_plan:
        for focus in seed_plan.get("extraction_focus", []):
            if isinstance(focus, dict):
                ref = focus.get("document", "")
                if ref:
                    for ds, path in lazy_docs:
                        if (ref == ds.name or ref in ds.name or ds.name in ref
                                or ref.lower() in path.lower()):
                            must_load.add(ds.name)

    for sig in blackboard.signals:
        if sig.type == "read_request" and sig.status == "open":
            ref = sig.content.split("'")[1] if "'" in sig.content else ""
            if ref:
                ref_norm = ref.lower().replace("\\", "/")
                for ds, path in lazy_docs:
                    path_norm = path.lower().replace("\\", "/")
                    if (ref in ds.name or ds.name in ref
                            or ref_norm in path_norm
                            or ds.name.lower() in ref_norm):
                        must_load.add(ds.name)

    _dir_match_from_task(blackboard.task_instruction, lazy_docs, must_load, should_load)

    to_materialize = []
    for ds, _ in lazy_docs:
        if ds.name in must_load:
            to_materialize.append(ds)
    remaining = _MAX_INITIAL_MATERIALIZE - len(to_materialize)
    for ds, _ in lazy_docs:
        if ds.name in should_load and ds.name not in must_load and remaining > 0:
            to_materialize.append(ds)
            remaining -= 1

    if not to_materialize:
        return

    def _do_materialize(ds):
        ds.materialize()

    with ThreadPoolExecutor(max_workers=min(4, len(to_materialize))) as pool:
        list(pool.map(_do_materialize, to_materialize))


def _dir_match_from_task(task_instruction: str, lazy_docs: list,
                        must_load: set[str], should_load: set[str]) -> None:
    """Match task instruction keywords against document directory paths.

    must_load: filing-type directory matches (high priority, no cap)
    should_load: broader matches like year-based or IR dirs (lower priority, capped)
    """
    import re
    from collections import defaultdict

    docs_by_dir: dict[str, list[tuple]] = defaultdict(list)
    for ds, path in lazy_docs:
        norm = path.replace("\\", "/")
        parts = norm.split("/")
        dir_key = "/".join(parts[-3:-1]) if len(parts) >= 3 else (
            parts[-2] if len(parts) >= 2 else "(root)")
        docs_by_dir[dir_key].append((ds, path))

    filing_patterns = re.findall(
        r'\b(10-[KQ]|424B\d|8-[KA]|S-[18]|DEF.?14A?|PRE.?14A?|'
        r'SC.?13[GD]|proxy|prospectus|annual.report|'
        r'press.releas\w*|news.releas\w*|earnings|analyst)\b',
        task_instruction, re.IGNORECASE,
    )
    year_patterns = re.findall(r'\b((?:19|20)\d{2})\b', task_instruction)

    for pattern in filing_patterns:
        pattern_lower = pattern.lower().replace(" ", "").replace("_", "").replace("-", "")
        for dir_key, docs in docs_by_dir.items():
            dir_norm = dir_key.lower().replace(" ", "").replace("_", "").replace("-", "")
            if pattern_lower in dir_norm or dir_norm in pattern_lower:
                for ds, _ in docs:
                    must_load.add(ds.name)

    if year_patterns and must_load:
        for ds, path in lazy_docs:
            if ds.name in must_load:
                continue
            path_lower = path.lower()
            for year in year_patterns:
                if year in path_lower or year in ds.name:
                    should_load.add(ds.name)
                    break

    task_lower = task_instruction.lower()
    has_ir_ref = (re.search(r'\bIR\b', task_instruction) is not None
                  or "press release" in task_lower
                  or "news release" in task_lower)
    if has_ir_ref:
        for dir_key, docs in docs_by_dir.items():
            if "ir" in dir_key.lower().split("/") or "news" in dir_key.lower():
                for ds, _ in docs:
                    should_load.add(ds.name)


def _build_doc_statuses(documents: list[Document]) -> list[DocumentStatus]:
    statuses = []
    large_corpus = len(documents) > PROFILE_THRESHOLD
    for doc in documents:
        raw_path = doc.metadata.get("path", "")
        src_path = str(Path(raw_path).resolve()) if raw_path else ""
        if large_corpus and doc._loader is not None:
            statuses.append(DocumentStatus(
                id=doc.id, name=doc.name, size_bytes=doc.size_bytes,
                source_path=src_path,
                headings=[], sections_unread=[],
                section_index=None, text="",
                read_status="unread",
            ))
            statuses[-1]._lazy_doc = doc
        else:
            idx = build_section_index(doc.text)
            statuses.append(DocumentStatus(
                id=doc.id, name=doc.name, size_bytes=doc.size_bytes,
                source_path=src_path,
                headings=[s.name for s in idx.sections],
                sections_unread=[s.name for s in idx.sections],
                section_index=idx, text=doc.text,
            ))
    return statuses


def _run_structural_profile(doc: DocumentStatus, task: Task,
                            caller: ModelCaller) -> tuple[dict, int]:
    sample = (
        f"HEADINGS:\n{chr(10).join(doc.headings[:50])}\n\n"
        f"FIRST 2000 CHARS:\n{doc.text[:2000]}\n\n"
        f"LAST 500 CHARS:\n{doc.text[-500:]}"
    )
    prompt = f"""Examine this document's structure:
Document: {doc.name} ({doc.size_bytes} bytes)

{sample}

Report:
1. numbered_items: count of individually numbered/lettered items (e.g., clauses, requests, conditions)
2. tables: data table count
3. sections: major section count
4. document_type: contract|brief|report|filing|letter|exhibit|memo|agreement|amendment|schedule|other
5. key_entities: main parties/companies/persons (list)
6. estimated_complexity: simple|medium|complex

Return JSON with these fields."""
    payload, tokens = call_model(caller, prompt, max_tokens=1024)
    return payload, tokens


def _execute_initial_reading(blackboard: Blackboard, task: Task,
                             caller: ModelCaller,
                             seed_plan: dict | None = None,
                             domain_lens: dict | None = None) -> tuple[list[Entry], int]:
    CHUNK_SIZE = 24000
    CHUNK_OVERLAP = 2000

    _materialize_selected_docs(blackboard, seed_plan)

    read_tasks = []
    for ds in blackboard.documents:
        if not ds.is_loaded:
            continue
        # Build density guidance from structural profile
        density_hint = ""
        if ds.structural_profile:
            n_items = ds.structural_profile.get("numbered_items", 0)
            if isinstance(n_items, (int, float)) and n_items > 0:
                density_hint = (
                    f"\nDENSITY GUIDANCE: This document contains approximately "
                    f"{int(n_items)} individually enumerable items. Extract ONE "
                    f"finding PER ITEM. Target at least {int(n_items)} findings "
                    f"from this document."
                )

        for section in ds.section_index.sections:
            if section.level > 2:
                continue
            text = ds.text[section.start_char:section.end_char]
            if len(text.strip()) < 50:
                continue
            if len(text) <= CHUNK_SIZE:
                read_tasks.append({
                    "doc_name": ds.name, "section_name": section.name,
                    "section_text": text, "density_hint": density_hint,
                    "seed_guidance": _initial_reading_seed_guidance(seed_plan, ds.name, domain_lens),
                })
            else:
                chunk_idx = 0
                offset = 0
                while offset < len(text):
                    chunk = text[offset:offset + CHUNK_SIZE]
                    if len(chunk.strip()) < 50:
                        break
                    chunk_idx += 1
                    read_tasks.append({
                        "doc_name": ds.name,
                        "section_name": f"{section.name} (part {chunk_idx})",
                        "section_text": chunk, "density_hint": density_hint,
                        "seed_guidance": _initial_reading_seed_guidance(seed_plan, ds.name, domain_lens),
                    })
                    offset += CHUNK_SIZE - CHUNK_OVERLAP

    all_entries: list[Entry] = []
    total_tokens = 0

    def read_one(rt):
        density_hint = rt.get("density_hint", "")
        seed_guidance = rt.get("seed_guidance", "")
        seed_block = ""
        if seed_guidance:
            seed_block = f"""
SEED-GUIDED INVESTIGATION LENS:
{seed_guidance}

Use this lens to decide which details are most material and which implicit
subquestions need evidence. Do not skip unrelated exact facts, terms, numbers,
dates, or provisions; the seed lens focuses extraction, it does not narrow it.
"""
        prompt = f"""Read this section of "{rt['doc_name']}" ({rt['section_name']}) and extract EVERY fact, term, and data point.

TASK: {task.instruction}
{density_hint}
{seed_block}

SOURCE:
{rt['section_text']}

EXTRACTION RULES — follow these EXACTLY:
1. Extract EVERY dollar amount, percentage, date, deadline, and time period
2. Extract EVERY party name with full legal entity designation (e.g., "Inc.", "AG", "LLC")
3. Extract EVERY numbered or lettered item in any list, schedule, or exhibit — EACH ONE SEPARATELY
4. Extract EVERY defined term and its definition
5. Extract EVERY obligation, condition, requirement, or restriction
6. Extract EVERY payment term: upfront amounts, milestones, royalties, equity investments
7. If a table exists, extract EACH ROW as a separate finding
8. Do NOT summarize — if the text says "$75,000,000 upfront payment due within 30 days", that is ONE finding with the exact amount and timeline
9. Aim for 20-50 findings per section. If you have fewer than 10, you are likely summarizing instead of enumerating.

For each finding:
- content: the specific fact with EXACT numbers, names, dates
- type: observation (for facts), calculation (for numbers), analysis (for implications)
- confidence: 0.9 for directly quoted facts, 0.7 for inferences
- epistemic_classification: fact | adversarial_claim | expert_opinion | strategic

Return JSON: {{"findings": [...]}}"""
        payload, tokens = call_model(caller, prompt, max_tokens=8192)
        entries = parse_worker_output(
            payload, 0, f"reader_{rt['doc_name'][:20]}", "initial_reading",
        )
        # Backfill source on entries that lack it — we KNOW the doc and section
        for e in entries:
            if not e.source or not e.source.document:
                e.source = EntrySource(
                    document=rt["doc_name"],
                    section=rt["section_name"],
                    evidence=e.source.evidence if e.source else "",
                )
        return entries, tokens, rt["doc_name"], rt["section_name"]

    max_w = min(len(read_tasks), 10)
    if max_w > 0:
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = [pool.submit(read_one, rt) for rt in read_tasks]
            for f in futures:
                entries, tokens, doc_name, sec_name = f.result()
                all_entries.extend(entries)
                total_tokens += tokens
                for ds in blackboard.documents:
                    if ds.name == doc_name:
                        ds.mark_section_read(sec_name)
                        # Also mark parent section for chunked reads
                        if " (part " in sec_name:
                            parent = sec_name.split(" (part ")[0]
                            ds.mark_section_read(parent)

    return all_entries, total_tokens


def _initial_reading_seed_guidance(seed_plan: dict | None, doc_name: str,
                                   lens: dict | None = None) -> str:
    """Format seed plan + domain lens as guidance for first-pass readers."""
    if not isinstance(seed_plan, dict) or not seed_plan:
        return ""

    parts: list[str] = []

    questions = [
        str(q).strip()
        for q in seed_plan.get("key_questions", [])
        if isinstance(q, str) and q.strip()
    ]
    if questions:
        parts.append("Key questions:\n" + "\n".join(f"- {q}" for q in questions))

    doc_focus = []
    for focus in seed_plan.get("extraction_focus", []):
        if not isinstance(focus, dict):
            continue
        target_doc = str(focus.get("document", "")).strip()
        text = str(focus.get("focus", "")).strip()
        if not target_doc or not text:
            continue
        target_l = target_doc.lower()
        doc_l = doc_name.lower()
        if target_l in doc_l or doc_l in target_l or target_l in {"all", "all documents"}:
            doc_focus.append(text)
    if doc_focus:
        parts.append(
            f"Document-specific focus for {doc_name}:\n"
            + "\n".join(f"- {focus}" for focus in doc_focus)
        )

    framework = str(seed_plan.get("analytical_framework", "")).strip()
    if framework:
        parts.append("Analytical framework:\n" + framework)

    context = seed_plan.get("context_enrichment", "")
    if isinstance(context, dict):
        context_text = json.dumps(context, ensure_ascii=True)
    else:
        context_text = str(context).strip()
    if context_text:
        parts.append("Context enrichment notes:\n" + context_text)

    criteria = [
        str(c).strip()
        for c in seed_plan.get("completeness_criteria", [])
        if isinstance(c, str) and c.strip()
    ]
    if criteria:
        parts.append(
            "Completeness signals:\n"
            + "\n".join(f"- {criterion}" for criterion in criteria)
        )

    if lens:
        lens_text = format_lens_guidance(lens)
        if lens_text:
            parts.append(lens_text)

    return "\n\n".join(parts)
