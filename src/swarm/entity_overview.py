"""Small, run-local entity catalogue and overview worker helpers."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .models import Entry, ModelCaller
from .worker_dispatch import call_model, parse_worker_output


_LEGAL_FORMS = {"ag", "gmbh", "inc", "incorporated", "ltd", "llc", "lp", "plc", "sa"}
INITIAL_OVERVIEW_CARDS = 6
REFRESH_OVERVIEW_CARDS = 4
INITIAL_OVERVIEW_BATCH_SIZE = 3

_OVERVIEW_SECTIONS = (
    "entity_profiles",
    "relationships_and_distinctions",
    "unresolved_or_conflicting_evidence",
)


def normalize_name(name: str) -> str:
    """Normalize a spelling for retrieval; this is not identity resolution."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(token for token in tokens if token not in _LEGAL_FORMS)


def parse_discovered_names(payload: dict) -> list[str]:
    """Accept the names-only response shape without treating malformed data as empty work."""
    values = payload.get("names", payload.get("entities", payload.get("value", [])))
    if not isinstance(values, list):
        raise ValueError("name discovery response must contain a names array")
    return list(dict.fromkeys(name.strip() for name in values if isinstance(name, str) and name.strip()))


def build_name_discovery_prompt(entries: list[Entry]) -> str:
    """Extract names and explicitly documented aliases in the same model call."""
    cards = "\n\n".join(f"[{entry.id}] {entry.content}" for entry in entries)
    return f"""List every organization and person name present in these original cards.
Return JSON only as {{"names": ["exact spelling"], "aliases": [
  {{"name": "full name", "short_form": "explicit short name", "source_card_id": "card ID"}}
]}}. Include legal forms and all alias spellings in names.
Only mark aliases when a supplied card explicitly identifies a short form, abbreviation,
trade name, or alias for that full name. Copy both names exactly from that card.
Co-occurrence, similar spelling, and a shared prefix do NOT establish an alias.
Do not join distinct companies or people. Omit uncertain relationships; return an empty
aliases array if none are explicit. Do not analyze or return blackboard findings.

ORIGINAL CARDS:
{cards}"""


@dataclass
class NameCatalogue:
    """Candidate retrieval groups keyed by a normalized discovered spelling."""

    variants: dict[str, set[str]] = field(default_factory=dict)

    def add(self, names: list[str]) -> list[str]:
        added = []
        for name in names:
            key = self.key_for(name)
            if not key:
                continue
            if key not in self.variants:
                self.variants[key] = set()
                added.append(key)
            self.variants[key].add(name)
        return added

    def key_for(self, name: str) -> str:
        """Reuse a persisted alias group when a spelling is discovered again."""
        normalized = normalize_name(name)
        return next((key for key, names in self.variants.items()
                     if any(normalize_name(value) == normalized for value in names)), normalized)

    def names_for(self, entity_id: str) -> list[str]:
        return sorted(self.variants.get(entity_id, set()))

    def match(self, entry: Entry, entity_id: str) -> str | None:
        """Return a broad retrieval reason, preserving uncertainty for the overview worker."""
        text = entry.content
        normalized_text = normalize_name(text)
        for spelling in self.names_for(entity_id):
            if re.search(rf"(?<!\w){re.escape(spelling)}(?!\w)", text, re.IGNORECASE):
                return f"literal spelling: {spelling}"
        if entity_id and re.search(rf"(?<!\w){re.escape(entity_id)}(?!\w)", normalized_text):
            return "normalized name"

        # ponytail: conservative fuzzy retrieval only; use richer matching if real runs miss known variants.
        target_tokens = entity_id.split()
        text_tokens = normalized_text.split()
        width = len(target_tokens)
        for index in range(len(text_tokens) - width + 1):
            phrase = " ".join(text_tokens[index:index + width])
            if (target_tokens and phrase[:1] == entity_id[:1]
                    and SequenceMatcher(None, entity_id, phrase).ratio() >= 0.88):
                return f"close spelling candidate: {phrase}"
        return None

    def matched_entries(self, entries: list[Entry], entity_id: str) -> list[tuple[Entry, str]]:
        return [(entry, reason) for entry in entries if (reason := self.match(entry, entity_id))]


def _catalogue_from_state(state: dict) -> NameCatalogue:
    return NameCatalogue({key: set(value) for key, value in state.get("variants", {}).items()})


def entry_state(entry: Entry) -> str:
    """Small content/status fingerprint for refresh decisions, not a revision system."""
    source = entry.source.document if entry.source else ""
    return sha256(f"{entry.status}\0{source}\0{entry.content}".encode()).hexdigest()


def record_discovered_names(state: dict, payload: dict, entries: list[Entry]) -> list[str]:
    """Apply source-backed alias links without extra calls or fuzzy group merging."""
    catalogue = _catalogue_from_state(state)
    previous_keys = set(catalogue.variants)
    catalogue.add(parse_discovered_names(payload))
    cards = {entry.id: entry.content for entry in entries}
    aliases = payload.get("aliases", [])
    if not isinstance(aliases, list):
        raise ValueError("name discovery aliases must be an array")
    validated = []
    for alias in aliases:
        if not isinstance(alias, dict):
            continue
        name, short, card_id = (alias.get(field) for field in ("name", "short_form", "source_card_id"))
        if not all(isinstance(value, str) and value.strip() for value in (name, short, card_id)):
            continue
        if not normalize_name(name) or not normalize_name(short):
            continue
        text = cards.get(card_id, "")
        # Presence validates provenance; the worker must establish the explicit relationship.
        if all(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.IGNORECASE)
               for value in (name, short)):
            validated.append((name, short))

    records = state.setdefault("overviews", {})
    for name, short in validated:
        # An ambiguous short form must not bridge different full-name groups.
        targets = {normalize_name(full) for full, form in validated
                   if normalize_name(form) == normalize_name(short)}
        if len(targets) != 1:
            continue
        catalogue.add([name, short])
        target, source = catalogue.key_for(name), catalogue.key_for(short)
        if target == source:
            continue
        if len({normalize_name(value) for value in catalogue.variants[source]}) > 1:
            # Do not use an already-linked short form to join a different full name later.
            continue
        # Keep the existing overview's ID, including when the short form was found first.
        if source in records and target not in records:
            target, source = source, target
        catalogue.variants[target].update(catalogue.variants.pop(source))
        records.pop(source, None)
        for field in ("pending", "failed"):
            if source in state.get(field, []):
                state[field] = list(dict.fromkeys(target if key == source else key
                                                 for key in state[field]))
    state["variants"] = {key: sorted(value) for key, value in catalogue.variants.items()}
    return sorted(set(catalogue.variants) - previous_keys)


def discover_pending_names(state: dict, entries: list[Entry], caller: ModelCaller) -> tuple[list[str], int]:
    """Discover only unprocessed original cards; failures leave them pending."""
    processed = set(state.setdefault("discovered_card_ids", []))
    pending = [
        entry for entry in entries
        if entry.status == "active" and entry.type != "entity_overview" and entry.id not in processed
    ]
    if not pending:
        return [], 0
    payload, tokens = call_model(caller, build_name_discovery_prompt(pending))
    added = record_discovered_names(state, payload, pending)
    state["discovered_card_ids"] = sorted(processed | {entry.id for entry in pending})
    return added, tokens


def overview_inventory(state: dict, entries: list[Entry], iteration: int = 999) -> list[dict]:
    """Describe current overview availability and non-duplicated update suggestions."""
    catalogue = _catalogue_from_state(state)
    records = state.setdefault("overviews", {})
    pending = set(state.setdefault("pending", []))
    failed = set(state.setdefault("failed", []))
    all_original = [entry for entry in entries if entry.type != "entity_overview"]
    original = [entry for entry in all_original if entry.status == "active"]
    by_id = {entry.id: entry for entry in all_original}
    inventory = []
    for entity_id in sorted(catalogue.variants):
        matched = catalogue.matched_entries(original, entity_id)
        matched_ids = [entry.id for entry, _ in matched]
        source_backed = [entry for entry, _ in matched if entry.source and entry.source.document]
        source_ids = [entry.id for entry in source_backed]
        source_documents = sorted({entry.source.document for entry in source_backed})
        record = records.get(entity_id, {})
        previous_ids = set(record.get("input_ids", []))
        new_count = len(set(source_ids) - previous_ids)
        previous_states = record.get("source_states", {})
        changed_ids = [entry.id for entry in source_backed
                       if entry.id in previous_states and previous_states[entry.id] != entry_state(entry)]
        inactive_ids = [entry_id for entry_id in previous_ids
                        if entry_id not in by_id or by_id[entry_id].status != "active"]
        interval_ready = iteration - record.get("last_success_iteration", iteration - 2) >= 2
        suggestion = ""
        if entity_id not in records and len(source_ids) >= INITIAL_OVERVIEW_CARDS:
            suggestion = "initial"
        elif entity_id in records and interval_ready and (new_count >= REFRESH_OVERVIEW_CARDS or changed_ids or inactive_ids):
            suggestion = "refresh"
        if entity_id in pending or entity_id in failed:
            suggestion = ""
        inventory.append({
            "entity_id": entity_id, "variants": catalogue.names_for(entity_id),
            "matched_card_ids": matched_ids, "overview_id": record.get("entry_id", ""),
            "source_card_ids": source_ids, "source_document_count": len(source_documents),
            "new_card_count": new_count, "changed_source_ids": changed_ids,
            "inactive_support_ids": inactive_ids, "suggestion": suggestion,
        })
    return inventory


def automatic_overview_tasks(inventory: list[dict], state: dict, iteration: int) -> list[dict]:
    """Drain initial coverage in small batches, then allow one routine update."""
    pending = state.setdefault("pending", [])
    queue = state.setdefault("initial_queue", [])
    eligible_initial = [item["entity_id"] for item in inventory if item["suggestion"] == "initial"]
    queue[:] = [entity_id for entity_id in queue if entity_id in eligible_initial]
    for entity_id in eligible_initial:
        if entity_id not in queue and entity_id not in pending:
            queue.append(entity_id)

    by_id = {item["entity_id"]: item for item in inventory}
    selected = queue[:INITIAL_OVERVIEW_BATCH_SIZE]
    if not selected:
        # After initial coverage, avoid spending more than one automatic call per iteration.
        selected = [item["entity_id"] for item in inventory if item["suggestion"] == "refresh"][:1]
    tasks = []
    covered = set()
    for entity_id in selected:
        item = by_id.get(entity_id)
        if not item or entity_id in pending or entity_id in covered:
            continue
        # Exact shared evidence is safe to batch; partial overlap stays separate.
        group_ids = [other_id for other_id in queue
                     if by_id.get(other_id, {}).get("source_card_ids") == item.get("source_card_ids")]
        group_ids = group_ids or [entity_id]
        covered.update(group_ids)
        for group_id in group_ids:
            if group_id not in pending:
                pending.append(group_id)
        variants = sorted({name for group_id in group_ids for name in by_id[group_id]["variants"]})
        tasks.append({
            "description": "Create or refresh the structured entity overview from matched evidence.",
            "expected_output_type": "entity_overview", "entity_overview_id": entity_id,
            "entity_overview_group_ids": group_ids, "entity_variants": variants,
            "reads_from_blackboard": [],
            "reads_from_documents": [], "priority": "high", "automatic_overview": True,
            "entity_refresh": item["suggestion"] == "refresh",
            "overview_queue_iteration": iteration,
        })
    return tasks


def attach_relevant_overviews(tasks: list[dict], entries: list[Entry], state: dict) -> list[dict]:
    """Attach at most two evidence-overlapping overview entries to substantive work."""
    active_overviews = {
        entry.id for entry in entries
        if entry.type == "entity_overview" and entry.status == "active"
    }
    records = state.get("overviews", {})
    for task in tasks:
        if task.get("expected_output_type") == "entity_overview":
            continue
        selected = set(task.get("reads_from_blackboard", []))
        if not selected:
            continue
        candidates = []
        for entity_id, record in records.items():
            overview_id = record.get("entry_id", "")
            overlap = selected & set(record.get("input_ids", []))
            if overview_id in active_overviews and len(overlap) >= 2:
                candidates.append((len(overlap), entity_id, overview_id))
        existing = list(dict.fromkeys(task.get("reads_from_blackboard", [])))
        attached = []
        for overlap, entity_id, overview_id in sorted(candidates, reverse=True)[:2]:
            if overview_id not in existing:
                existing.append(overview_id)
                attached.append({"overview_id": overview_id, "entity_id": entity_id, "overlap": overlap})
        if attached:
            task["reads_from_blackboard"] = existing
            task["entity_overview_attachments"] = attached
    return tasks


def defer_overview_dependent_tasks(tasks: list[dict], inventory: list[dict], state: dict,
                                   iteration: int) -> list[dict]:
    """Let readers proceed, but wait when analysis needs a pending missing overview."""
    pending = set(state.get("pending", []))
    by_id = {item["entity_id"]: set(item["matched_card_ids"]) for item in inventory}
    ready = []
    for task in tasks:
        if task.get("expected_output_type") not in {"analysis", "calculation", "strategy"}:
            ready.append(task)
            continue
        selected = set(task.get("reads_from_blackboard", []))
        blocking = next((entity_id for entity_id in pending
                         if len(selected & by_id.get(entity_id, set())) >= 2), "")
        if not blocking:
            ready.append(task)
            continue
        state.setdefault("deferred", []).append({
            "iteration": iteration, "entity_id": blocking,
            "description": task.get("description", "")[:200],
        })
    return ready


def prepare_overview_tasks(tasks: list[dict], inventory: list[dict], state: dict,
                           iteration: int) -> list[dict]:
    """Reject unscheduled overview work before it reaches a worker."""
    inventory_by_id = {item["entity_id"]: item for item in inventory}
    accepted = []
    for task in tasks:
        if task.get("expected_output_type") != "entity_overview":
            accepted.append(task)
            continue
        entity_id = task.get("entity_overview_id")
        item = inventory_by_id.get(entity_id)
        if entity_id in state.get("pending", []) and not task.get("automatic_overview"):
            continue
        if not item or (not item["suggestion"] and not item.get("matched_card_ids")):
            state.setdefault("rejected_requests", []).append({
                "iteration": iteration,
                "entity_id": entity_id,
                "reason": "missing or unavailable overview suggestion",
            })
            continue
        task["entity_variants"] = item["variants"]
        task["entity_refresh"] = item["suggestion"] == "refresh"
        if not item["suggestion"]:
            task["requested_overview"] = True
        pending = state.setdefault("pending", [])
        if entity_id not in pending:
            pending.append(entity_id)
        accepted.append(task)
    return accepted


def build_overview_prompt(entity_id: str, names: list[str], matched: list[tuple[Entry, str]],
                          previous_state: dict | None = None,
                          previous_input_ids: list[str] | None = None) -> str:
    cards = "\n\n".join(
        f"[{entry.id}] match={reason}\nsource={entry.source.document if entry.source else 'none'}\n{entry.content}"
        for entry, reason in matched
    )
    return f"""Produce a detailed, reusable overview of the target and candidate name variants.

These cards were retrieved broadly for relevance, not because they necessarily
describe one entity. They may concern one entity under several names, distinct
entities with similar names, or related entities such as shareholders,
subsidiaries, counterparties, and screening candidates.

Organize the evidence into distinct entity profiles where supported. Preserve
uncertainty where identity cannot be established. Explicitly distinguish
documented aliases, possible name variants, corporate relationships, and
screening matches. A screening match is not an alias or proof of identity.

Preserve original names, legal forms, jurisdictions, identifiers, and conflicting
values. Attribute each important fact to its supporting original cards. Do not
transfer attributes between profiles merely because names resemble each other.
Keep relevant evidence whose entity assignment remains uncertain in an
unresolved section.

Combine repetition, retain documented no-match or cleared outcomes separately from
transaction approval, and do not invent missing values. Keep a document claim, an
inference, and an unresolved question distinguishable.

TARGET RETRIEVAL ID: {entity_id}
CANDIDATE SPELLINGS: {', '.join(names)}

MATCHED ORIGINAL CARDS (all are supplied in full):
{cards}

{("REFRESH: Previous structured state follows. Keep still-supported statements, revise changed evidence, and never treat this state as independent evidence. Newly supplied evidence IDs: " + ", ".join(entry.id for entry, _ in matched if entry.id not in set(previous_input_ids or [])) + "\n" + render_structured_overview(previous_state) if previous_state else "")}

Return JSON with a \"findings\" array. Return exactly one finding with
\"type\": \"entity_overview\" and \"structured_overview\" containing these flexible
arrays: \"entity_profiles\", \"relationships_and_distinctions\", and
\"unresolved_or_conflicting_evidence\". Each array item has a \"label\" and a
\"statements\" array. Each statement has \"text\", \"supports_entries\" (original card
IDs), and optional \"kind\" (fact, derived, or unresolved). Empty sections are allowed.
Use a short readable \"content\" summary too. You may additionally return only
\"analysis\" or \"gap\" findings for new synthesis or actionable unanswered questions.
Each additional finding must include content and its original supporting card IDs in
supports_entries. Do not use other finding types. Do not repeat original findings."""


def validate_structured_overview(value: object, eligible_ids: set[str]) -> tuple[dict, list[str]]:
    """Keep valid evidence-linked statements; reject a response with none."""
    if not isinstance(value, dict):
        raise ValueError("entity overview response must include structured_overview")
    structured: dict[str, list[dict]] = {}
    referenced: list[str] = []
    for section in _OVERVIEW_SECTIONS:
        items = value.get(section, [])
        if not isinstance(items, list):
            continue
        valid_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            statements = item.get("statements", [])
            if not isinstance(statements, list):
                continue
            valid_statements = []
            for statement in statements:
                if not isinstance(statement, dict):
                    continue
                text = str(statement.get("text", "")).strip()
                supports = statement.get("supports_entries", [])
                if not text or not isinstance(supports, list):
                    continue
                supports = list(dict.fromkeys(
                    entry_id for entry_id in supports
                    if isinstance(entry_id, str) and entry_id in eligible_ids
                ))
                if not supports:
                    continue
                kind = str(statement.get("kind", "fact")).strip().lower()
                if kind not in {"fact", "derived", "unresolved"}:
                    kind = "fact"
                valid_statements.append({"text": text, "supports_entries": supports, "kind": kind})
                referenced.extend(supports)
            if valid_statements:
                valid_items.append({"label": label or section.replace("_", " "), "statements": valid_statements})
        structured[section] = valid_items
    if not referenced:
        raise ValueError("entity overview response has no valid source-backed statements")
    return structured, list(dict.fromkeys(referenced))


def render_structured_overview(structured: dict) -> str:
    """Render structured state directly, so workers receive every valid statement."""
    headings = {
        "entity_profiles": "Entity profiles",
        "relationships_and_distinctions": "Relationships and distinctions",
        "unresolved_or_conflicting_evidence": "Unresolved or conflicting evidence",
    }
    lines = []
    for section in _OVERVIEW_SECTIONS:
        items = structured.get(section, [])
        if not items:
            continue
        lines.append(headings[section] + ":")
        for item in items:
            lines.append(f"- {item['label']}")
            for statement in item["statements"]:
                refs = ", ".join(statement["supports_entries"])
                lines.append(f"  - [{statement['kind']}] {statement['text']} (supports: {refs})")
    return "\n".join(lines)


def run_entity_overview(catalogue: NameCatalogue, entries: list[Entry], entity_id: str,
                        caller: ModelCaller, iteration: int = 0,
                        previous_state: dict | None = None,
                        previous_input_ids: list[str] | None = None) -> tuple[list[Entry], list[str], int, dict]:
    """Build one overview from all matching active original cards."""
    matched = [(entry, reason) for entry, reason in catalogue.matched_entries(entries, entity_id)
               if entry.status == "active" and entry.type != "entity_overview"]
    if not matched:
        return [], [], 0, {}
    payload, tokens = call_model(
        caller,
        build_overview_prompt(entity_id, catalogue.names_for(entity_id), matched, previous_state, previous_input_ids),
    )
    findings = payload.get("findings", payload.get("value", []))
    if not isinstance(findings, list):
        findings = []
    overview_finding = next((
        finding for finding in findings
        if isinstance(finding, dict) and finding.get("type") == "entity_overview"
    ), None)
    eligible_ids = {entry.id for entry, _ in matched if entry.status == "active" and entry.type != "entity_overview"}
    structured, referenced_ids = validate_structured_overview(
        overview_finding.get("structured_overview") if overview_finding else None, eligible_ids,
    )
    # Structured output is sufficient; a prose summary must not be a hidden prerequisite.
    overview_finding["content"] = render_structured_overview(structured)
    overview_finding["supports_entries"] = referenced_ids
    allowed = [
        finding for finding in findings
        if isinstance(finding, dict)
        and finding.get("type") in ("entity_overview", "analysis", "gap")
    ]
    output = parse_worker_output(
        {"findings": allowed}, iteration, "entity_overview", "entity overview",
    )
    overviews = [entry for entry in output if entry.type == "entity_overview"]
    if len(overviews) != 1:
        raise ValueError("entity overview response must contain exactly one entity_overview finding")
    return output, [entry.id for entry, _ in matched], tokens, structured
