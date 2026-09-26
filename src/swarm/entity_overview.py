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


def _is_qualifying_source(entry: Entry, state: dict) -> bool:
    """Count direct document findings, not later analysis that inherited a source."""
    if not entry.source or not entry.source.document:
        return False
    # Initial extraction observations are direct by construction. Other finding
    # types qualify only when their worker actually read a document section.
    return entry.type == "observation" or entry.id in state.get("direct_source_entry_ids", [])


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


def overview_inventory(state: dict, entries: list[Entry], iteration: int = 999,
                       registered_docs: set[str] | None = None) -> list[dict]:
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
        source_backed = [entry for entry, _ in matched if _is_qualifying_source(entry, state)
                         and (registered_docs is None or entry.source.document in registered_docs)]
        source_ids = [entry.id for entry in source_backed]
        source_documents = sorted({entry.source.document for entry in source_backed})
        record = records.get(entity_id, {})
        previous_ids = set(record.get("input_ids", []))
        if record.get("cards"):
            new_count = max(len(set(source_ids) - set(card.get("semantic_input_ids", previous_ids)))
                            for card in record["cards"])
            prior_states = {i: old for card in record["cards"]
                            for i, old in card.get("support_states", {}).items()}
        else:
            new_count = len(set(source_ids) - previous_ids)
            prior_states = record.get("source_states", {})
        changed_ids = [entry.id for entry in source_backed
                       if entry.id in prior_states and prior_states[entry.id] != entry_state(entry)]
        inactive_ids = [entry_id for entry_id in prior_states
                        if entry_id not in by_id or by_id[entry_id].status != "active"]
        interval_ready = iteration - record.get("last_success_iteration", iteration - 2) >= 2
        eligibility = ""
        if entity_id not in records and len(source_ids) >= INITIAL_OVERVIEW_CARDS:
            eligibility = "initial"
        elif entity_id in records and interval_ready and (new_count >= REFRESH_OVERVIEW_CARDS or changed_ids or inactive_ids):
            eligibility = "refresh"
        suggestion = eligibility
        if entity_id in pending or entity_id in failed:
            suggestion = ""
        inventory.append({
            "entity_id": entity_id, "variants": catalogue.names_for(entity_id),
            "matched_card_ids": matched_ids, "overview_id": record.get("entry_id", ""),
            "source_card_ids": source_ids, "source_document_count": len(source_documents),
            "new_card_count": new_count, "changed_source_ids": changed_ids,
            "new_worker_count": len({
                entry.created_by.worker_id for entry in source_backed
                if entry.id not in previous_ids and entry.created_by.worker_id
            }),
            "inactive_support_ids": inactive_ids,
            "last_success_iteration": record.get("last_success_iteration", -1),
            "eligibility": eligibility, "suggestion": suggestion,
        })
    return inventory


def automatic_overview_tasks(inventory: list[dict], state: dict, iteration: int) -> list[dict]:
    """Drain initial coverage after ordinary reading; deltas have a separate cadence."""
    pending = state.setdefault("pending", [])
    queue = state.setdefault("initial_queue", [])
    eligible_initial = [item["entity_id"] for item in inventory if item["suggestion"] == "initial"]
    queue[:] = [entity_id for entity_id in queue if entity_id in eligible_initial]
    for entity_id in eligible_initial:
        if entity_id not in queue and entity_id not in pending:
            queue.append(entity_id)

    by_id = {item["entity_id"]: item for item in inventory}
    selected = queue[:INITIAL_OVERVIEW_BATCH_SIZE]
    selection_pool = list(queue)
    reasons = {entity_id: "initial coverage" for entity_id in selected}
    tasks = []
    covered = set()
    for entity_id in selected:
        item = by_id.get(entity_id)
        if not item or entity_id in pending or entity_id in covered:
            continue
        # Exact shared evidence is safe to batch; partial overlap stays separate.
        group_ids = [other_id for other_id in selection_pool if (
            item.get("source_card_ids")
            and by_id.get(other_id, {}).get("source_card_ids") == item.get("source_card_ids")
            or (
                item.get("overview_id")
                and by_id.get(other_id, {}).get("overview_id") == item.get("overview_id")
            )
        )]
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
            "entity_source_ids": item["source_card_ids"],
            "reads_from_blackboard": [],
            "reads_from_documents": [], "priority": "high", "automatic_overview": True,
            "entity_refresh": item["suggestion"] == "refresh",
            "overview_selection_reason": reasons[entity_id],
            "overview_queue_iteration": iteration,
        })
        state.setdefault("selection_log", []).append({
            "iteration": iteration, "entity_id": entity_id,
            "group_ids": group_ids, "reason": reasons[entity_id],
            "new_card_count": item.get("new_card_count", 0),
            "new_worker_count": item.get("new_worker_count", 0),
        })
    return tasks


def resolve_overview_request(requested: object, inventory: list[dict], state: dict) -> tuple[str, str]:
    """Resolve a retrieval key, current/old Entry ID, or unambiguous known name."""
    value = str(requested or "").strip()
    if not value:
        return "", "missing or unavailable overview suggestion"
    by_id = {item["entity_id"]: item for item in inventory}
    if value in by_id:
        return value, ""

    id_matches = []
    for entity_id, record in state.get("overviews", {}).items():
        known_ids = {record.get("entry_id", ""), *record.get("superseded_entry_ids", [])}
        if value in known_ids:
            id_matches.append(entity_id)
    if len(id_matches) == 1:
        return id_matches[0], ""

    normalized = normalize_name(value)
    name_matches = [
        item["entity_id"] for item in inventory
        if normalized and any(normalize_name(name) == normalized for name in item.get("variants", []))
    ]
    matches = list(dict.fromkeys(id_matches or name_matches))
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return "", "ambiguous overview identifier"
    return "", "unknown overview identifier"


def attach_relevant_overviews(tasks: list[dict], entries: list[Entry], state: dict) -> list[dict]:
    """Attach at most two evidence-overlapping overview entries to substantive work."""
    by_id = {entry.id: entry for entry in entries}
    active_overviews = {
        entry.id for entry in entries
        if entry.type == "entity_overview" and entry.status == "active"
    }
    records = state.get("overviews", {})
    stale_ids = {
        card_id for record in records.values() for card_id in record.get("card_ids", [record.get("entry_id", "")])
        if any(
            entry_id not in by_id or by_id[entry_id].status != "active"
            or (entry_id in record.get("source_states", {})
                and record["source_states"][entry_id] != entry_state(by_id[entry_id]))
            for entry_id in record.get("input_ids", [])
        )
    }
    for task in tasks:
        if task.get("expected_output_type") == "entity_overview":
            continue
        existing = list(dict.fromkeys(entry_id for entry_id in task.get("reads_from_blackboard", [])
                                      if entry_id not in stale_ids))
        task["reads_from_blackboard"] = existing
        selected = set(existing)
        if not selected:
            continue
        candidates = []
        for entity_id, record in records.items():
            for card in record.get("cards", []):
                overview_id = card["id"]
                overlap = selected & set(card["identity_source_ids"])
                if overview_id in active_overviews and overview_id not in stale_ids and overlap:
                    candidates.append((len(overlap), entity_id, overview_id))
            if not record.get("cards"):
                overview_id = record.get("entry_id", "")
                overlap = selected & set(record.get("input_ids", []))
                if overview_id in active_overviews and overview_id not in stale_ids and len(overlap) >= 2:
                    candidates.append((len(overlap), entity_id, overview_id))
        attached = [
            {"overview_id": card["id"], "entity_id": entity_id, "overlap": 0}
            for entity_id, record in records.items() for card in record.get("cards", [])
            if card["id"] in existing
        ]
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
    """Only Python maintenance may create overview work."""
    accepted = []
    for task in tasks:
        if task.get("expected_output_type") != "entity_overview":
            accepted.append(task)
            continue
        state.setdefault("rejected_requests", []).append({
            "iteration": iteration, "entity_id": task.get("entity_overview_id"),
            "reason": "overview jobs are scheduled only by Python maintenance",
        })
    return accepted


def build_overview_prompt(entity_id: str, names: list[str], matched: list[tuple[Entry, str]],
                          previous_state: dict | None = None,
                          previous_input_ids: list[str] | None = None) -> str:
    cards = "\n\n".join(
        f"[{entry.id}] match={reason}\nsource={entry.source.document if entry.source else 'none'}\n{entry.content}"
        for entry, reason in matched
    )
    return f"""Create compact navigation cards from these original findings.
The retrieval group may contain different people or companies with similar names.
Return one card per supported identity. Leave uncertain sources as candidate_source_ids;
never assign their facts to an identity. Keep original legal forms and spellings.
Every displayed name and fact must cite original card IDs. Do not infer identity
from name similarity, shared address, or a screening hit. Do not create analysis findings.

RETRIEVAL ID: {entity_id}
CANDIDATE SPELLINGS: {', '.join(names)}
ORIGINAL CARDS:
{cards}

Return JSON only: {{"cards": [{{"name": "display name",
"identity_source_ids": ["original ID"], "candidate_source_ids": ["original ID"],
"facts": [{{"text": "short material fact", "source_ids": ["original ID"]}}],
"identity_clues": [{{"type": "birth_date|identifier|address|relationship|other",
"value": "exact source value", "source_ids": ["original ID"]}}]}}]}}.
Use an empty cards array if no identity is supported."""


def validate_pointer_cards(value: object, eligible_ids: set[str]) -> list[dict]:
    """Reject invented references and facts assigned to uncertain candidates."""
    if not isinstance(value, list):
        raise ValueError("overview response must contain cards array")
    cards = []
    for raw in value:
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            raise ValueError("overview card needs a display name")
        confirmed = raw.get("identity_source_ids", [])
        candidates = raw.get("candidate_source_ids", [])
        if (not isinstance(confirmed, list) or not confirmed
                or not isinstance(candidates, list)
                or any(not isinstance(i, str) or i not in eligible_ids for i in confirmed + candidates)):
            raise ValueError("overview card has invalid original pointers")
        confirmed = list(dict.fromkeys(confirmed))
        candidates = [i for i in dict.fromkeys(candidates) if i not in confirmed]
        facts = []
        if not isinstance(raw.get("facts", []), list) or not isinstance(raw.get("identity_clues", []), list):
            raise ValueError("overview facts and identity clues must be arrays")
        for fact in raw.get("facts", []):
            if not isinstance(fact, dict) or not str(fact.get("text", "")).strip():
                raise ValueError("overview fact needs text")
            refs = fact.get("source_ids", [])
            if not isinstance(refs, list) or not refs or any(not isinstance(i, str) or i not in confirmed for i in refs):
                raise ValueError("overview fact needs confirmed original support")
            facts.append({"text": str(fact["text"]).strip(), "source_ids": list(dict.fromkeys(refs))})
        clues = []
        for clue in raw.get("identity_clues", []):
            if not isinstance(clue, dict) or not str(clue.get("type", "")).strip() or not str(clue.get("value", "")).strip():
                raise ValueError("identity clue needs type and exact value")
            refs = clue.get("source_ids", [])
            if not isinstance(refs, list) or not refs or any(not isinstance(i, str) or i not in confirmed for i in refs):
                raise ValueError("identity clue needs confirmed original support")
            clues.append({"type": str(clue["type"]).strip(), "value": str(clue["value"]).strip(),
                          "source_ids": list(dict.fromkeys(refs))})
        cards.append({"name": str(raw["name"]).strip(), "identity_source_ids": confirmed,
                      "candidate_source_ids": candidates, "facts": facts, "identity_clues": clues})
    if not cards:
        raise ValueError("overview response has no source-backed identity cards")
    return cards


def render_pointer_card(card: dict) -> str:
    """Keep worker-visible context tied to current original-card references."""
    lines = [f"{card['name']} (identity: {', '.join(card['identity_source_ids'])})"]
    lines.extend(f"- {'[unresolved] ' if fact.get('kind') == 'unresolved' else ''}"
                 f"{fact['text']} (supports: {', '.join(fact['source_ids'])})"
                 for fact in card["facts"])
    if card["candidate_source_ids"]:
        lines.append("Unassigned candidate pointers: " + ", ".join(card["candidate_source_ids"]))
    return "\n".join(lines)


def sync_overview_references(state: dict, entries: list[Entry],
                             registered_docs: set[str], iteration: int) -> None:
    """Update pointer membership and hide facts whose original support is gone."""
    catalogue = _catalogue_from_state(state)
    by_id = {entry.id: entry for entry in entries}
    seen_cards: dict[str, dict] = {}
    for group_id, record in state.get("overviews", {}).items():
        current = [entry.id for entry, _ in catalogue.matched_entries(entries, group_id)
                   if entry.status == "active" and entry.source
                   and entry.source.document in registered_docs
                   and _is_qualifying_source(entry, state)]
        record["pointer_ids"] = current
        for index, card in enumerate(record.get("cards", [])):
            if card["id"] in seen_cards:
                record["cards"][index] = seen_cards[card["id"]]
                continue
            seen_cards[card["id"]] = card
            known = set(card["identity_source_ids"]) | set(card["candidate_source_ids"])
            # New retrieved cards stay candidates until a delta cites them for this identity.
            card["candidate_source_ids"] = [i for i in dict.fromkeys(
                card["candidate_source_ids"] + [i for i in current if i not in known])
                if i in current and i not in card["identity_source_ids"]]
            support_states = card.setdefault("support_states", {
                i: record.get("source_states", {}).get(i, "") for i in card["identity_source_ids"]
            })
            valid = {i for i in card["identity_source_ids"]
                     if i in current and (not support_states.get(i)
                                          or support_states[i] == entry_state(by_id[i]))}
            card["identity_source_ids"] = [i for i in card["identity_source_ids"] if i in valid]
            card["candidate_source_ids"] = [i for i in card["candidate_source_ids"] if i not in valid]
            retained = []
            for fact in card["facts"]:
                old = fact["source_ids"]
                fact["source_ids"] = [i for i in old if i in valid]
                if len(old) != len(fact["source_ids"]):
                    hint = next((i for i in current if i not in valid
                                 and fact["text"].casefold() in by_id[i].content.casefold()), "")
                    if not hint:
                        hint = next((candidate for candidate in current if candidate not in valid
                                     and any(by_id.get(lost) and by_id[lost].source
                                             and by_id[candidate].source.document == by_id[lost].source.document
                                             and by_id[candidate].source.section == by_id[lost].source.section
                                             for lost in old)), "")
                    key = (card["id"], fact["text"], tuple(old))
                    if key not in {(e["key"][0], e["key"][1], tuple(e["key"][2]))
                                   for e in state.setdefault("reference_errors", [])}:
                        state["reference_errors"].append({
                            "key": list(key), "iteration": iteration, "entity_id": card["id"],
                            "job_kind": "reference_sync", "affected_card_ids": list(set(old) - valid),
                            "reason": "displayed fact lost original support", "candidate_id": hint,
                        })
                    card.setdefault("repair_hints", []).append({
                        "text": fact["text"], "lost_ids": list(set(old) - valid),
                        "candidate_id": hint,
                    })
                if fact["source_ids"]:
                    retained.append(fact)
            card["facts"] = retained
            entry = by_id.get(card["id"])
            if entry:
                entry.status = "active" if card["identity_source_ids"] else "inactive"
                entry.supports_entries = card["identity_source_ids"][:]
                entry.content = render_pointer_card(card) if card["identity_source_ids"] else ""


def build_delta_prompt(card: dict, source_entries: list[Entry], hints: list[dict]) -> str:
    originals = "\n".join(f"[{entry.id}] {entry.content}" for entry in source_entries)
    return f"""Maintain exactly this one identity navigation card. Do not merge similar names.
Return small supported operations only, not a rewritten card or analysis finding.
An unassigned candidate is not a confirmed identity source until a cited operation
supports its assignment. A repair candidate is only a hint, never proven support.

CURRENT CARD: {render_pointer_card(card)}
CURRENT FACTS (zero-based indices): {[(i, fact) for i, fact in enumerate(card['facts'])]}
REPAIR / WORKER HINTS: {hints}
NEW OR CHANGED ORIGINAL CARDS:
{originals}

Return JSON only: {{"operations": [
{{"op": "add|replace|remove|flag", "fact_index": 0,
"text": "short fact for add/replace/flag", "source_ids": ["original ID"]}}
]}}. Use fact_index only for replace/remove. Each source ID must be in the
supplied originals or current card's confirmed pointers. Return an empty list
if evidence does not justify a change."""


def apply_delta(card: dict, payload: dict, allowed_ids: set[str],
                active_entries: dict[str, Entry]) -> dict:
    """Validate the whole patch before changing supported state."""
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("delta must contain operations array")
    revised = {**card, "facts": [dict(fact) for fact in card["facts"]],
               "identity_source_ids": card["identity_source_ids"][:],
               "candidate_source_ids": card["candidate_source_ids"][:],
               "repair_hints": []}
    for op in operations:
        if not isinstance(op, dict) or op.get("op") not in {"add", "replace", "remove", "flag"}:
            raise ValueError("unsupported delta operation")
        kind = op["op"]
        refs = op.get("source_ids", [])
        if (not isinstance(refs, list) or not refs
                or any(not isinstance(i, str) or i not in allowed_ids or i not in active_entries
                       for i in refs)):
            raise ValueError("delta operation has unsupported original references")
        refs = list(dict.fromkeys(refs))
        if kind in {"replace", "remove"}:
            index = op.get("fact_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(revised["facts"]):
                raise ValueError("delta fact index is invalid")
        if kind == "remove":
            revised["facts"].pop(index)
            continue
        fact_text = op.get("text")
        if not isinstance(fact_text, str) or not fact_text.strip():
            raise ValueError("delta fact needs text")
        fact = {"text": fact_text.strip(), "source_ids": refs}
        if kind == "flag":
            fact["kind"] = "unresolved"
        if kind == "replace":
            revised["facts"][index] = fact
        else:
            revised["facts"].append(fact)
        # A cited new card becomes confirmed only for this one identity.
        for entry_id in refs:
            if entry_id not in revised["identity_source_ids"]:
                revised["identity_source_ids"].append(entry_id)
            if entry_id in revised["candidate_source_ids"]:
                revised["candidate_source_ids"].remove(entry_id)
    revised["support_states"] = {
        i: entry_state(active_entries[i]) for i in revised["identity_source_ids"]
        if i in active_entries
    }
    return revised


def select_delta_tasks(state: dict, inventory: list[dict], entries: list[Entry]) -> list[dict]:
    """Choose at most three precise jobs; leave the rest visible as backlog."""
    by_id = {entry.id: entry for entry in entries if entry.status == "active"}
    candidates = []
    seen_cards = set()
    for item in inventory:
        group_id = item["entity_id"]
        record = state.get("overviews", {}).get(group_id, {})
        for card in record.get("cards", []):
            if card["id"] in seen_cards:
                continue
            seen_cards.add(card["id"])
            cursor = set(card.get("semantic_input_ids", record.get("input_ids", [])))
            new_ids = [i for i in item["source_card_ids"] if i not in cursor]
            changed = [i for i, old in card.get("support_states", {}).items()
                       if i not in by_id or old != entry_state(by_id[i])]
            requests = [r for r in state.get("overview_review_requests", [])
                        if r["overview_id"] == card["id"]]
            hints = card.get("repair_hints", []) + requests
            if len(new_ids) < REFRESH_OVERVIEW_CARDS and not changed and not hints:
                continue
            signature = sorted(new_ids + changed + [str(h) for h in hints])
            if signature == card.get("failed_delta_signature"):
                continue
            source_ids = list(dict.fromkeys(new_ids + changed + card["identity_source_ids"][:2]
                                            + [h.get("candidate_id", "") for h in hints]))
            source_ids = [i for i in source_ids if i in by_id]
            candidates.append((not bool(hints or changed), -len(new_ids), group_id,
                               card["id"], {
                "expected_output_type": "entity_delta", "entity_group_id": group_id,
                "entity_card_id": card["id"], "entity_delta_source_ids": source_ids,
                "entity_new_ids": new_ids, "entity_hints": hints,
                "entity_signature": signature,
            }))
    candidates.sort(key=lambda item: item[:4])
    state["delta_backlog"] = [item[3] for item in candidates[3:]]
    return [item[4] for item in candidates[:3]]


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
                        previous_input_ids: list[str] | None = None,
                        eligible_source_ids: set[str] | None = None) -> tuple[list[Entry], list[str], int, list[dict]]:
    """Build separate source-linked cards from one broad retrieval group."""
    matched = [(entry, reason) for entry, reason in catalogue.matched_entries(entries, entity_id)
               if entry.status == "active" and entry.type != "entity_overview"
               and (eligible_source_ids is None or entry.id in eligible_source_ids)]
    if not matched:
        return [], [], 0, {}
    payload, tokens = call_model(
        caller,
        build_overview_prompt(entity_id, catalogue.names_for(entity_id), matched, previous_state, previous_input_ids),
    )
    eligible_ids = {entry.id for entry, _ in matched if entry.status == "active" and entry.type != "entity_overview"}
    cards = validate_pointer_cards(payload.get("cards"), eligible_ids)
    findings = []
    for card in cards:
        findings.append({"type": "entity_overview", "content": render_pointer_card(card),
                         "supports_entries": card["identity_source_ids"]})
    output = parse_worker_output({"findings": findings}, iteration, "entity_overview", "entity overview")
    return output, [entry.id for entry, _ in matched], tokens, cards
