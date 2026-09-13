"""Small, run-local entity catalogue and overview worker helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .models import Entry, ModelCaller
from .worker_dispatch import call_model, parse_worker_output


_LEGAL_FORMS = {"ag", "gmbh", "inc", "incorporated", "ltd", "llc", "lp", "plc", "sa"}
INITIAL_OVERVIEW_CARDS = 3
REFRESH_OVERVIEW_CARDS = 2


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


def overview_inventory(state: dict, entries: list[Entry]) -> list[dict]:
    """Describe current overview availability and non-duplicated update suggestions."""
    catalogue = _catalogue_from_state(state)
    records = state.setdefault("overviews", {})
    pending = set(state.setdefault("pending", []))
    failed = set(state.setdefault("failed", []))
    original = [entry for entry in entries if entry.status == "active" and entry.type != "entity_overview"]
    inventory = []
    for entity_id in sorted(catalogue.variants):
        matched_ids = [entry.id for entry, _ in catalogue.matched_entries(original, entity_id)]
        record = records.get(entity_id, {})
        previous_ids = set(record.get("input_ids", []))
        new_count = len(set(matched_ids) - previous_ids)
        suggestion = ""
        if entity_id not in records and len(matched_ids) >= INITIAL_OVERVIEW_CARDS:
            suggestion = "initial"
        elif entity_id in records and new_count >= REFRESH_OVERVIEW_CARDS:
            suggestion = "refresh"
        if entity_id in pending or entity_id in failed:
            suggestion = ""
        inventory.append({
            "entity_id": entity_id, "variants": catalogue.names_for(entity_id),
            "matched_card_ids": matched_ids, "overview_id": record.get("entry_id", ""),
            "new_card_count": new_count, "suggestion": suggestion,
        })
    return inventory


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
        if not item or not item["suggestion"]:
            state.setdefault("rejected_requests", []).append({
                "iteration": iteration,
                "entity_id": entity_id,
                "reason": "missing or unavailable overview suggestion",
            })
            continue
        task["entity_variants"] = item["variants"]
        pending = state.setdefault("pending", [])
        if entity_id not in pending:
            pending.append(entity_id)
        accepted.append(task)
    return accepted


def build_overview_prompt(entity_id: str, names: list[str], matched: list[tuple[Entry, str]]) -> str:
    cards = "\n\n".join(
        f"[{entry.id}] match={reason}\nsource={entry.source.document if entry.source else 'none'}\n{entry.content}"
        for entry, reason in matched
    )
    return f"""Produce a detailed, reusable overview of the target and candidate name variants.
Organize identity and naming, activities and roles, relationships and transactions,
relevant dates/amounts/identifiers, contradictions, and open questions. Preserve
specific details rather than broad summaries. Combine repetition while retaining
supporting card references. Include potentially useful findings beyond the immediate
question. Cards were retrieved using broad name matching: similar or identical names
may refer to different entities, and different spellings may refer to one entity.
Do not assume identity from retrieval. Explain supported associations, distinctions
and uncertainty, citing original cards.

TARGET RETRIEVAL ID: {entity_id}
CANDIDATE SPELLINGS: {', '.join(names)}

MATCHED ORIGINAL CARDS (all are supplied in full):
{cards}

Return JSON with a \"findings\" array. Return exactly one finding with
\"type\": \"entity_overview\" and a detailed freestyle \"content\" string containing
the dossier; use supports_entries for its original card IDs. You may additionally
return only \"analysis\" or \"gap\" findings for new synthesis or actionable unanswered
questions. Each additional finding must include content and its original supporting
card IDs in supports_entries. Do not use other finding types or rigid nested schemas.
Do not repeat original findings."""


def run_entity_overview(catalogue: NameCatalogue, entries: list[Entry], entity_id: str,
                        caller: ModelCaller, iteration: int = 0) -> tuple[list[Entry], list[str], int]:
    """Build one overview from all matching active original cards."""
    matched = catalogue.matched_entries(entries, entity_id)
    if not matched:
        return [], [], 0
    payload, tokens = call_model(
        caller,
        build_overview_prompt(entity_id, catalogue.names_for(entity_id), matched),
    )
    findings = payload.get("findings", payload.get("value", []))
    if not isinstance(findings, list):
        findings = []
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
    return output, [entry.id for entry, _ in matched], tokens
