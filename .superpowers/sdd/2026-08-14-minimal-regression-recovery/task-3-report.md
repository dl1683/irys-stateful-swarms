# Task 3 report: decision-only final synthesis injection

## Scope delivered

- Moved final entity maintenance to immediately after `consolidate_items(...)`, so curation completes before confirmed decisions are projected.
- Removed the predecessor generic entity-maintenance packet path. Active `duplicate_name_resolution` cards that are not already represented in `must_include` are appended one at a time.
- Each injected item preserves the card's JSON `content` as its summary and has `section="Entity resolution"`, `importance="critical"`, and `source="entity_maintenance"`.
- No `entity_information` cards, JSON parsing, profile injection, new helper, or resolver changes were added.

## TDD evidence

1. Replaced the predecessor generic-card test with `test_final_duplicate_decisions_are_explicit_synthesis_items_after_curation`.
2. Red run before the production change:

   ```text
   AssertionError: assert 2 < 1
   events: ['periodic', 'final', 'curate']
   ```

   This demonstrated the old final-maintenance-before-curation order.
3. Green run after the production change:

   ```text
   1 passed, 5 deselected in 0.07s
   ```

## Verification

```text
/home/gvw/Desktop/AI_coding/legal_AI/iryst_stateful_swarm_duplicate_name_resolution_venv/bin/python -m pytest -q tests/test_entity_maintenance.py tests/test_entity_candidate_scoring.py tests/test_entity_profiles.py
19 passed in 0.11s
```

Deferred removal scan:

```text
rg -n --glob '*.py' 'entity_information|project_entity_information_cards' .
exit code 1; no matches

rg -n 'entity_information|project_entity_information_cards' src
exit code 1; no matches
```

An unfiltered text scan of the full checkout still finds expected historical references in planning documents and recorded `judge_comparisons` artifacts; no live Python or `src` references remain.

## Scope and worktree hygiene

- Only `src/swarm/__init__.py`, `tests/test_entity_maintenance.py`, and this report are Task 3 changes.
- Existing unrelated modifications remain unstaged and untouched.
