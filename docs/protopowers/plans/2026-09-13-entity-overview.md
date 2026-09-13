# Entity overview implementation plan

**Status:** Prepared from the approved conversational design; review before execution.
**Design:** [Entity overview design](../specs/2026-09-13-entity-overview-design.md).
**Goal / innovation:** Organize original swarm findings into reusable, detailed entity
overviews, with occasional new analysis/gap cards, and test downstream benefit.
**Base:** `0eb01d822cfea107299eb7004e6bae4932c79e34` in this clean project.
**Architecture:** Original swarm + Python name catalogue/matching + overview worker
and normal orchestrator signals. No V2 graph/query/synthesis machinery.
**Success:** Useful original-card coverage, full overview context actually supplied
to workers, meaningful follow-up findings, and complete reports with measurable
cost. First live checkpoint is Task 1, before scheduling and final-stage integration.

## Global constraints and reuse

Keep original cards and their schema, worker loop, direct reading, ordinary task
allocation and final report process. Only new overview behavior and its necessary
context/type filters change them. Overview text is detailed and never truncated
when assigned. All matched original input cards enter overview construction.
Use candidate-1 prompt from the design. Additional analysis/gap output is allowed
only for new synthesis/actionable gaps, with original-card references.

Reuse existing ModelCaller, Entry, signal generation, task queue and run artifacts.
Review and adapt only useful normalization/literal matching helpers from the older
duplicate-name project, adding AG. Do not copy typed profiles, specialist review,
identity merge infrastructure or tests enforcing those removed requirements.
Review local provider patches against current upstream; port only demonstrated
needed fixes, not whole adapters or earlier feature branches. Preserve provenance,
required output validation, artifact path protection and benchmark isolation.

Use no fixed name-discovery call quota or exactly-once state machine. Track pending
cards and overview coverage with small Python state. No new external dependency.
File/helper names below are provisional; binding behavior is in the design. Keep
important matching and scheduling logic explained by short code comments.
Do not commit automatically, use /tmp, or access unrelated folders. Any execution
requiring an external interpreter/dataset must use the authorized project setup.
No implementation or live invocation is authorized merely by saving this plan.

### Task 1: Demonstrate names-to-overview-to-worker context

**Outcome:** A finite card fixture produces a name catalogue, all matching cards,
a detailed overview plus optional ordinary findings, and full overview context
for an existing worker. This is the first runnable vertical slice.

**Relevant files:** `src/swarm/models.py`, `worker_dispatch.py`; small name-matching
and overview helpers in `src/swarm`; project-local fixture/runner only if needed.
Inspect existing Entry parsing/type handling before extending it.

**Interfaces:** Name discovery returns a string list to Python only. Python groups
observed spellings/forms under retrieval IDs and matches cards with reasons, not
identity proof. Overview uses the existing response/card/link contract. Preserve
original cards; record overview input IDs outside them. Reuse normal worker prompt
assembly with a complete-content exception for `entity_overview`.

**Minimum checks:** Names-only response parsing, AG/core/fuzzy matches, word boundaries,
unrelated same-name people, complete input and output delivery beyond 300 characters,
valid original-card references and an ordinary analysis/gap emitted beside an overview.
Do not write per-helper test matrices or import the older test suite.

**Live checkpoint:** Run the finite fixture through configured real callers for name
discovery, overview and downstream worker. Inspect saved inputs/outputs and usage;
show all matched-card IDs and whether variant ambiguity is preserved. Prompt comparisons
are deferred. Present results for feedback before secondary integration. Resolve
provider configuration, finite fixture scope and external access before launch.

### Task 2: Integrate iteration matching and overview prerequisites

**Outcome:** Normal rounds discover names only when direct workers produced new
direct findings; Python evaluates overview suggestions after every iteration.
Substantive entity assignments get an overview first without blocking unrelated work.

**Relevant files:** `src/swarm/__init__.py`, `orchestrator.py`, `worker_dispatch.py`,
`blackboard.py`, existing signal helpers, small catalogue state serialization.

**Interfaces:** Track pending original cards. Initial reading qualifies. Match all
known names against new cards every iteration; match newly discovered names against
old cards too. Keep pending discovery visible for derived-only tails. Python exposes
available overview IDs, new-card counts and suggestions in the normal orchestrator
prompt. Choose conservative initial/update thresholds using Task 1 observations and
user feedback; no unapproved numeric threshold is assumed.

Deduplicate pending overview signals/tasks. Create before dependent reasoning work;
allow direct readers/unrelated work to run. Orchestrator chooses refresh versus full
old overview plus new cards. Exclude overviews from their own name discovery/input
and trigger counts. Reuse existing task batches instead of a dependency framework.
Persist enough state for resumed execution without requiring strict network exactly-once.

**Minimum checks:** Direct-read/new-findings trigger versus derived-only iteration,
all-name matching on new cards, new-name matching on old cards, missing/current/update
states, no duplicate suggestions or recursive refresh, and unrelated work proceeds.
A failed overview is visible and cannot create an automatic endless prerequisite loop.
Test relevant upstream card status/content changes without adding a revision engine.

**Demonstration:** An integration fixture shows an overview being created, selected
by ID and delivered in full; a later matching card produces an update suggestion.

### Task 3: Preserve original final selection while carrying new findings through

**Outcome:** Overview cards guide workers but are not automatically selected for the
report. Additional analysis/gap cards follow the original final-report path.

**Relevant files:** `src/swarm/curation.py`, `obligations.py`, `synthesis_packet.py`,
`synthesis.py`, `__init__.py`; source/quality/verification consumers only where the
new type requires handling.

**Interfaces:** Filter `entity_overview` from automatic curation, obligation and
final-synthesis evidence pools, including fallback/safety-net paths. Keep all ordinary
card selection unchanged. Preserve supporting references of additional findings;
overviews are not independent corroboration. Do not simply classify them as normal
analysis to get them through old filters.

**Minimum checks:** A long overview is fully delivered upstream to a worker but absent
from final selection; a new supported analysis and actionable gap from the same
response survive appropriate ordinary paths. An ordinary-card-only fixture retains
baseline selection. Exercise actual report assembly with fake providers and required
output checks. Do not redesign must_include selection or repair unrelated upstream
heuristics as part of this task.

### Task 4: Run the integrated experiment and assess reuse

**Outcome:** A fresh-label completed run demonstrates useful overview guidance and
truthful accounting. Compare with the same pinned upstream and provider configuration.

**Relevant files:** Existing runner/metrics hooks as needed, project-local results,
and a short findings report in `docs/protopowers/`.

Run focused changed-path checks, plus an integration smoke check. Before paid work,
resolve concrete run/model/token scope, dataset access and any judging authorization.
Do not transplant historical metrics or claim the August run is a controlled baseline.
Inspect unmatched cards and name coverage, full context delivery, ambiguous names,
new overview-derived findings and final-report completeness. Count discovery/overview
calls and tokens separately from ordinary workers, and identify actual overview
recipients and repeated refresh costs. Supplied context alone does not prove use.

Record remaining upstream selection omissions separately. If overviews reveal useful
facts that final selection loses, propose a later overview-assisted obligation test;
do not silently add it. The two alternative overview prompts and larger corpora also
remain later, separately authorized comparisons.

## Execution handoff

Recommend inline execution with `protopowers:executing-plans`: these changes share
worker/context and card-selection interfaces. Subagent-driven execution remains an
alternative after selecting exact role/model profiles. Written design and plan review
comes before implementation; live checkpoint feedback comes before secondary work.
