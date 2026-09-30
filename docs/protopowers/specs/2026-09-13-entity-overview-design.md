# Entity overviews for swarm context management

**Status:** Design direction approved in conversation; written specification for
review. No implementation or live model calls performed.
**Baseline:** Upstream dl1683/irys-stateful-swarms, commit
`0eb01d822cfea107299eb7004e6bae4932c79e34`, cloned into this clean project.
**Plan:** [Implementation plan](../plans/2026-09-13-entity-overview.md).

## Hypothesis and scope

A few detailed, reusable entity overviews can help ordinary swarm workers combine
findings and avoid repeated searching. Test this while preserving upstream task
allocation, document reading, card schema, curation and synthesis as far as possible.
The user is investigating document collections, initially the existing small
sanctions example. Completion alone is not evidence of cost or quality improvement.

The architecture is: ordinary blackboard cards → batched name discovery → Python
name catalogue and matching → orchestrator-requested entity overview → downstream
workers. No V2 canonical store, typed knowledge graph, identity-resolution engine,
worker graph-query loop, local model or external retrieval service is required.

This design supersedes the earlier name-card navigation/V2 retention proposals
for this project. Historical projects remain references, not implementation bases.
No broad code imports, historical-run migration, automatic commits or cleanup of
those projects is included.

## Name discovery and matching

Name-list workers return names and explicitly source-supported alias relationships
to Python, never blackboard cards. Each alias includes the full name, short form,
and supporting card ID. This uses the existing discovery call. Python checks that
both names occur in the cited input card, combines their retrieval groups while
preserving an existing overview, and remembers the spellings for later discovery.
The worker must establish the explicit relationship; co-occurrence alone is not
permission to report an alias. Conflicting full-name targets for a short form are
not merged. This remains retrieval grouping, not general identity resolution.
Run them after initial direct reading and after iterations in which direct-document
workers actually produced new direct findings. Batch pending original blackboard
cards within practical model size limits; do not resend the whole blackboard by
default. Simple processed-card bookkeeping is enough. At-least-once coverage is
more important than strict exactly-once processing. There is no indexing-call cap
or per-card validation/repair state machine. Basic malformed-list/provider handling
must preserve pending work and never confuse a failed request with an empty list.

After every iteration, search all existing and newly discovered names against new
cards. Newly discovered names are also searched against earlier cards. A derived-only
iteration can match known names but cannot discover unknown names until another
eligible name-list pass; report pending discovery at completion. Name-free cards
remain ordinary findings available through the original workflow.

Python keeps name variants, original spellings and normalized forms in a small
run-local catalogue. A stable internal ID groups variants for overview retrieval,
not as proof of a single real-world identity. Duplicate names are deduplicated.
Use aggressive candidate matching: Unicode/case/spacing/punctuation normalization,
legal-form removal including AG, and useful token/close-spelling comparisons.
Preserve the spelling and match reason. All variants/forms participate in search.
Do not apply transitive fuzzy matching that silently expands a group through an
unbounded chain of weak matches. Similar-name matches remain explicit candidates.

Review helpers in the duplicate-name project before adapting selected code. Its
normalization and literal matching are useful references; do not import its typed
profiles, pairwise specialist review, merge decisions or full test suite. Aggressive
normalization is acceptable for discovery, not automatic identity assertions.

Original card fields and content remain unchanged. The catalogue, matched-card IDs,
coverage and overview input sets live outside cards in minimal run-local state.
No entity index or entity metadata is attached to original cards. Preserve the
upstream card lifecycle rather than imposing the former V2 immutability contract.
If upstream changes/deactivates card content, matching and overview coverage must
reflect that using simple existing lifecycle information, not a new revision engine.

## Overview requests and scheduling

Python evaluates overview suggestions after every iteration, regardless of whether
name discovery ran. Suggest initial creation after a high distinct-card threshold,
and updates after a substantial number of new matching cards since the prior
overview. Threshold values remain to be chosen from initial observed card counts;
no specific number is approved. Avoid duplicate suggestions while one is pending.
Overview cards themselves must not drive discovery/refresh counts or become their
own evidence. Existing derived analysis can be included but remains labelled derived.

The orchestrator receives a compact inventory containing overview IDs, target names,
and pending-update information. For substantive entity-focused assignments it
checks whether an overview exists. If absent, commission it before the dependent
worker; unrelated tasks and document readers can proceed. Narrow direct extraction
or incidental name mentions do not require an overview.

If new cards exist, the orchestrator chooses a refresh or uses the existing overview
plus the new cards. A Python update signal means new material exists, not that the
old overview is automatically invalid. Use the normal task queue and signal system;
no extra planning call, automatic overview scheduler or general dependency engine.
Python prepares the overview input directly from the request's name/group ID.
Deduplicate pending requests for the same retrieval group. Repeated provider failure
must not create an endless prerequisite loop; surface it and allow explicit fallback
to original-card work through the existing orchestration path.

## Overview context and output

Provide all cards mentioning the target or its candidate variants, in full, with
original IDs, available sources and match reasons. Include a card mentioning both
A and B when preparing A's overview; do not expand to cards mentioning only B.
No top-K filtering or silent input truncation. Current fixtures should fit. If an
actual input exceeds model capacity, report that fact and reconsider the process;
do not prebuild multi-stage summarization for a hypothetical large corpus.

Candidate 1, the detailed entity dossier, is the selected starting prompt:

> Produce a detailed, reusable overview of the target and candidate name variants.
> Organize identity and naming, activities and roles, relationships and transactions,
> relevant dates/amounts/identifiers, contradictions, and open questions. Preserve
> specific details rather than broad summaries. Combine repetition while retaining
> supporting card references. Include potentially useful findings beyond the immediate
> question. Cards were retrieved using broad name matching: similar or identical
> names may refer to different entities, and different spellings may refer to one
> entity (including abbreviations and typos). Do not assume identity from retrieval.
> Explain supported associations, distinctions and uncertainty, citing original cards.

Return one detailed freestyle `entity_overview` card within the existing worker
response envelope. Additionally, emit ordinary `analysis` or `gap` cards only when
comparison reveals new synthesis or actionable unanswered questions. Do not reproduce
existing findings. Reference original supporting cards; the overview is not new
independent evidence. Use existing question/signal fields to suggest investigation;
the orchestrator decides which work to dispatch. Avoid demanding a rigid dossier
schema, a new review call or a separate worker just to extract follow-up tasks.

Store the overview's input-card set in Python so later additions can be detected.
Refresh creates a new overview card using the existing replacement/supersession
mechanism as appropriate; old overview text is not silently rewritten. Do not feed
previous overviews back as independent evidence for new overviews.

Two prompt alternatives are deferred comparisons on identical input cards:
question-focused briefing, and evidence/contradictions review. Compare detail retained,
unsupported identity joins, useful new findings, downstream use, calls and tokens.
No extra prompt-comparison calls are implicitly authorized by choosing candidate 1.

## Downstream use and final synthesis

Assign relevant overview IDs through `reads_from_blackboard`. When supplied to a
worker, an overview is never cut to upstream's 300-character entry excerpt. Pass
its complete text and references. Keep ordinary entry rendering unchanged. Workers
use the overview for orientation, follow original supporting cards for consequential
claims, and treat it as derived reasoning. Make available overviews visible even
when they are no longer among the current iteration's recent entries.

For the first experiment, exclude `entity_overview` from automatic curation,
obligation generation and direct final-synthesis evidence pools, including fallback
paths. This is a narrow new-type filter; do not change upstream selection for ordinary
cards. Additional analysis/gap cards from the overview worker enter those stages
normally. New analytical connections can therefore reach the report without the
full overview becoming a mandatory report statement.

Upstream curation currently sees 400 characters per card. Obligations select known
analytical types and qualifying observations. Selected `must_include` items are
grouped by section and drafted with referenced evidence; final completeness checks
verify selection survival, not completeness of the original selection. This is a
known baseline limitation, not a new redesign target. If observed omissions justify
it, a later experiment may use full overviews as guidance for obligation selection.
Do not modify worker guidance and final selection simultaneously in the first test.

## Minimal state, failure boundaries and evidence

Reuse ModelCaller, Entry, existing signals/task queue, JSON run files, and standard
library matching. No new database or dependency. Keep credentials in existing
configuration and never copy or print them. Do not drop source findings after a
name-list error. Preserve failed-work visibility and actual/unknown provider usage;
count shared batch usage once rather than inventing per-card token allocations.

Earliest live slice: a finite set of real or synthetic cards → real name-list call
→ Python matching → real entity-overview call → original worker receives the full
overview. Inspect original names, all matched input IDs, similar-name ambiguity,
overview references and any additional analysis/gap cards. Include a supported
variant case and unrelated same-name entities. A fake caller is sufficient to test
failure handling and scheduling branches; it is not live extraction evidence.

First check parsing and boundary matching, full-context delivery, no recursive
counting, overview prerequisites without blocking unrelated work, and exclusion from
final selection while additional ordinary cards survive. Measure overview requests,
input/output tokens, downstream recipients and final-report quality. Supplied context
is observable; internal model attention is not. An unused overview is cost without
demonstrated benefit.

Use the same pinned upstream and provider settings for baseline and augmented runs.
The August completed run is historical context, not the controlled baseline. Live
runs, prompt comparisons and judging require concrete execution scope; no such calls
have been made while preparing this design.
