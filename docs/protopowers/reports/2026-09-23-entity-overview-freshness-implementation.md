# Entity overview freshness implementation checkpoint

Task 1 scheduling and labelled transfer are committed in `75b1174`. The live
runner fixes are committed in `e7f37e7` and `056a91d`. Task 2 extends the same
request, scheduling, publication, and recipient path to supervisor rounds. A
frozen final pass attempts affected overviews once, in batches of at most three,
after evidence-producing stages and before synthesis projection. It includes a
single late direct-document finding below the routine four-card threshold. The
sidecar now shows eligible backlog, scheduled work, failures, budget-limited work,
selection reasons, and final attempts.

## Evidence

- Offline: 80 focused overview, packet, and synthesis tests passed. A fake-model
  lifecycle introduced one Swiss residency card during supervisor work, refreshed
  the Petrov profile in the final pass, and projected a subject-labelled row
  supported by the original card. The screening candidate remained separate.
  [Before supervisor](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/pytest-task2-verified/test_supervisor_attachment_the0/swarm/blackboard_iter_5_before_supervisor.json),
  [after supervisor](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/pytest-task2-verified/test_supervisor_attachment_the0/swarm/blackboard_iter_5_after_supervisor.json),
  [final state](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/pytest-task2-verified/test_supervisor_attachment_the0/swarm/blackboard_iter_5_final_check.json),
  and [packet](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/pytest-task2-verified/test_supervisor_attachment_the0/packet.json).
- Offline: an unusable refresh retained the prior valid record and reported the
  failed group. A budget-limited pass made no model call and exposed the backlog.
  Inactive support was excluded from both worker attachment and packet projection.
- Real model: the September 22 finite checkpoint produced no usable model
  response. In the final attempt, two logical refresh calls made ten provider
  attempts total; each logical call ended with a Gemini `503 UNAVAILABLE`
  high-demand error after the configured 20-second retry waits. Individual
  attempt errors were not recorded. The downstream worker and draft were not
  reached. [Attempt record](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/live-20260923-3/entity_overview_freshness.json).
- Real model, Gemini 3.5 Flash-Lite: the bounded Task 1 checkpoint completed four
  logical calls, each in one provider attempt. Both refreshed personal overviews,
  the ordinary worker, and the short draft completed. The labelled packet places
  Petrov's Swiss permanent residency and Volkov's UAE residency under the
  respective subjects and retains distinct screening profiles. The draft states
  both residency facts. [Task 1 record](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/live-20260923-gemini35-task1/entity_overview_freshness.json).
- Real model, Gemini 3.5 Flash-Lite: the Task 2 final pass used one logical call
  and one provider attempt on a saved Petrov slice with exactly one new direct
  source card, below the routine four-card trigger. It published a new overview;
  the packet attaches Swiss permanent residency to the client/UBO profile and
  keeps the sanctioned namesake in separate labelled rows. The command initially
  exited with a false-negative verifier error because it only recognized the word
  “candidate,” whereas the model labelled the separate profile “Sanctioned
  Entity.” The saved live response passed the corrected verifier offline; no paid
  rerun was made. [Task 2 record](../../../.protopowers/runs/2026-09-23-entity-overview-freshness/live-20260923-gemini35-task2/entity_overview_freshness.json).

## Limits and next experiment

The final pass treats the existing evidence catalogue as its relevance boundary.
For much larger corpora, inspect whether this admits too many low-value groups
before narrowing by task or artifact context. The budget check uses the actual
constructed prompt's UTF-8 byte length plus the requested 8,192 output tokens;
provider billing tokens remain the source of truth after each call. No full swarm
or judge run was launched. Repeated trust, vessel-history, and identity
investigations must be reviewed in the next authorized evaluation for added
evidence, corrections, or confusion. If the user elects a wider evaluation,
reconsider report deduplication then. These finite checkpoints show the named
paths, not full-run report quality or repeated-investigation behavior.
