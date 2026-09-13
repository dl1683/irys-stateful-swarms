# Entity Overview Full Run Findings

## Scope

This report records the completed Harvey LAB run for `international-trade-sanctions/extract-transaction-entity-details`. The run used Gemini 3.5 Flash-Lite for the swarm, a 5,000,000-token ceiling, and a 250-request cap. It completed successfully and produced `entity-extraction-report.docx`.

## Run outcome

- Status: completed after 15 iterations and 1,407.8 seconds.
- Usage: 1,701,267 total tokens (1,218,899 input; 319,370 output), USD 1.1641 recorded cost, and 160 Gemini 3.5 Flash-Lite calls.
- Coverage: all seven supplied documents were read; no documents were skipped.
- Output: one DOCX report was produced.

## Entity overview evidence

The implemented overview mechanism was used by the running swarm, rather than merely being available in code.

- Name discovery made 16 calls (48,002 tokens) and processed 952 original cards into 44 retrieval groups / 46 candidate spellings.
- The swarm made three overview calls (24,321 tokens). The final blackboard holds three active overview cards:
  - `e642` (iteration 4), Crestmoor Trading AG;
  - `e689` (iteration 6), a refresh for the same retrieval group; and
  - `e778` (iteration 10), a separately discovered `crestmoor` group.
- Sixteen worker tasks received an overview card in context. Eight downstream outputs explicitly cited an overview ID in `supports`, demonstrating actual use rather than only context delivery.
- The cited outputs include ownership, sanctions-match, identity-resolution, and OFAC 50 percent-rule analysis. For example, downstream work used the Crestmoor overview to investigate the Volkov/Orion/Zenith ownership chain and the Petrov date-of-birth discrepancy.
- There were no failed overview calls or name-discovery failures. One malformed request at iteration 8 used entry ID `e642` as an overview retrieval ID; the new guard rejected it cleanly instead of dispatching a broken worker.

## Findings and limitations

1. The core behavior succeeded: broad matched-card input was synthesized into reusable context and that context was cited by later analytical work.
2. Grouping needs refinement. `crestmoor trading` and `crestmoor` were treated as separate retrieval groups, producing overlapping overview work. The report retains only the latest overview record for a group, so its `overview_count` is two while `overview_calls` is three; this is expected from refresh replacement but can be clearer.
3. The final report is not production quality. It renders to 102 pages, repeats large sections, includes literal Markdown markers such as `**`, and ends with a nearly blank final page. These are final-synthesis/formatting issues, not evidence that the overview feature failed.
4. No controlled no-overview baseline was run. A score can measure this run's overall benchmark performance, but cannot establish the overview feature's causal contribution.

## Recommendation

Run the Harvey scorer with Gemini 3.1 Flash-Lite to capture an absolute benchmark score and criterion-level failures. Interpret the result as a whole-pipeline diagnostic. Then fix name-group deduplication and final-synthesis bloat before attempting a controlled overview-versus-baseline comparison.

## Scoring attempt and rate-limit diagnosis

The first scoring attempt used Gemini 3.1 Flash-Lite at 10 requests per minute and scored zero criteria. Gemini returned `429 RESOURCE_EXHAUSTED` for the free-tier input-token-per-minute metric, whose limit is 250,000 input tokens per minute. This is independent of the remaining daily API-call allowance.

The rendered DOCX contains about 230,571 characters (roughly 57,600 input tokens before the rubric instructions). Harvey scoring sends the relevant deliverable to the judge for each of the 85 criteria. Ten requests per minute can therefore exceed the 250,000-token-minute quota even with sequential criterion execution. Use two requests per minute for the retry; this leaves a substantial margin for rubric text and tokenization variation. The retry will be slow (roughly 43 minutes for 85 criteria, excluding retries) but does not rerun the swarm.

## Evidence

- Run status and metrics: `results/entity_overview_full_20260913_flash35_250/international-trade-sanctions/extract-transaction-entity-details/status.json` and `metrics.json`.
- Entity telemetry: `results/entity_overview_full_20260913_flash35_250/international-trade-sanctions/extract-transaction-entity-details/swarm/entity_overview_report.json`.
- Final state: `results/entity_overview_full_20260913_flash35_250/international-trade-sanctions/extract-transaction-entity-details/swarm/blackboard_iter_15_final.json`.
- Deliverable: `results/entity_overview_full_20260913_flash35_250/international-trade-sanctions/extract-transaction-entity-details/output/entity-extraction-report.docx`.
