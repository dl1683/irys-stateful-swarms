# Pointer and delta overview: Task 4 measurement

This note measures the prototype from Tasks 1–3 against two saved Harvey runs. It separates an **offline packet replay** from small **Gemini 3.5 Flash-lite calls**. No new end-to-end Harvey run or judge score was produced for the pointer design.

## Offline replay

The replay loaded each final blackboard and packet, removed only rows whose source was `entity_overview_projection`, then built section prompts with the current writer code. Candidate counts use the current name inventory on saved findings. Historical curation, original findings, and packet ordering were held fixed. Prompt sizes are UTF-8 bytes, not billed tokens.

| Saved run | Direct candidate groups / unique cards | Packet rows: saved → without projection | Section writer jobs: saved → replay | Aggregate section prompt bytes: saved → replay | Projected-only original IDs absent from replay writer evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-22 | 29 / 259 | 405 → 251 | 29 → 24 | 392,889 → 227,767 (−42.0%) | 90 of 142 |
| 2026-09-26 | 33 / 409 | 480 → 251 | 31 → 23 | 601,489 → 339,626 (−43.5%) | 346 of 436 |

All 154 September 22 and 229 September 26 projected rows had original-card IDs. The replay retained 238 and 239 curation rows respectively, plus 13 and 12 artifact-contract rows. The pointer evidence path recovered 52 of 142 and 90 of 436 original IDs that otherwise appeared only in projection. The remaining IDs are absent from the simulated section prompts. This counts **source cards, not unique facts or failed rubric criteria**; repeated findings can support one fact. It identifies a material coverage risk when ordinary curation does not select an important fact.

The saved September 22 report passed **83/85** judged criteria. The saved September 26 report passed **82/85**. Those scores describe the older structured-overview pipeline and cannot be assigned to this prototype. The saved reports contained approximately 13,185 and 15,906 words by the same DOCX XML extraction method. No new full report exists, so a report-length change is unmeasured.

## Calls and cost ledger

| Job kind | Observation | Input / output tokens | Requests, validation, latency |
| --- | --- | ---: | --- |
| Name discovery | Historical September 22: 16 calls, 24,047 tokens; September 26: 22 calls, 25,449 tokens | Not split in sidecar | No new discovery call; current-run cost unmeasured |
| Initial overview | Task 2 saved-finding replay produced one card with cited originals | 550 / 299 | 2 provider attempts; passed; 7.25 s for the initial stage |
| Routine delta | Task 2 replay added source-linked facts | 493 / 284 | 1 attempt; passed; 7.54 s |
| Repair delta | After source e120 was invalidated, its sole-supported fact stayed hidden and e10 was only a candidate hint | 788 / 5 on final repair call | 1 attempt on that call; passed; 0.87 s. A prior empty-operation repair call also made 1 request. |
| Identity review | Task 3 saved-finding pair with a shared register number redirected the later card | 540 / 113 | 1 attempt; passed. Latency was not recorded. |
| Exceptional rebuild | A confirmed split rebuilt separate cards in fixture tests | — | No paid rebuild call or price measurement |
| Final section writer | Task 4 replay's six-item `Entity Profile` section | 4,658 / 805 | 1 attempt; 3.91 s measured inside the call; 526 output words |

The historical September 22 and September 26 runs recorded 29 and 98 overview calls, costing 68,598 and 415,940 tokens; they combined creation and refresh under the old lifecycle. Their whole-run recorded costs were **$0.5925** and **$0.8010**. These are not prices for the new jobs. The microcalls did not expose a reliable USD price, and the historical metrics do not isolate final-writing cost. Provider attempts, latency, and model validation were recorded for the bounded calls where available; no savings in actual billed tokens or total runtime can be inferred from prompt bytes alone.

The live writer prompt contained 52 rendered evidence entries. Its output made six substantive bullets and had **zero explicit entry-ID citations**. The packet and prompt retained source attribution, but this section draft did not display it. That is a report behavior to inspect in any full trial, especially where a conclusion depends on a disputed identity.

## Verification and remaining limit

The focused entity, pointer, packet, section-writer, and resume suite passed **85 tests**. A saved iteration-14 checkpoint loaded with 604 entries, seven documents, and 32 entity groups. Task 3's fixture checks cover unresolved/budget-limited pairs, reversible redirects, and split rebuilds; the paid pair review covers one supported merge only.

The smallest remaining shortcut is reliance on existing curation to select facts for writing. Pointer lookup adds up to four extra originals for a relevant overview, but it cannot guarantee inclusion of a fact that has no selected item. **Upgrade trigger:** if a controlled full run shows a required sourced fact omitted from the packet or report, add a domain-neutral selection check at curation/packet construction, then remeasure. Do not restore broad overview projection or add report-length controls without that evidence. A full Harvey run and judge comparison are needed to establish quality, total cost, and report length for the new pipeline.

Reproducible local artifacts are under `.protopowers/runs/2026-09-26-entity-overview-pointer-delta/task4/`: `replay.py`, `replay_result.json`, `live_writer.py`, `live_writer_result.json`, and the single section output. Historical source snapshots, packets, metrics, and DOCX files are under `results/harvey_runs/20260922_gemini_31_flash-lite/` and `results/harvey_runs/20260926_gemini_31_flash-lite/` for `international-trade-sanctions/extract-transaction-entity-details`.
