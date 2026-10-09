# Prompt Injection Classifier — Specialized Neural Network

A compact text classifier that flags prompt-injection content without calling an LLM. Built to
scan retrieved RAG chunks, web-search results, and uploaded files inside Open WebUI before they
reach the model, and to be importable from any Python app as a plain function.

Sibling of [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn): same
pipeline discipline (Hugging Face sources only, fingerprinted dataset builds, leakage-safe splits,
baselines before the neural model, locked holdouts), same multi-device notebooks.

Replaces the per-call LLM round trip of the existing
[prompt-injection LoRA](https://github.com/beaudamore/safety) where latency matters. The LoRA
stays the detector for typed chat prompts; this model covers the retrieval path, where a single
turn can carry ten or more chunks.

**Status (2026-10-09):** two data builds (v1, v2), Phases 2 to 4 run on each, every run reported
under `reports/`. v2 fixed part of what v1 exposed and uncovered a smaller, deeper problem in the
indirect-injection source; v3 is the next build. Phase 3 has not yet cleared the promotion gate.
See `docs/PLAN.md` for the design and `docs/PROCESS.md` for how a run works.

---

## Progress and findings

This section is the product's progression to completion, kept honest: what was built, what the
numbers say, what they do not say, and what changes next. Per-run detail lives in `reports/`.

### v1 data (2026-10-09, fingerprint `febe6837…`)

Eight Hugging Face sources (see `docs/PLAN.md` §2), exact and MinHash deduplication, held-out
isolation of the LoRA's comparison sets, grouped 80/10/10 splits balanced over label, source,
reason and shape, a locked adversarial set.

| Split | Rows | Injection | Safe |
| --- | ---: | ---: | ---: |
| train | 50,481 | 26,344 | 24,137 |
| val | 5,640 | 2,618 | 3,022 |
| test | 5,661 | 2,649 | 3,012 |
| adversarial_test | 2,649 | 2,649 | 0 |
| lora_holdout | 2,086 | 1,141 | 945 |

### Phase 2 and Phase 3 on v1 (2026-10-09, locked test split, DGX Spark CUDA)

| Model | F1 | Recall | FPR | Recall at FPR 0.1% | Recall at FPR 1% | Adversarial recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF logistic regression (Phase 2 winner) | 0.903 | 0.892 | 7.4% | 0.139 | 0.655 | 0.952 |
| XGBoost on 17 features | 0.795 | 0.909 | 33.2% | 0.200 | 0.384 | 0.978 |
| CNN + BiGRU, two heads (Phase 3) | 0.891 | 0.888 | 9.2% | 0.385 | 0.660 | 0.958 |

Phase 3: 6.7M parameters (6.4M of them the embedding table), 26.7 MB, CPU p95 36 ms for a batch of
ten 512-token chunks. Reason-head accuracy on true positives 0.96, top-2 0.99, over five populated
reasons.

### What the v1 numbers mean

1. **The neural model does not beat the bag-of-words baseline at the F1 operating point.** It wins
   only where the filter actually operates: at an FPR budget of 0.1% it catches 38.5% of attacks
   against 13.9% for TF-IDF. The promotion gate is therefore recall at a stated FPR budget plus the
   document-shaped FPR, not F1 (`docs/PLAN.md` §4).
2. **The model learned BIPIA's documents, not its attacks.** BIPIA reuses about 850 base documents
   under 125 attack strings. v1 kept each attack string in one split but let a base document appear
   attacked in train and clean in test. Result: 61% of clean BIPIA documents are flagged. On
   Wikipedia prose the FPR is 0.8%; the failure is specific to reused structured documents.
   This is a data-construction flaw, fixed in v2 by partitioning documents and attack strings
   jointly and by adding structured benign documents that are not BIPIA.
3. **Chat-shaped attack data is template-heavy.** 42-43% of val/test positives share their first six
   words with a train positive; the largest split group holds 277 rows. Chat-shaped recall and the
   reason-head numbers are optimistic for that reason: reasons were assigned per source, so part of
   the reason head's accuracy is source recognition. Reason labels stay "report, do not gate".
4. **Adversarial-but-unharmful WildGuardMix prompts are flagged 21% of the time.** They are
   labelled SAFE to match the LoRA's contract; the slice is kept visible rather than hidden.

The process did its job: baselines, locked holdouts and slices surfaced all four before any number
was promoted. The same thing happened in `email-classifier-nn` (v1 contamination, rebuilt as v2).

### v2 data (2026-10-09, fingerprint `d3d3fa8d…`)

Changes from v1: BIPIA partitioned so that an injected test row carries a document and an attack
string both unseen in train; legitimate emails from `puyang2025/seven-phishing-email-datasets`
added as structured document-shaped SAFE rows; template key shortened to 8 words and the
near-duplicate threshold lowered to 0.80.

| Split | Rows | Injection | Safe |
| --- | ---: | ---: | ---: |
| train | 52,563 | 26,144 | 26,419 |
| val | 5,814 | 2,449 | 3,365 |
| test | 5,775 | 2,389 | 3,386 |
| adversarial_test | 2,389 | 2,389 | 0 |
| lora_holdout | 2,086 | 1,141 | 945 |

### Phase 2 and Phase 3 on v2 (2026-10-09, locked test split, DGX Spark CUDA)

| Model | F1 | Recall | FPR | Recall at FPR 0.1% | Recall at FPR 1% | Adversarial recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF logistic regression (Phase 2 winner) | 0.906 | 0.913 | 7.3% | 0.269 | 0.621 | 0.967 |
| XGBoost on 17 features | 0.768 | 0.861 | 26.9% | 0.141 | 0.319 | 0.966 |
| CNN + BiGRU, two heads (Phase 3) | 0.884 | 0.883 | 8.1% | 0.377 | 0.629 | 0.962 |

Test F1 95% bootstrap interval for the CNN: 0.875 to 0.894. Slices: legitimate email FPR 0.0%,
Wikipedia 1.2%, WildGuardMix benign 1.0%, adversarial-but-unharmful 21.3%; Nemotron recall
0.969, WildGuardMix adversarial 0.802, BIPIA injected 1.000. Six-word template twin rate 39%
(v1: 43%). Model: 6.7M parameters, 26.7 MB, CPU p95 53 ms for ten 512-token chunks.

### Phase 4 on v2: the LoRA's held-out sets (2,086 rows)

| | Value |
| --- | ---: |
| Accuracy, SAFE vs INJECTION | 0.901 |
| Invalid outputs | 0 |
| False positives | 89 / 945 (9.4%) |
| False negatives | 118 / 1,141 (10.3%) |
| Nemotron held-out jailbreaking, recall | 0.970 |
| WildGuardTest, recall / FPR | 0.724 / 9.4% |
| Recall at FPR 1% / 0.5% / 0.1% | 0.526 / 0.374 / 0.223 |
| Average precision (95% CI) | 0.969 (0.960 to 0.977) |

Written as `eval_wildguard_test.json` in the run directory with the field names the LoRA eval
notebook uses. The LoRA's own eval has not been run, so there is no head-to-head yet.

### What the v2 numbers mean

1. **Held-out isolation works.** The LoRA holdout numbers track the test split (F1 0.908 vs 0.884,
   FPR within a point). Nothing suggests leakage into the comparison set.
2. **Phase 3 does not clear the gate on v2.** The TF-IDF baseline closed most of its gap at the
   low-FPR budgets; at a 1% budget the two are tied. The neural model earns its place only if v3
   separates them.
3. **The BIPIA fix was defeated by a bug, and the real problem is smaller than it looked.** The
   v2 document key hashed the raw base text; removing an attack from the middle of a document
   leaves different whitespace than removing it from the end, so one document became up to three
   ids and the partition treated them as different documents. Checked from the raw data: BIPIA has
   only **182 distinct base documents** (47 emails, 84 tables, 51 code files); 18 are attacked in
   both train and test; 5 of the 14 clean test documents were seen attacked in train. In train,
   injected BIPIA rows outnumber clean ones 5,792 to 146, so each training document appears
   about 35 times as an injection and once as clean. With that prior the model learns
   "BIPIA-style document", not "attack inside document". The clean-BIPIA slice (43% of 14 rows
   flagged) is too small to measure anything and is not a result.
4. **The template twin rate barely moved** (43% to 39%). The jailbreak corpora are generated
   from skeletons, and a shorter key plus a lower MinHash threshold does not change that. The
   indicator stays in the manifest; the fix is more diverse attack data, which is Phase 4's job.

**v3 (next build).** Key BIPIA documents by their whitespace-normalized base text; cap injected
rows per base document (three attacks each) so BIPIA becomes a small, honest indirect-injection
slice of roughly 500 train rows; report the clean-BIPIA slice with its row count. Tables and code
still have no benign counterpart outside BIPIA; emails now do.

---

## What it will produce

| Artifact | Purpose |
|---|---|
| `src/` package | `classify(text)` and `classify_batch(texts)` returning a probability, a verdict, and a reason from the nine-reason taxonomy shared with the Open WebUI filter. |
| `filter/` | An Open WebUI `request`-hook filter that scans every `<source>` block the retrieval pipeline injected, drops flagged chunks, and removes them from the citation list. No Open WebUI patch required. |
| `models/<run-id>/` | Versioned weights, vocab, thresholds, metrics, and a run manifest tied to the dataset fingerprint. Gitignored; rebuilt by running the notebooks. |

## Output contract

Matches the existing filter and LoRA so the two detectors are interchangeable at the call site:

```text
SAFE
INJECTION: <reason>
```

Valid reasons: Override Attempt, Fake System Injection, Meta-Command Injection, Jailbreak
Technique, Delimiter Confusion, Privilege Escalation, Embedded Instruction, Role-Play Bypass,
Chain-of-Instructions.

## Repo layout

```text
prompt-injection-nn/
├── README.md
├── CLAUDE.md          Handoff context for Claude Code on any machine
├── .claude/skills/    pipeline-status, notebook-editing
├── docs/
│   ├── PLAN.md        Scope, data strategy, architecture, Open WebUI integration, phase roadmap
│   ├── PROCESS.md     How to run the pipeline on the DGX Spark, a Mac, or CPU
│   └── EVALUATION.md  Verification protocol: leakage audit, metrics, CIs, robustness, filter tests, human review
├── notebooks/
│   ├── datagen/       Phase 1: build fingerprinted splits from Hugging Face sources
│   ├── training/      Phase 2 baselines, Phase 3 neural model
│   └── eval/          Standalone evaluation against the LoRA's held-out sets
├── src/               Installable inference package
├── filter/            Open WebUI request-hook filter
├── data/              (gitignored) raw pulls and clean splits
└── models/            (gitignored) versioned run artifacts
```

## Related

- [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) — the live filters, including the inlet prompt-injection filter this complements
- [safety](https://github.com/beaudamore/safety) — the prompt-injection and content-safety LoRAs; the dataset recipe and eval sets here are taken from its notebooks
- [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn) — the pipeline this repo mirrors
