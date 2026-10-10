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

**Status (2026-10-09):** four data builds (v1 to v4), Phases 2 to 4 run on v1 to v3, every run
reported under `reports/`. **On v3 the neural model lost the promotion gate to the TF-IDF baseline at
every FPR budget**, and the v3 test set turned out to measure the wrong thing: 82% chat-shaped rows,
53 document-shaped attacks from one source. v4 rebuilds the data around the product's actual input
(documents, emails, tool results, agent messages) from eleven newly verified Hugging Face sources;
Phases 2 to 4 on v4 are the next run. New to the project? Read `docs/LEARNING.md` first. Design in
`docs/PLAN.md`, how to run in `docs/PROCESS.md`.

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

### v3 data (2026-10-09, fingerprint `a4c8dd47…`)

Changes from v2: BIPIA documents keyed by whitespace-normalized base text (v2 split one document into
up to three ids); injected BIPIA rows capped at three per base document, so BIPIA became a small,
honest indirect-injection slice (554 train rows against 5,792 in v2).

| Split | Rows | Injection | Safe |
| --- | ---: | ---: | ---: |
| train | 47,570 | 21,034 | 26,536 |
| val | 5,419 | 2,102 | 3,317 |
| test | 5,419 | 2,102 | 3,317 |
| adversarial_test | 2,102 | 2,102 | 0 |
| lora_holdout | 2,086 | 1,141 | 945 |

### Phase 2 and Phase 3 on v3 (2026-10-09, locked test split, DGX Spark CUDA)

| Model | F1 | Recall | FPR | Recall at FPR 1% | Recall at FPR 0.5% | Recall at FPR 0.1% | Adversarial recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF logistic regression (Phase 2 winner) | 0.877 | 0.901 | 9.7% | **0.641** | **0.539** | **0.423** | 0.955 |
| XGBoost on 17 features | 0.744 | 0.883 | 31.0% | 0.280 | 0.207 | 0.137 | 0.960 |
| CNN + BiGRU, two heads (Phase 3) | 0.860 | 0.885 | 10.9% | 0.577 | 0.504 | 0.328 | 0.958 |

Test F1 95% bootstrap interval for the CNN: 0.849 to 0.871. Document slice: CNN recall 0.981 / FPR
1.3% against TF-IDF 0.849 / 1.6%, on 53 injected and 918 clean rows. Clean-BIPIA FPR 3 of 18.
Adversarial-but-unharmful slice flagged 28% (TF-IDF 26%). Latency: p95 34.9 ms for ten 512-token
chunks on the Spark's CPU, target 100 ms, met. Phase 4 on the LoRA held-out sets: accuracy 0.905,
FP 12.4%, FN 7.1%, recall at FPR 1% 0.616.

### What the v3 numbers mean

1. **The gate failed.** The baseline catches more attacks than the neural model at every FPR budget.
   On v2 the neural model had won at the 0.1% budget; the BIPIA fix removed the document shortcut it
   had been using. Per the process, the baseline is what ships unless v4 changes the verdict.
2. **Both numbers are too low to call either model a detector.** Dropping 1% of good chunks while
   missing 36 to 42% of attacks is a speed bump.
3. **The test set measured the wrong thing.** The product scans retrieved documents and tool
   results; 82% of the test rows were chat-style jailbreaks, and the document-shaped slice held 53
   attacks from one source. The neural model's only clear win was on that slice, which is a hint,
   not a result.
4. **Four of nine reasons had zero training rows** (Fake System Injection, Delimiter Confusion,
   Role-Play Bypass, Chain-of-Instructions). Reason-head accuracy on the others was 0.96.
5. **Template twin rate 34%**, barely moved from v2 (39%); the jailbreak corpora are skeleton-based.

### v4 data (2026-10-09, fingerprint `7f42d9d8…`)

Built to measure what the product actually reads. Eleven sources added after a Hugging Face search
on 2026-10-09 in which every candidate's licence, schema and row sample were checked (the list, the
rejections and the reasons are in the v4 datagen notebook header and in `docs/LEARNING.md` §7):
real adaptive attack emails from Microsoft's LLMail-Inject challenge, injected tasks planted in RAG
passages (PIArena), instructions hidden in tool-returned business records (Nemotron agentic IPI),
injections inside real Kubernetes manifests, three 3nesdeniz agentic and direct-attack sets with
matched hard negatives, real in-the-wild jailbreaks and ordinary prompt templates, everyday benign
requests and code (ogma), S-Labs hard negatives, and NotInject, which is forced into `test` so
over-blocking is measured and never learned. Every clean-and-injected pair shares a group key.
LLMail rows are re-rendered to the same layout as the legitimate emails so the challenge wrapper
cannot become the signal. A third shape, `agentic`, covers tool requests, agent messages and memory.

| Split | Rows | Injection | Safe |
| --- | ---: | ---: | ---: |
| train | 82,800 | 39,315 | 43,485 |
| val | 9,367 | 3,931 | 5,436 |
| test | 9,456 | 3,904 | 5,552 |
| adversarial_test | 3,904 | 3,904 | 0 |
| lora_holdout | 2,086 | 1,141 | 945 |

What changed for the product's question, v3 to v4, on the test split:

| | v3 | v4 |
| --- | ---: | ---: |
| Document-shaped attacks | 53, one source | 1,352, six sources |
| Document-shaped rows | 971 | 2,615 |
| Agentic-shaped rows | 0 | 531 |
| Reasons with training rows | 5 of 9 | 9 of 9 |
| Chat share of test | 82% | 67% |

Smallest reasons in train: Chain-of-Instructions 212, Role-Play Bypass 264, Delimiter Confusion 329.
Enough to learn, thin enough that their per-reason numbers will carry wide intervals.

Twin-rate caveats, recorded per source in the manifest: Kubernetes manifests share their JSON header
(94%), which measures file format, not attack reuse; LLMail attack bodies are heavily reused by
contestants (64% even after the subject line is excluded from the key); PIArena and BIPIA are near
zero. Overall chat twin rate 32%, document 59%, agentic 19%.

**Next run:** Phases 2, 3 and 4 on v4, then re-apply the gate. If the neural model still loses to
TF-IDF at the FPR budgets, the baseline ships behind the `src/` API.

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
│   ├── LEARNING.md    Every concept from zero: injection, splits, leakage, the gate, the scores, the data search
│   ├── PLAN.md        Scope, data strategy, architecture, Open WebUI integration, phase roadmap
│   ├── PROCESS.md     How to run the pipeline on the DGX Spark, a Mac, or CPU
│   └── EVALUATION.md  Verification protocol: leakage audit, metrics, CIs, robustness, filter tests, human review
├── notebooks/
│   ├── datagen/       Phase 1: build fingerprinted splits from Hugging Face sources (v3 kept, v4 current)
│   ├── training/      Phase 2 baselines, Phase 3 neural model
│   └── eval/          Phase 4: evaluation against the LoRA's held-out sets
├── reports/           One folder per executed run: report.md and plots (tracked; data and models are not)
├── src/               Installable inference package
├── filter/            Open WebUI request-hook filter
├── data/              (gitignored) raw pulls and clean splits
└── models/            (gitignored) versioned run artifacts
```

## Related

- [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) — the live filters, including the inlet prompt-injection filter this complements
- [safety](https://github.com/beaudamore/safety) — the prompt-injection and content-safety LoRAs; the dataset recipe and eval sets here are taken from its notebooks
- [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn) — the pipeline this repo mirrors
