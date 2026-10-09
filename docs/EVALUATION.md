# Evaluation and Verification Protocol

**Last updated:** 2026-10-09
**Scope:** what has to be true, and what has to be measured, before a model from this repo is called good and before the filter is loaded into a production Open WebUI. Written so a reviewer can check each item against the notebooks, the run manifests, and the filter tests.

This protocol is the email-classifier-nn protocol adapted to a document-scanning task with a reason head and a latency budget. Everything is **Open** today because nothing has been built; the status column will be maintained as phases land. Section 11 is the backlog order.

---

## 1. Data integrity and leakage audit

| Check | Method |
|---|---|
| Exact-duplicate removal | SHA-256 over normalized text; removed count reported |
| Near-duplicate grouping | MinHash LSH over word shingles; clusters never straddle splits. WildGuardMix contains paraphrase families and Nemotron jailbreaks reuse templates, so this matters more than it did for email |
| Template-aware grouping | Jailbreak prompts that share a template skeleton (DAN variants, "developer mode" variants) are grouped by skeleton, not only by shingle overlap, so a template seen in train is not graded in test |
| Group-aware, stratified splitting | By cluster and template, stratified by label, reason, and source |
| Cross-split overlap test | Zero shared fingerprints or clusters between any two splits |
| Held-out-set isolation | WildGuardMix `wildguardtest` and Nemotron held-out `jailbreaking` rows are excluded from train and val by ID and by fingerprint, so the LoRA comparison is clean |
| Label and reason contract | Label in {SAFE, INJECTION}; reason is one of nine for INJECTION and empty for SAFE; per-split reason distribution in the manifest |
| Shape balance | Share of chat-shaped vs document-shaped text per class per split, so the model cannot learn "document-shaped means safe" |
| Reproducible build | Config fingerprint in the manifest; rebuild yields identical counts |
| Datasheet | Source cards, licenses, the subcategory-to-reason map, known gaps (six reasons empty in the base recipe), synthetic-row provenance |
| Label-noise estimate | Human review per section 10 |

## 2. Experimental protocol

| Requirement | Method |
|---|---|
| Three-way split discipline | `train` fits; `val` drives early stopping, calibration, threshold, selection; `test`, `adversarial_test`, and the LoRA held-out sets are predicted once after selection is frozen |
| Seed and environment capture | As in the email repo: seeds for every RNG, package versions, device, CUDA version |
| Run manifest | Run ID, dataset fingerprint and version, tokenizer config, selection criteria, holdout policy, synthetic mixture manifest if any |
| Baseline before complexity | TF-IDF logistic regression and XGBoost on engineered features set the bar for the CNN+BiGRU |
| Multi-seed variance | 3 to 5 seeds per final configuration; mean ± standard deviation |
| Ablations | Binary head only vs two heads; cased vs lowercased vocab; with vs without delimiter-preserving tokenization; with vs without synthetic rows; 256 vs 512 tokens |
| Learning curves | Metrics vs training fraction |
| Hyperparameter search | On `val` only, space and trial count recorded |

## 3. Metrics

### 3.1 Binary injection head

| Metric | Note |
|---|---|
| Average precision (PR-AUC) | Headline ranking metric |
| ROC-AUC | Secondary |
| Precision, recall, F1, specificity, FPR, confusion matrix | At the selected threshold |
| **Recall at fixed FPR** (recall@FPR=1%, 0.5%, 0.1%) | The deployment question. A dropped benign chunk silently degrades an answer, so the FPR budget on documents is the binding constraint, not recall |
| FPR at fixed recall (FPR@recall=95%) | Inverse framing |
| Brier, log loss, ECE, MCE, reliability diagram | Calibration; Platt on `val`, isotonic compared |

### 3.2 Reason head

| Metric | Note |
|---|---|
| Macro-F1 over the nine reasons, computed on true positives only | Reason is reported only when the binary head fires; grade it there |
| Per-reason precision and recall | Six reasons start with zero training rows; their numbers are reported as undefined until Phase 4 fills them, never silently averaged in |
| Confusion matrix over reasons | Which attack types are confused with which |
| Top-2 accuracy | Reason labels are weak (map-derived); top-2 is the honest number |

### 3.3 Operating point

F1-max on `val` for development. For deployment, the threshold is chosen from the recall@FPR table against an FPR budget stated in advance by the operator (section 10.3). The chosen point and the budget are recorded in `thresholds.json`.

### 3.4 Slice-based evaluation

Every 3.1 metric per slice, with the worst slice called out.

| Slice |
|---|
| Per source (Nemotron, WildGuardMix, each verified 2.2 candidate, synthetic) |
| Chat-shaped vs document-shaped text |
| Per length bucket, including chunks longer than the token window |
| **Benign lookalikes**: documentation about prompt injection, security write-ups quoting attacks, fiction with "ignore what I said". This slice's FPR is reported on its own; it is where keyword-spotting models fail |
| Indirect injection (instruction embedded in prose) vs direct |
| Per reason, for positives |

## 4. Statistical rigor

| Requirement | Method |
|---|---|
| Confidence intervals | Percentile bootstrap, 1,000 resamples, 95% CI on AP, ROC-AUC, F1, recall, FPR, recall@FPR |
| Paired model comparison | Paired bootstrap on the metric difference; McNemar's test on disagreements; DeLong for AUC |
| **Comparison with the LoRA** | Same held-out rows, same metrics, same `eval_*.json` field names as the `safety` eval notebooks. The LoRA's eval has not been run yet; running it is a prerequisite for any claim that the NN is comparable |
| Effect size | Absolute differences and chunks-per-10,000 affected, not only p-values |
| Multiple comparisons | Stated and corrected, or labeled exploratory |

## 5. Robustness

| Test | Method |
|---|---|
| Adversarial obfuscation | Locked set: homoglyphs, zero-width characters, delimiter wrapping, Unicode tag characters, base64 or ROT13-wrapped instructions, case and whitespace noise. Recall and probability shift reported |
| Leave-one-source-out | Train without one source, test on it |
| Position sensitivity | The same injected instruction at the start, middle, and end of a long benign chunk; recall per position |
| Dilution | Injection text as 50%, 10%, 2% of a chunk; recall vs dilution curve |
| Language | Non-English rows if a verified source carries them; metrics per language or an explicit "English only" statement |
| Label-noise sensitivity | Flip 5% and 10% of training labels; measure degradation |

## 6. Error analysis

| Practice | Method |
|---|---|
| False-negative review | 50 sampled FNs categorized: obfuscation, indirect phrasing, novel template, dilution, label error |
| False-positive review | 50 sampled FPs categorized: documentation about injection, quoted attack, imperative-heavy benign text (recipes, manuals), fiction, label error |
| Hard-example table | Top-20 most confident mistakes in the run directory |
| Reason confusion review | For reason-head errors, whether the map-derived label or the model is wrong |

## 7. Explainability

| Method |
|---|
| Integrated gradients or saliency over tokens for the CNN+BiGRU; the attributed span is surfaced to the operator as the "why" |
| SHAP for the XGBoost baseline |
| Randomized-model sanity check on the attribution method |
| Span agreement: on a reviewed sample, does the attributed span overlap the span a human marks as the injection (section 10.2) |

## 8. Efficiency

The filter sits on the critical path of every chat turn with retrieval.

| Measurement | Target |
|---|---|
| Parameter count and model size | Hundreds of thousands of parameters; single-digit MB |
| Latency, batch of 10 chunks at 512 tokens, CPU | p50 and p95 over 1,000 warm calls; **p95 under 100 ms** on the Spark's CPU and on a laptop CPU, reported with hardware |
| Latency, batch of 1 | p50 and p95 |
| Throughput | Chunks per second at batch 32, CPU |
| Peak memory | During inference |
| Cold start | Time from import to first verdict; the filter loads once per process |

## 9. Software verification

| Practice | Method |
|---|---|
| Notebook structural validation | As in the email repo (`notebook-editing` skill) |
| Input-contract assertions | Every training notebook asserts manifest counts, label and reason contracts, holdout boundaries before training |
| Unit tests, `src/` package | Tokenizer determinism; delimiter and tag preservation; `classify` returns the exact contract string; `classify_batch` order preservation; empty and oversized input handling; thread safety of the loaded model |
| Unit tests, filter | Given a body with N `<source>` blocks and a stubbed classifier that flags block k: block k is removed from the message text, the matching entry is removed from `metadata["sources"]`, the other blocks are byte-identical, a status event is emitted, the turn is not blocked. Also: no sources present leaves the body untouched; malformed tags do not raise; classifier failure fails open and logs |
| Contract test against the RAG template | The parser is tested against the exact `<source id name resource-type resource-id>` rendering from the pinned Open WebUI version, and the test states that version |
| Integration test | Filter loaded into a real Open WebUI of the pinned version with a knowledge base containing one planted injected document; the model never sees the planted chunk; the citation list omits it |
| Golden-metrics regression | Re-run reproduces `metrics.json` within tolerance |
| CI | Lint, unit tests, schema tests, filter tests, notebook smoke test on a sample |

## 10. Human verification

### 10.1 Label quality

| Step | Method | Output |
|---|---|---|
| Annotation guideline | A rubric defining injection vs benign for *documents*, with the edge cases decided: documentation about injection, quoted attacks in reports, imperative benign text, fiction, system-prompt-like boilerplate in legitimate docs | `docs/ANNOTATION_GUIDELINES.md` |
| Reason rubric | One paragraph and two examples per reason; the boundary between Jailbreak Technique and Role-Play Bypass, and between Override Attempt and Fake System Injection, decided explicitly | same file |
| Stratified blind review | 200 rows per class per source, recorded seed, reviewed without the source label | Review sheet |
| Inter-annotator agreement | Two reviewers; Cohen's kappa on the binary label (target ≥ 0.8) and on the reason (target ≥ 0.6, reported separately because reason labels are inherently fuzzier) | Kappa table |
| Adjudication and rubric revision | Third reviewer or discussion; decisions logged; rubric updated | Adjudication log |
| Label-noise estimate | Overturn rate per source; sources above 5% flagged in the datasheet | Noise table |
| Gold set | The adjudicated sample, frozen, never trained on, reported next to `test` | `data/gold-v1/` |

### 10.2 Model error and explanation review

| Step | Method | Output |
|---|---|---|
| Blind error review | Reviewer sees chunk and verdict, not the label; marks model-wrong vs label-wrong and the failure category | Categorized FN/FP table |
| Span marking | On 100 true positives, reviewer highlights the injected span; overlap with the attributed span (section 7) is reported as mean IoU | Span agreement score |
| Calibration sanity | 20 chunks each from the 0.4 to 0.6 and 0.9 to 1.0 bands, reviewed for whether the uncertainty matches intuition | Qualitative note |

### 10.3 Acceptance testing

| Step | Method | Output |
|---|---|---|
| FPR budget interview | Ask the Open WebUI operator how many dropped benign chunks per 1,000 retrievals is acceptable; that number becomes the operating-point budget (section 3.3) | Recorded budget |
| Shadow mode | Filter runs in log-only mode on a real instance for a week; every would-be drop is reviewed | Agreement rate, disagree list |
| Red-team session | A tester plants injected documents into a knowledge base, including styles not in any corpus (indirect, multilingual, diluted, in PDF tables, in code comments); recall per planted document, reported separately from corpus metrics | Red-team set and recall |
| Operator review of the UX | Confirm that the status message on a dropped chunk is understandable and the citation list is correct | Note |
| Sign-off | Named reviewer, date, run ID, Open WebUI version, the checklist in section 11 | Sign-off record |

### 10.4 Audit trail

Every human decision above records who, when, which rows, and why. Reviewer identities may be pseudonymous in the public repo.

## 11. Reporting and sign-off

Before a model is promoted or the filter is enabled in production:

1. Run directory contains manifest, metrics, thresholds, package versions, plots, and the dataset fingerprint.
2. Headline table with 95% bootstrap CIs on `test`, `adversarial_test`, and the LoRA held-out sets.
3. Recall@FPR table and the chosen operating point with its budget.
4. Reason-head table with undefined reasons marked as such.
5. Slice table, benign-lookalike FPR on its own line, worst slice called out.
6. Paired comparison against the Phase 2 winner and, once its eval exists, against the LoRA.
7. Error-analysis summary.
8. Latency table at batch 10, with the p95 target met or the miss explained.
9. Filter unit and integration tests passing against a named Open WebUI version.
10. Model card: intended use (retrieval-path scanning), what it is not for (typed chat prompts; that is the LoRA's job), training data and gaps, limitations.
11. Sign-off by a reviewer who did not train the model.

### Backlog order once Phase 3 exists

1. Bootstrap CIs and recall@FPR (sections 3, 4).
2. Benign-lookalike slice and indirect-injection slice (3.4).
3. Filter unit tests and the RAG-template contract test (section 9).
4. Latency measurement (section 8).
5. LoRA eval run and paired comparison (section 4).
6. Annotation rubric, agreement study, gold set (section 10).
7. Multi-seed, ablations (section 2).
8. Red-team set and shadow mode (10.3).
9. Integrated gradients and span agreement (sections 7, 10.2).
10. CI, model card, datasheet.

## 12. After deployment

- Shadow mode precedes enforcement (10.3).
- Drift monitoring on the score distribution and on chunk-shape features per knowledge base; alert on a shift.
- Every dropped chunk is logged with its score, reason, attributed span, source ID, and knowledge base, so an operator can review and restore.
- Operator overrides become labeled rows with provenance for the next data version.
- Re-verify the filter against each Open WebUI upgrade: the `<source>` rendering and the `request` hook are internal details that can change.
