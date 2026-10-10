# Model Card: prompt-injection-nn, v4 data build

**Runs:** `phase2-v4-7f42d9d83806880d` (TF-IDF logistic regression, XGBoost) and `phase3-cnn-bigru-v4-7f42d9d83806880d` (CNN + BiGRU, two heads). **Shipped backend:** `ensemble_mean`, the mean of the TF-IDF and CNN calibrated probabilities, as recommended by the Phase 3 gate.
**Trained:** 2026-10-10 on an NVIDIA DGX Spark (GB10, CUDA 13, PyTorch 2.10, scikit-learn 1.7, XGBoost 3.4). Single seed (42).
**Data fingerprint:** `7f42d9d83806880d95a44645173227c6008b5291538e8af40eaae0ed37cf0486` (v4; see the Phase 1 report under `reports/`).
**Status:** research candidate cleared for a shadow-mode pilot. Not yet run against live traffic. See `../README.md` for the full progression.

## Intended use

Score a chunk of text that is about to be placed in an LLM's context, retrieved knowledge-base passages, uploaded file sections, web-search results, tool results and agent messages, and return `SAFE` or `INJECTION: <reason>` with a calibrated probability. Consumed by the Open WebUI request-hook filter in `filter/`, which drops flagged chunks from the context and the citations. Typed chat prompts are out of scope; the prompt-injection LoRA in the `safety` repo handles those.

## Out of scope

- Judging typed user prompts (chat-shaped text). Measured for comparison only; recall at a 1% false-positive budget on chat rows is 55 to 64%.
- Languages other than English.
- Images, audio, or binary attachments.
- Any use where a dropped benign chunk is not an acceptable failure mode without a human review path.

## Training data

Twenty Hugging Face sources (licences on each card; the v4 datagen notebook header lists every source, licence, row count, label, reason and shape, and every rejected source with the reason). Pipeline: schema normalisation, exact and MinHash near-duplicate deduplication, held-out isolation of the LoRA comparison sets, grouped 80/10/10 splits balanced by label, source, reason and shape, paired clean-and-injected sources kept together by group key, train-only adversarial augmentation.

| Split | Rows | Injection | Safe |
|---|---:|---:|---:|
| train | 82,800 | 39,315 | 43,485 |
| val | 9,367 | 3,931 | 5,436 |
| test | 9,456 | 3,904 | 5,552 |
| adversarial_test | 3,904 | 3,904 | 0 |
| lora_holdout | 2,086 | 1,141 | 945 |

Test rows by shape: chat 6,310; document 2,615; agentic 531. Reason labels are map-derived from source or attack family, not human annotated; report, do not gate. Template twin rate (share of test attacks whose first six words also open a training attack): chat 32%, document 59% (LLMail contestants reuse attack bodies; Kubernetes manifests share a JSON header), agentic 19%; per source in the manifest.

## Architectures

**TF-IDF logistic regression.** Word 1-2-grams (200k features) and character 3-5-grams within word boundaries (300k), cased, sublinear TF; liblinear logistic regression, C=2, balanced class weights; sigmoid calibrator fitted on validation. Model files 22 MB.

**CNN + BiGRU.** Cased punctuation-preserving tokenizer (letter runs, digit runs, symbol runs), 50,000-token train-only vocabulary, 512-token window. 128-d embeddings, one 1-D convolution (128 channels, kernel 3), one bidirectional GRU (128 per direction) on packed sequences, masked max-pooling, shared dense layer, a binary injection head and a nine-way reason head. AdamW, lr 2e-3, batch 256, BF16 on CUDA, early stopping on validation log loss (patience 4), best epoch 3 of 7. 6,681,610 parameters, 26.7 MB. Padding-invariant: a text scores identically alone or in any batch (a padding leak found by the package tests on 2026-10-10 was fixed before this run).

**Ensemble.** Mean of the two calibrated probabilities; the reason comes from the CNN's reason head.

## Evaluation

Validation drives selection; `test`, `adversarial_test` and `lora_holdout` are opened once afterwards. 95% percentile bootstrap intervals, 1,000 resamples.

### Locked test, F1-max threshold from validation

| Model | Precision | Recall | F1 | ROC-AUC | Avg precision | FPR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF logistic regression | 0.873 | 0.928 | 0.900 | 0.975 | 0.968 | 9.5% |
| CNN + BiGRU | 0.868 | 0.927 | 0.896 | 0.976 | 0.969 | 9.9% |
| **Ensemble (mean)** | **0.900** | 0.926 | **0.913** | **0.981** | **0.975** | **7.2%** |

CNN test F1 interval 0.889 to 0.903, recall 0.918 to 0.934, FPR 9.1% to 10.7%.

### Recall at a false-positive budget, threshold chosen within the slice

| Slice | Model | FPR 1% | FPR 0.5% | FPR 0.1% |
|---|---|---:|---:|---:|
| all test rows | TF-IDF | 0.702 | 0.630 | 0.378 |
| all test rows | CNN | 0.704 | 0.648 | 0.465 |
| all test rows | Ensemble | **0.750** | **0.686** | **0.507** |
| document + agentic (the product's input) | TF-IDF | 0.908 | 0.867 | 0.843 |
| document + agentic | CNN | 0.954 | 0.930 | 0.890 |
| document + agentic | Ensemble | **0.961** | **0.943** | **0.909** |
| chat | Ensemble | 0.636 | 0.544 | 0.319 |

### The gate (PLAN.md section 4)

Judged on the document and agentic test rows at all three budgets, plus adversarial recall within one point of the baseline. Both the CNN and the ensemble clear it; the ensemble is recommended and is the default backend. Written to `gate.json` in the Phase 3 run.

### Retrieval operating points

Thresholds chosen on validation document and agentic rows only (what the filter sees), reported on the locked test document and agentic rows. These are the operating points the filter ships with (`retrieval_fpr_*` budgets).

| Model | Budget | Threshold | Test recall (document + agentic) | Test FPR (document + agentic) |
|---|---|---:|---:|---:|
| **Ensemble** | retrieval_fpr_0.01 (default) | 0.455 | **0.971** | 1.55% |
| Ensemble | retrieval_fpr_0.005 | 0.518 | 0.961 | 1.03% |
| Ensemble | retrieval_fpr_0.001 | 0.723 | 0.904 | 0.06% |
| CNN | retrieval_fpr_0.01 | 0.407 | 0.956 | 1.23% |
| CNN | retrieval_fpr_0.001 | 0.857 | 0.897 | 0.13% |
| TF-IDF | retrieval_fpr_0.01 | 0.658 | 0.926 | 1.42% |
| TF-IDF | retrieval_fpr_0.001 | 0.893 | 0.840 | 0.06% |

A threshold chosen for 1% on validation lands at 1.0 to 1.6% on test: the budget is approximate at the 1% level and tight at 0.1%. An operator who needs a hard 1% should take the 0.5% setting.

### Slices by shape, F1-max threshold

| Shape | Rows | TF-IDF recall / FPR | CNN recall / FPR | Ensemble recall / FPR |
|---|---:|---:|---:|---:|
| document | 2,615 | 0.970 / 4.9% | 0.972 / 2.4% | 0.978 / 1.8% |
| agentic | 531 | 0.963 / 2.6% | 0.946 / 0.0% | 0.966 / 0.4% |
| chat | 6,310 | 0.900 / 11.4% | 0.899 / 13.0% | 0.891 / 9.4% |

### Adversarial test (3,904 obfuscated test attacks)

Recall: TF-IDF 0.966, CNN 0.965, ensemble 0.967. Weakest variant for every model is whitespace noise (about 0.92 to 0.93).

### Reason head (CNN, test true positives, 3,617 rows)

Accuracy 0.943, top-2 0.983, macro-F1 over the nine reasons 0.811. Strong: Embedded Instruction (P 0.98 / R 0.97), Jailbreak Technique (0.94 / 1.00), Fake System Injection (1.00 / 0.84). Weak: Privilege Escalation (0.79 / 0.54), Override Attempt (0.55 / 0.74), Role-Play Bypass (0.88 / 0.60, 25 rows), Chain-of-Instructions (0.83 / 0.77, 13 rows). Reasons are reported, not gated.

### The LoRA's held-out sets (chat-shaped, 2,086 rows, CNN only)

Accuracy 0.889, false positives 13.3%, false negatives 9.2%, recall at FPR 1% 0.578. The LoRA's own eval has not been run, so there is no head-to-head yet.

### Latency, CPU

Measured by `scripts/benchmark_latency.py` on the DGX Spark's CPU (aarch64, 20 cores, package default of 8 torch threads), ten real document chunks per call, 200 calls after warm-up.

| Backend | Ten 1,000-character chunks (Open WebUI default size) p50 / p95 | Ten stored test chunks, 118 to 891 tokens, several windows each, p50 / p95 |
|---|---:|---:|
| TF-IDF | 4.0 / 4.0 ms | 8.3 / 8.4 ms |
| CNN | 12.9 / 16.2 ms | 31.7 / 34.5 ms |
| **Ensemble** | **24.8 / 27.0 ms** | 43.9 / 47.4 ms |

Target: p95 under 100 ms for ten chunks. Met with a wide margin. Without thread pinning the CNN's p95 on the stored chunks was 110 ms, which is why the package pins threads (half the cores, at most 8) and the filter caps windows per chunk at 4.

## Known failure modes and caveats

- **Chat-shaped text is weak** and out of scope: at a 1% budget the ensemble catches 64% of chat attacks. Keep the LoRA on the typed-prompt path.
- **Benign text that argues or instructs** is the main false-positive source: adversarial-but-unharmful WildGuardMix prompts are flagged about a third of the time at the F1 point, NotInject trigger-word prompts 11 to 14%.
- **Reason labels are weak** (map-derived), and four reasons have fewer than 40 test rows.
- **Template reuse** inflates document-shaped numbers somewhat (twin rate 59%); near-duplicate grouping at 0.80 removes close copies, shorter reuse remains and is reported.
- **Mostly synthetic sources.** LLMail, in-the-wild prompts and Kubernetes manifests are real; the rest are generated. Novel, hand-written attacks are not represented; a red-team set is open.
- **Single seed, no multi-seed variance.** Intervals reflect test-set sampling only.
- **Package scores in fp32 on CPU; the notebook in bf16 on CUDA.** Parity test: TF-IDF exact, CNN recall and FPR within 0.0013.

## Artifacts

`models/phase2-v4-7f42d9d83806880d/`: `tfidf_word.joblib`, `tfidf_char.joblib`, `logreg.joblib`, `logreg_calibrator.joblib`, `xgboost.json`, `thresholds.json` (all-rows and retrieval budgets), `metrics.json`, `probabilities_{val,test,adversarial_test}.parquet`, `run_manifest.json`.
`models/phase3-cnn-bigru-v4-7f42d9d83806880d/`: `model_state.pt`, `vocab.json`, `tokenizer.json`, `config.json`, `thresholds.json` (per candidate, all-rows and retrieval budgets, recommended backend), `gate.json`, `metrics.json` (includes every candidate's tables), `probabilities_*.parquet`, `training_history.json`, `eval_wildguard_test.json`, `run_manifest.json`.

Artifacts are not in git. The executed notebooks and the report folders under `reports/` are the portable record.
