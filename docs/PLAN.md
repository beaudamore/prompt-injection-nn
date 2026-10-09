# Prompt Injection Classifier — Specialized NN Training Plan

**Hardware:** DGX Spark (GB10, 128 GB unified memory) primary; Apple Silicon or CPU supported.
**Project root:** `prompt-injection-nn/` (any checkout; see `PROCESS.md` for path resolution).

Implementation status and measured results are tracked in `PROCESS.md` and the README.

---

## 1. Scope

**Binary classification: prompt-injection vs. benign text, with a reason label on positives.**

The unit of classification is a *chunk of text that is about to be placed in a model's context*:
a retrieved knowledge-base passage, a web-search snippet, a section of an uploaded file, or a
tool result. These are documents, not chat turns, so the model must handle:

- Direct instruction-hierarchy attacks: override, fake system messages, meta-commands, privilege escalation.
- Jailbreak and role-play framing.
- Delimiter confusion and fake structure (`###`, `---`, `<|im_start|>`, fake `<source>` tags).
- **Indirect injection**: instructions embedded inside otherwise normal prose, intended for the model rather than the reader.
- Benign lookalikes: documentation *about* prompt injection, quoted attack text in a security report, fiction containing "ignore what I said".

Out of scope for phase 1: content safety (the Aegis taxonomy is a separate classifier), multilingual beyond what the sources carry, images and QR codes.

### Output contract

Identical to the LoRA and the live filter so either can serve the same call site:

```text
SAFE
INJECTION: <reason>
```

Nine valid reasons: Override Attempt, Fake System Injection, Meta-Command Injection, Jailbreak
Technique, Delimiter Confusion, Privilege Escalation, Embedded Instruction, Role-Play Bypass,
Chain-of-Instructions.

The Python API additionally returns the calibrated probability and the reason-head distribution so callers can set their own threshold.

## 2. Data Strategy

All public data is acquired exclusively from Hugging Face. Same discipline as `email-classifier-nn`: schema normalization, quality gates, exact and MinHash near-duplicate deduplication, grouped leakage-safe splits, a fingerprinted manifest, and a locked adversarial evaluation set.

### 2.1 Sources carried over from the LoRA recipe (verified in `safety/notebooks/prompt_injection_lora_qwen38_27b.ipynb`)

| Source | HF location | Role | Notes |
|---|---|---|---|
| Nemotron Safety Guard v3 | `nvidia/Nemotron-Safety-Guard-Dataset-v3` | `jailbreaking`-tagged rows → INJECTION, reason Jailbreak Technique | Train split for training; the held-out `jailbreaking` rows form part of the eval set, matching the LoRA eval notebooks. |
| WildGuardMix | `allenai/wildguardmix`, config `wildguardtrain` | adversarial + `prompt_harm_label == harmful` → INJECTION; benign → SAFE | Subcategory-to-reason map copied from the LoRA notebook. `wildguardtest` is held out for eval. |

These two gave the LoRA 11,872 training rows: 4,500 WildGuardMix adversarial, 3,500 Nemotron jailbreaking, 4,500 WildGuardMix benign. Reason distribution was heavily Jailbreak Technique (7,266), with Privilege Escalation (644) and Meta-Command Injection (90). The other six reasons had no training rows. That gap is the first thing the NN build must close.

### 2.2 Candidate sources to verify before use

Verified on Hugging Face on 2026-10-09 (schema, license, splits, and a sample of rows). The datagen notebook uses the accepted ones.

| Candidate | What it adds | Status |
|---|---|---|
| `deepset/prompt-injections` | Short direct-injection vs. benign prompts, en/de | **accepted.** 546 train + 116 test rows, `text` + `label` (1 = injection), apache-2.0. Positives mapped to Override Attempt (source-level, weak). |
| `jackhhao/jailbreak-classification` | DAN-style jailbreak vs. benign role-play prompts | **accepted.** 1,044 train + 262 test rows, `prompt` + `type`, apache-2.0. Jailbreak rows mapped to Jailbreak Technique. Overlaps Nemotron templates; handled by the template grouping. |
| `geodesic-research/bipia` | BIPIA: instructions embedded in emails, tables and code at a known position, the RAG case | **accepted.** 41,250 rows, cc-by-sa-4.0; `context` already contains `attack_str` at `position`; 125 distinct attack strings over about 850 base documents. Injected rows grouped by attack string, 48 rows per string; clean documents recovered by removing the attack string. Mapped to Embedded Instruction. |
| `wikimedia/wikipedia` `20231101.en` | Prose-shaped SAFE negatives | **accepted.** cc-by-sa-3.0, streamed; first 1,500 articles split into paragraphs of 200+ characters, capped at 6,000. |
| `xTRam1/safe-guard-prompt-injection` | 10k injection vs. safe prompts | **rejected:** no license on the dataset card. |
| `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` | Indirect injection inside tool results, agentic | **deferred:** cc-by-4.0, but a raw `train.jsonl` of agent traces with no documented schema. Revisit for a later data version. |

### 2.3 Generated hard examples

The under-represented reasons (Fake System Injection, Delimiter Confusion, Embedded Instruction, Role-Play Bypass, Chain-of-Instructions, Override Attempt) will need generated rows. Follow the Phase 4 lifecycle from `email-classifier-nn/docs/PLAN.md` §4.1: generated rows carry `synthetic=true`, `attack_type`, `generation_model`, `prompt_version`; they are gated, deduplicated against real data, and never used for threshold or model selection. A separate generated adversarial set is locked for evaluation only.

Benign lookalikes must be generated alongside: security write-ups quoting attacks, fiction, documentation that explains "system prompt". Without them the model learns keyword spotting.

### 2.4 Pipeline (datagen notebook)

1. **Download** sources into `data/source-raw/`.
2. **Normalize** to `text, label, reason, source, synthetic, attack_type`.
3. **Quality gates**: drop empty or near-empty text, label sanity, reason must be one of the nine or empty for SAFE.
4. **Dedupe**: SHA-256 exact; MinHash LSH near-dup over word shingles. The LoRA sources overlap in places and WildGuardMix contains near-paraphrases.
5. **Grouped splits** 80/10/10, grouped by near-dup cluster and by source template where identifiable, stratified by label and source.
6. **Features** for the Phase 2 baseline: delimiter and tag counts, imperative-verb density, second-person-to-model phrases, zero-width and homoglyph counts, length, non-ASCII ratio, "ignore/override/system/instruction" lexicon hits.
7. **Adversarial augmentation** (train only): homoglyph substitution, zero-width injection, delimiter wrapping, case and whitespace noise. A separately generated fixed set is the locked adversarial test.
8. **Verify and save** `data/source-clean-v1/{train,val,test,adversarial_test}.parquet` + `manifest.json` with fingerprint, counts, feature order, reason distribution per split.

### 2.5 Known risks

- Chat-shaped vs. document-shaped bias (see 2.2, last row).
- **Document reuse inside BIPIA (found on v1).** BIPIA repeats about 850 base documents under 125 attack strings. Grouping by attack string alone lets a base document appear attacked in train and clean in test, and the model learns the document. v2 partitions base documents and attack strings jointly, so test injected rows have an unseen document and an unseen attack string, and adds structured benign documents from outside BIPIA. Found on v2: BIPIA has only 182 distinct base documents once whitespace is normalized, v2's document key was not normalized, and injected rows outnumbered clean ones 40:1 in train. v3 keys documents by normalized text and caps injected rows at three per document.
- **Template reuse inside the jailbreak corpora (found on v1).** A 12-word leading-template key left a 42% six-word twin rate between train and test. v2 uses a shorter template key and a lower near-duplicate threshold, and reports the twin rate as the regression indicator.
- Reason labels are weak: most come from a subcategory map, not human annotation. Report reason accuracy separately and do not gate on it in phase 1.
- Live attack strings in the data. Never execute or paste them into a real model session outside the notebooks.

## 3. Model Architecture

**Stage A, baselines** (`notebooks/training/phase_2_baselines.ipynb`): logistic regression on TF-IDF (word and char n-grams) and XGBoost on the engineered features. Calibrated on validation only; thresholds by validation F1. These set the bar.

**Stage B, neural model** (`notebooks/training/phase_3_text_cnn_bigru.ipynb`): the `email-classifier-nn` Phase 3 architecture, a 1D-CNN over a train-only vocabulary feeding a bidirectional GRU with masked max-pooling, with **two heads**: a binary injection head and a nine-way reason head. Reason loss is masked to positive rows. Reason is reported only when the binary head fires. Target: a few hundred thousand parameters, sub-10 ms per chunk on CPU in a batch.

Tokenization must preserve the things attacks are made of: punctuation runs, angle brackets, pipe characters, and case. Do not lowercase blindly; keep a cased vocab or add case-shape tokens.

**Not doing:** an LLM backbone. The LoRA already exists for that.

## 4. Training and Evaluation

- **Metrics**: precision, recall, F1, ROC-AUC, average precision, Brier, log loss, confusion counts, FPR. Reason-head accuracy and per-reason precision/recall reported separately.
- **Gates for promotion**: beat the Phase 2 winner on injection recall at an FPR low enough for document traffic. FPR matters more here than in the chat filter: dropping a benign chunk silently degrades answers. Target FPR on the benign-document negatives must be reported on its own.
  **Made concrete after the v1 runs (2026-10-09):** the gate is recall at FPR budgets of 1%, 0.5% and 0.1% on the locked test split, the false-positive rate on each document-shaped SAFE source reported separately, and adversarial recall. F1 at the max-F1 threshold is reported but is not a gate: on v1 the TF-IDF baseline matched the neural model on F1 while the neural model caught nearly three times as many attacks at the 0.1% budget.
- **Comparability with the LoRA**: evaluate on exactly the LoRA's held-out sets, WildGuardMix `wildguardtest` plus Nemotron held-out `jailbreaking`, and write `eval_*.json` with the same fields the LoRA eval notebooks write. Note that no LoRA `eval_*.json` exists yet; running the LoRA eval is the only way to get a real head-to-head.
- **Adversarial robustness** on the locked obfuscated set from day one.
- **Explainability**: integrated gradients over tokens, so a reviewer can see which span triggered the verdict.

## 5. Open WebUI Integration (no patch)

Verified against the Open WebUI source at 0.11.4 (`backend/open_webui/utils/middleware.py`).

**Where retrieval meets the pipeline.** Inside `process_chat_payload`, `inlet` filters run first and see only the user message and file references. Retrieval runs after that, for attached files, knowledge bases, and web-search results alike, and every hit lands in one `sources` list. Each chunk is rendered as `<source id="n" name="...">chunk</source>`, wrapped in the RAG template, and appended to the system message (or prepended to the user message). The same list is stored on `metadata["sources"]` for citations. Then the `request` filter hook runs, with the fully assembled body and `__metadata__`. That hook exists from 0.11.2.

**The filter is a `request` hook.** It:

1. Finds every `<source ...>...</source>` block in the system and user messages.
2. Classifies the bodies in one batch.
3. Removes flagged blocks from the message text and the matching entries from `metadata["sources"]`, so they neither reach the model nor appear as citations.
4. Emits a status event naming the dropped source, and records the event the way the existing block-aware filters do.
5. Continues the turn. Dropping a chunk is the action; the turn is not blocked.

Covered by this hook because they all flow through the same sources list: knowledge-base retrieval, uploaded files (chunked or full-context), web search, and tool results injected through the native tool-call loop (the `request` hook re-runs inside that loop).

**Not covered.** Text typed directly into the prompt never passes through retrieval; the existing inlet filter stays in place for that. Ingest time (knowledge upload) has no filter hook; scanning at upload would need an Open WebUI patch or an external pre-upload scanner, both out of scope for now.

**Latency budget.** Top-k is typically 3 to 10 chunks, more with a reranker. The hook is on the critical path of every turn. A batch of ten 512-token chunks on CPU must stay well under 100 ms.

## 6. Directory Layout

```text
prompt-injection-nn/
  docs/PLAN.md, PROCESS.md
  data/source-raw/, source-clean-v1/, synthetic/     (gitignored)
  notebooks/datagen/phase_1_prompt_injection_datagen_v2.ipynb   current
  notebooks/training/phase_2_baselines.ipynb
  notebooks/training/phase_3_text_cnn_bigru.ipynb
  notebooks/eval/phase_4_eval_against_lora_holdouts.ipynb
  src/prompt_injection_nn/                         package: tokenizer, model, classify API
  filter/safety_filter_rag_injection_scan.py       Open WebUI request-hook filter
  models/<run-id>/                                 (gitignored)
```

## 7. Phase Roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1 | Verified source table (2.1 + 2.2 resolved), datagen notebook, fingerprinted v1 splits, locked adversarial set | Manifest written, no cross-split leakage, reason distribution reported per split |
| 2 | TF-IDF logistic regression and XGBoost baselines | Calibrated, thresholded, locked test and adversarial numbers recorded |
| 3 | CNN+BiGRU two-head model | Beats Phase 2 on injection recall at acceptable document-FPR; eval written in the LoRA's format |
| 4 | Generated hard examples for the six empty reasons and benign lookalikes; controlled retrain | Promote only if the locked gates improve; otherwise keep the real-data model |
| 5 | `src/` package and the Open WebUI request-hook filter; latency measured on CPU | Filter verified against a named Open WebUI version; batch-of-ten latency recorded |
