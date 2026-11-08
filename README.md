# Prompt Injection Classifier — Specialized Neural Network

A compact text classifier that flags prompt-injection content without calling an LLM. Built to
scan retrieved RAG chunks, web-search results, and uploaded files inside Open WebUI before they
reach the model, and to be importable from any Python app as a plain function.

Sibling of [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn): same
pipeline discipline (Hugging Face sources only, fingerprinted dataset builds, leakage-safe splits,
baselines before the neural model, locked holdouts), same multi-device notebooks.

Replaces the per-call LLM round trip of the existing
[prompt-injection LoRA](https://github.com/beaudamore/safety) where latency matters. Until
2026-11-05 the LoRA stayed the detector for typed chat prompts and this repo covered only the
retrieval path, where a single turn can carry ten or more chunks. Since 2026-11-05 the repo also
ships a filter for typed prompts that uses the same guard (see the update below).

**Status (2026-10-10):** four data builds; Phases 2 to 4 run on each; every run reported under `reports/`.
**On v4 the neural model clears the promotion gate on the product's input.** Judged on document and
agentic test rows at false-positive budgets of 1%, 0.5% and 0.1%, both the CNN + BiGRU and its mean
ensemble with the TF-IDF baseline beat the baseline at every budget; the ensemble is the shipped
backend (recall 0.961 / 0.943 / 0.909 on those rows, baseline 0.908 / 0.867 / 0.843). The `src/`
package, the Open WebUI request-hook filter (shadow mode by default), unit and filter tests and CI
exist. Not yet done: a shadow-mode pilot on live traffic, the operator's false-positive budget, the
LoRA head-to-head, and human label review. New to the project? Read `docs/LEARNING.md` first; the
numbers are in `docs/MODEL_CARD.md`; design in `docs/PLAN.md`; how to run in `docs/PROCESS.md`.

**Update (2026-11-05): two filters, one guard service.** `filter/` now holds two Open WebUI filters
that both send text to the prompt-injection guard (`Horizon-Labs/prompt-injection-guard-base`) over
`POST /classify`. The guard service is vLLM on the DGX Spark (port 8009) or the MLX server on the Mac
at `http://host.docker.internal:8194` (`serve/mlx/server.py`, see `serve/README.md`); the only filter setting
that differs is `guard_url`. The vLLM stack is not in this repo; it lives on the DGX Spark as
`compose/live/vllm-prompt-injection-guard.yml`.

- `filter/safety_filter_rag_injection_scan.py` ("RAG Injection Scan") was renamed
  `filter/safety_filter_prompt_injection_retrieved_content.py`, titled "Prompt Injection Protection
  (Retrieved Content)", version 0.4.0. Behaviour was unchanged by the rename; 0.4.1 the same day reworded the notice that replaces a flagged tool result (see `filter/README.md`).
- 0.5.0 (2026-11-05): the same filter also scans the files attached to a chat when a message is sent (a new `inlet` hook next to the `request` hook): the whole stored text of each file, and in enforce mode a flagged file blocks the turn and is named to the user. Deleting the flagged file is a valve, off by default. A file is scored only when the sending user is an admin, owns it or has read access by Open WebUI's own rules; deleting needs write access. The upload scan has not been run live (checked by reading the `v0.13.0` source only), and the copy installed in Open WebUI is still 0.4.1 until the owner has it updated.
- Both filters (2026-11-05): text is normalized with Unicode NFKC and split into pieces of at most 2,000 characters (6,000 before), because one input over the guard's 8,192-token limit makes it refuse the whole request and the filters fail open. Measured and remaining limits: `filter/README.md`, "Chunk size".
- New: `filter/safety_filter_prompt_injection_chat.py`, "Prompt Injection Protection (Chat)", version
  3.0.0. It is the inlet filter for typed prompts from `openwebui-safety-filters` with its LLM/LoRA
  detector replaced by the guard; blocking, violation logging, lockout and the lockout webhook are kept.

Live (2026-11-05): the retrieval filter ran in Open WebUI 0.13.0 in shadow mode against the MLX
server and flagged the planted test document (`filter/test_files/`). On that model the file reached the
model as a native tool result (built-in `view_file`), not as `<source>` chunks, so it was scored whole.

Not done: the guard has **not been measured on typed (chat-shaped) prompts**; the retrieval filter has
not been run live in enforce mode and its scan of attached files at send (0.5.0) has not been run live at all; the chat filter is not installed anywhere yet (it was checked by
reading the `v0.13.0` source only) and ships in shadow mode. Details: `filter/README.md`.

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

### Phase 2 and Phase 3 on v4 (2026-10-10, locked test split, DGX Spark CUDA)

Three candidates are scored on identical rows: the Phase 2 TF-IDF baseline, the Phase 3 CNN + BiGRU
(padding-invariant since 2026-10-10), and their mean ensemble.

| Model | F1 | Recall | FPR | Recall at FPR 1% | 0.5% | 0.1% | Adversarial recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF logistic regression (Phase 2 winner) | 0.900 | 0.928 | 9.5% | 0.702 | 0.630 | 0.378 | 0.966 |
| CNN + BiGRU, two heads (Phase 3) | 0.896 | 0.927 | 9.9% | 0.704 | 0.648 | 0.465 | 0.965 |
| **Ensemble, mean of both (shipped)** | **0.913** | 0.926 | **7.2%** | **0.750** | **0.686** | **0.507** | **0.967** |

On the product's rows, test document + agentic (3,146 rows, 1,590 attacks), recall at FPR budget
with the threshold chosen within those rows:

| Model | FPR 1% | FPR 0.5% | FPR 0.1% |
| --- | ---: | ---: | ---: |
| TF-IDF logistic regression | 0.908 | 0.867 | 0.843 |
| CNN + BiGRU | 0.954 | 0.930 | 0.890 |
| **Ensemble** | **0.961** | **0.943** | **0.909** |

What the filter ships with: thresholds chosen on validation document + agentic rows. At the default
`retrieval_fpr_0.01` the ensemble reaches recall 0.971 at 1.55% FPR on the test document + agentic
rows; at `retrieval_fpr_0.001`, 0.904 at 0.06%. Latency for ten 1,000-character chunks on the
Spark's CPU: ensemble p95 27 ms, CNN 16 ms, TF-IDF 4 ms.
Slices at the F1-max threshold: document recall / FPR, ensemble 0.978 / 1.8% (TF-IDF 0.970 / 4.9%);
agentic 0.966 / 0.4% (0.963 / 2.6%); chat 0.891 / 9.4% (0.900 / 11.4%). Reason head: all nine reasons
defined, accuracy 0.943, macro-F1 0.811. Every number with its interval: `docs/MODEL_CARD.md`.
Phase 4 on the LoRA held-out sets (chat-shaped): accuracy 0.889, FP 13.3%, FN 9.2%.

### What the v4 numbers mean

1. **The gate is cleared, on the rows that matter.** `PLAN.md` section 4 was made concrete: beat the
   baseline on document and agentic rows at every budget, keep adversarial recall. Both neural
   candidates do; the ensemble does by the widest margin and is the recommended backend in `gate.json`.
2. **The target is met on retrieval text.** The working target was 90% recall at 1% FPR on document
   and agentic data; the ensemble reaches 96% at 1% and 91% at 0.1%. The baseline alone would have
   shipped at 91% / 84%.
3. **Chat remains weak and out of scope** (64% recall at 1% FPR for the ensemble). The LoRA keeps the
   typed-prompt path. The Phase 4 holdout numbers, which are chat-shaped, moved down slightly from v3
   for the same reason: v4 put the training weight on documents.
4. **The baseline's false positives come from structured documents** (clean BIPIA documents 83%,
   Kubernetes manifests 16%) that a bag of words reads as attack-shaped; the CNN and the ensemble
   handle them (17% and 5% for the CNN).
5. **What the tests caught.** The package's batch-versus-single test found that the BiGRU read the
   padding, so scores depended on batch composition. The forward pass now uses packed sequences and the
   model was retrained; numbers above are from the fixed model. Parity between the package (fp32,
   CPU) and the notebook (bf16, CUDA): TF-IDF exact, CNN within 0.0013.
6. **Still open before enforcement:** a week of shadow mode on a real instance, the operator's
   false-positive budget, the LoRA eval for the head-to-head, a red-team set of hand-written attacks,
   and the human label review (`docs/EVALUATION.md` sections 10 and 11).

---

## What it produces

**Deploying the filters needs no training and no model files in the Open WebUI container.** Run the
guard service, add the filter file(s) under Admin, Functions, set `guard_url`. Step by step in
`filter/README.md`.

**Which model the filter runs (2026-10-10).** Shadow mode on a live instance produced false positives on
web-page boilerplate and API JSON dumps that the in-house models had never seen. A survey of published
prompt-injection classifiers (`docs/LEARNING.md`) found one built for exactly this input and trained on
the diverse data we lack: `Horizon-Labs/prompt-injection-guard-base` (ModernBERT, 308M, Apache-2.0). It
scores those false positives at 0.007 and 0.05 and every planted attack we tried at 0.997 or above. It
is now a backend of the package (`hf_guard`, CPU, and `hf_guard_http`, served on the GPU by the
`vllm-prompt-injection-guard` stack) and is judged on our locked v4 rows by the same gate before it
ships; the in-house CNN supplies the reason label. Speed on the GPU is tens of milliseconds per batch.

**The guard on a Mac (2026-11-05).** `serve/mlx/server.py` serves the same model with MLX on Apple
Silicon, answering the same `POST /classify` under the same served name. On a Mac Studio M1 Ultra
(48 GPU cores, 128 GB) its scores match Hugging Face transformers in float32 to within 1.71e-07 (CPU)
and 2.33e-07 (MPS) in P(INJECTION) over 15 check texts, with identical token counts. Ten chunks of
1,200 characters in one request: p50 74.3 ms over HTTP, against 83.6 ms for torch on MPS and 336.2 ms
for torch on CPU in process. Setup, the three corrections it makes to a plain `mlx-embeddings` load,
all measurements and known limits: `serve/README.md`.

| Artifact | Purpose |
|---|---|
| `src/prompt_injection_nn` | `load_detector()`, `classify(text)`, `classify_batch(texts)` returning a `Verdict` (label, calibrated probability, reason from the nine-reason taxonomy, the operating threshold). Three backends: `cnn_bigru`, `tfidf_logreg`, `ensemble_mean`; the default is the Phase 3 gate's recommendation. Budgets `retrieval_fpr_0.01` (default), `_0.005`, `_0.001` are thresholds chosen on validation document and agentic rows. Long inputs are scored in overlapping windows. CLI: `prompt-injection-nn "text"`. See `src/README.md`. |
| `filter/safety_filter_prompt_injection_retrieved_content.py` | "Prompt Injection Protection (Retrieved Content)", 0.5.0 (renamed 2026-11-05 from `safety_filter_rag_injection_scan.py`; 0.4.1 rewords the notice that replaces a flagged tool result; 0.5.0, 2026-11-05, adds the `inlet` hook). One Open WebUI filter with two hooks. `request` hook: finds every `<source>` block retrieval appended and every native tool result, sends the text to the guard service in batches of up to 64 pieces, and in enforce mode removes flagged blocks from the context and from `metadata["sources"]` so they neither reach the model nor appear as citations. `inlet` hook (0.5.0): when a message is sent, scores the whole stored text of every file attached to the chat that the sending user may read; in enforce mode a flagged file blocks the turn and is named to the user, and `delete_flagged_files` (off by default) also deletes it; no violation count, lockout or webhook. Shadow (log-only) mode by default, one `log_only` valve for both hooks. Fails open. Self-contained: one file, no package and no model files in the container. `request` hook: verified live against Open WebUI 0.11.4; run live in 0.13.0 in shadow mode on 2026-11-05 (tool-result path); enforce mode not run live. `inlet` hook: checked against the `v0.13.0` source by reading it; not run live. The copy installed in Open WebUI is still 0.4.1 until the owner has it updated. See `filter/README.md`. |
| `filter/safety_filter_prompt_injection_chat.py` | "Prompt Injection Protection (Chat)", 3.0.0 (new 2026-11-05). Open WebUI `inlet`-hook filter for the prompt the user typed: scores it with the same guard service; in enforce mode blocks the turn, logs the violation to a knowledge base, counts it, locks the account at the limit (admins exempt) and can post a lockout webhook. Shadow (log-only) mode by default because the guard has not been measured on typed prompts. Fails open. Checked against the Open WebUI `v0.13.0` source by reading it; not run in a live instance. See `filter/README.md`. |
| `serve/mlx/server.py`, `serve/check_guard_server.py` | The guard service on Apple Silicon (new 2026-11-05): `Horizon-Labs/prompt-injection-guard-base` on `mlx-embeddings`, `POST /classify` in vLLM's format, port 8194, no torch. The checker tests any `/classify` server for contract, batch invariance, parity (against transformers or another server) and latency. See `serve/README.md`. |
| `filter/test_files/` | `planted_injection_vendor_guide.md`, a fictional vendor guide with one planted instruction, for live tests of the retrieval filter. See `filter/README.md`, Test file. |
| `tests/` | Tokenizer, detector contract, behaviour of both filters (`test_filter.py`, `test_filter_uploads.py`, `test_filter_chat.py`; stub scorer and stub `/classify` server), and artifact parity against the notebooks' `metrics.json`. `pytest` runs anywhere; parity needs the model folders. CI in `.github/workflows/tests.yml`. |
| `models/<run-id>/` | Versioned weights, vocab, thresholds per budget, metrics, per-row probabilities, `gate.json`, and a run manifest tied to the dataset fingerprint. Gitignored; rebuilt by running the notebooks. |

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
│   ├── MODEL_CARD.md  Intended use, data, architectures, every v4 number, failure modes, artifacts
│   ├── PLAN.md        Scope, data strategy, architecture, Open WebUI integration, phase roadmap
│   ├── PROCESS.md     How to run the pipeline on the DGX Spark, a Mac, or CPU
│   └── EVALUATION.md  Verification protocol: leakage audit, metrics, CIs, robustness, filter tests, human review
├── notebooks/
│   ├── datagen/       Phase 1: build fingerprinted splits from Hugging Face sources (v3 kept, v4 current)
│   ├── training/      Phase 2 baselines, Phase 3 neural model
│   └── eval/          Phase 4: evaluation against the LoRA's held-out sets
├── reports/           One folder per executed run: report.md and plots (tracked; data and models are not)
├── src/               Installable inference package (prompt_injection_nn)
├── filter/            Two Open WebUI filters (retrieved content: request and inlet hooks; chat: inlet hook), shadow mode by default
│   └── test_files/    Planted-injection test document for live runs
├── serve/             The guard service: mlx/ (MLX server for Apple Silicon), check_guard_server.py, README.md
├── tests/             Unit, filter and parity tests; .github/workflows/tests.yml runs them
├── scripts/           package_release.py (model bundle for Releases), publish_to_openwebui.py (deploy), benchmark_latency.py, export_html.sh
├── data/              (gitignored) raw pulls and clean splits
└── models/            (gitignored) versioned run artifacts
```

## Related

- [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) — the live filters, including the inlet prompt-injection filter (v2) that `filter/safety_filter_prompt_injection_chat.py` was ported from
- [safety](https://github.com/beaudamore/safety) — the prompt-injection and content-safety LoRAs; the dataset recipe and eval sets here are taken from its notebooks
- [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn) — the pipeline this repo mirrors
