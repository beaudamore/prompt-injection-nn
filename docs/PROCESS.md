# Prompt Injection Classifier — End-to-End Process

**Last updated:** 2026-10-09
**Audience:** anyone running or extending this repo, on the DGX Spark or on their own machine.
**Companion docs:** `LEARNING.md` (every concept from zero), `PLAN.md` (why and what), `../README.md` (status and results).

This repo follows the `email-classifier-nn` process. Phases 1 to 4 have notebooks; section 7 (the filter) is pending Phase 5.

---

## 1. The pipeline in one picture

```
Hugging Face sources (+ gated generated rows, Phase 4)
      │
      ▼
[Phase 1] notebooks/datagen/phase_1_prompt_injection_datagen_v4.ipynb
      │   normalize → quality gates → exact + MinHash dedupe → grouped 80/10/10 split
      │   → engineered features → adversarial variants (train only) → manifest
      ▼
data/source-clean-v4/{train,val,test,adversarial_test,lora_holdout}.parquet + manifest.json
      │
      ├──▶ [Phase 2] notebooks/training/phase_2_baselines.ipynb           
      │         TF-IDF logistic regression + XGBoost → models/phase2-v4-<fp>/
      │
      └──▶ [Phase 3] notebooks/training/phase_3_text_cnn_bigru.ipynb      
                CNN + BiGRU, binary + reason heads → models/phase3-cnn-bigru-v4-<fp>/
                      │
                      ▼
              src/ package + filter/ request-hook filter                 (Phase 5)
```

`<fp>` is the first 16 characters of the Phase 1 config fingerprint.

---

## 2. Data versions

| Version | Directory | Sources | Status |
|---|---|---|---|
| v1 | `data/source-clean-v1/` | eight sources per `PLAN.md` §2.1-2.2; fingerprint `febe6837be209929` | Built 2026-10-09; Phases 2-3 trained on it. **Superseded pending v2**: BIPIA base documents straddle splits (61% FPR on clean BIPIA documents) and chat-shaped templates repeat across splits (42% six-word twin rate). Frozen for the record. |
| v2 | `data/source-clean-v2/` | v1 sources plus legitimate emails; BIPIA partitioned jointly by document and attack string; template key 8 words, MinHash 0.80; fingerprint `d3d3fa8d932f3280` | Built 2026-10-09; Phases 2-4 run on it (reports in `reports/`). **Superseded pending v3**: the BIPIA document key was not whitespace-normalized, so 18 base documents are still attacked in both train and test, and injected BIPIA rows outnumber clean ones 40:1 in train. Frozen for the record. |
| v3 | `data/source-clean-v3/` | v2 with BIPIA keyed by normalized base text and capped at three attacks per base document; fingerprint `a4c8dd4786013cfd` | Built 2026-10-09; Phases 2-4 run on it. **Superseded**: Phase 3 lost the gate to TF-IDF at every FPR budget, and the test set was 82% chat-shaped with 53 document-shaped attacks from one source. |
| v4 | `data/source-clean-v4/` | v3 plus eleven document, agentic and hard-negative sources (LLMail-Inject, PIArena, Nemotron agentic IPI, Kubernetes manifests, three 3nesdeniz sets, in-the-wild prompts, ogma, S-Labs benign, NotInject); third shape `agentic`; all nine reasons populated | Built 2026-10-09 (see the v4 report under `reports/`). **Current.** Phases 2-4 rerun 2026-10-10 with the padding-invariant model and the explicit gate: cleared on document + agentic rows by the CNN and by the TF-IDF+CNN ensemble, which is the shipped backend. Reports under `reports/`, numbers in `MODEL_CARD.md`. |

When a version is retired, record why here. The `email-classifier-nn` v1 contamination note is the model.

---

## 3. Where it runs

All notebooks carry the same resolver as `email-classifier-nn`:

**Project root**, in order: `PI_NN_DIR` environment variable; `/workspace/training/prompt-injection-nn` (Spark container); `/home/spark/projects/training/prompt-injection-nn` (Spark host); otherwise walk up from the working directory until a folder containing `docs/PLAN.md` and `notebooks/` is found.

**Compute device**: CUDA, then Apple MPS, then CPU. BF16 autocast on CUDA only. XGBoost uses CUDA when present, else CPU.

| Machine | Role |
|---|---|
| DGX Spark | Primary. JupyterLab in the `unsloth-notebook` container, host port 8889. Data and models live here. CUDA capped at 55% so vLLM keeps running. |
| Apple Silicon Mac | Optional fallback for public users. MPS is not bit-for-bit deterministic. |
| CPU | Works. The training set is roughly a tenth the size of the email corpus, so Phase 3 on CPU is minutes per epoch, not tens of minutes. |

---

## 4. Running a full cycle

Run the notebooks in this order; each asserts its inputs before doing any work.

1. **Phase 1**: run the datagen notebook top to bottom; confirm the manifest counts and per-split reason distribution; confirm no cross-split leakage.
2. **Phase 2**: run the baselines notebook; the selected winner is the bar for Phase 3.
3. **Phase 3**: run the neural notebook; compare against Phase 2 on the gates in `PLAN.md` §4.
4. **Eval**: run the eval notebook against the LoRA's held-out sets; it writes `eval_*.json` in the same format the `safety` repo's eval notebooks use.
5. **Reports**: the last cell of every notebook writes `report.md` plus its plots into a folder per run, `reports/<UTC stamp>_<phase>_<dataset version>_<fingerprint>/`. `reports/` is tracked; `models/` and `data/` are not, so the reports are what travels. After Jupyter has saved the executed notebooks, run `scripts/export_html.sh` to export them as self-contained HTML into `reports/html/<UTC stamp>/` as the full audit trail.
6. **After a run**: update the README results table with numbers, device, and date; commit the executed notebooks and the reports from the machine that ran them.

Never execute or paste attack strings from the data into a real model session outside the notebooks.

---

## 5. What is in git and what is not

Tracked: notebooks, docs, README, `src/`, `filter/`, `.gitignore`.

Not tracked: `data/`, `models/`, checkpoints, logs, environments. Rebuild `data/` by running Phase 1.

Executed notebooks are committed with outputs. They are the record of a run that travels with the repo.

---

## 6. Working across machines

GitHub is the hub; each machine is a disposable checkout.

- Pull before editing. Push when done.
- Never hold unpushed edits on two machines at once.
- Edit anywhere; execute on the Spark or Mac; commit executed notebooks from the machine that ran them.
- A stale checkout with nothing worth keeping gets reset to GitHub, after a `git status` check.

---

## 7. Deploying the filter

The filter is `filter/safety_filter_rag_injection_scan.py`, an Open WebUI `request`-hook filter (see `PLAN.md` §5 and `filter/README.md`). Steps:

1. Install the package where Open WebUI runs: `pip install /path/to/prompt-injection-nn` in the image, or mount the repo and set the filter's `package_path` valve to its `src/` folder.
2. Provide a `models/` folder holding a `phase2-*/` and a `phase3-cnn-bigru-*/` run (not in git; copy from the Spark or rerun the notebooks) and point the `models_dir` valve or `PI_NN_MODELS_DIR` at it.
3. Add the file under Admin, Functions as a filter; enable it globally. Leave `log_only` on. Review the would-drop status lines for a week; set the `budget` valve from the operator's dropped-benign-chunk budget (default `retrieval_fpr_0.01`); then set `log_only` off.
4. After every Open WebUI upgrade, rerun `pytest tests/test_filter.py` and re-read `get_source_context` in `backend/open_webui/utils/middleware.py`; the header line in the filter names the version it was verified against.

Locally: `pip install -e .[test]` then `pytest` (unit and filter tests run anywhere; `tests/test_parity.py` needs the data and model folders and runs on the Spark).
