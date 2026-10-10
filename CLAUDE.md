# CLAUDE.md — prompt-injection-nn

Project context for Claude Code. Read this first on any machine. The user's global rules
(`~/.claude/CLAUDE.md`) take precedence over anything here.

## What this repo is

A compact prompt-injection text classifier with no LLM, built to scan retrieved RAG chunks,
web-search results, and uploaded files inside Open WebUI before they reach the model, and to be
importable from any Python app. Repo: https://github.com/beaudamore/prompt-injection-nn

It mirrors https://github.com/beaudamore/email-classifier-nn phase for phase. When in doubt about
pipeline structure, notebook conventions, or evaluation, look at how the email repo does it.

It complements, not replaces, the prompt-injection LoRA in https://github.com/beaudamore/safety.
The LoRA (Qwen3.8-27B, served from the Spark's vLLM stack) handles typed chat prompts through
the existing inlet filter. This model handles the retrieval path, where one turn can carry ten or
more chunks and an LLM call per chunk is too slow.

## Where things stand (2026-10-09)

- Four data builds: v1 (`febe6837…`), v2 (`d3d3fa8d…`), v3 (`a4c8dd47…`), v4 (see README). Phases 2-4 run on v1 to v3; per-run reports under `reports/`. Numbers and what they mean: README "Progress and findings". New reader: `docs/LEARNING.md`.
- v1 to v3 are superseded; never promote a model trained on them. v3 fixed BIPIA (normalized document key, three attacks per document) and is the honest baseline for comparison.
- **v3 verdict: Phase 3 lost the promotion gate** (`docs/PLAN.md` §4). TF-IDF logistic regression beats the CNN+BiGRU at every FPR budget on the locked test (recall at FPR 1%: 0.641 vs 0.577). Both are far below a usable number. The v3 test set was 82% chat-shaped with 53 document-shaped attacks from one source, so it barely measured the product's input.
- v4 (`7f42d9d8…`) rebuilds the data around documents, emails, tool results and agent messages from eleven newly verified Hugging Face sources (licences, schemas and rejections recorded in the v4 datagen notebook and manifest); all nine reasons have training rows; a third shape `agentic` exists. Phase 2/3/4 notebooks point at v4.
- **v4 verdict (Phases 2-4 rerun 2026-10-10 with the padding-invariant model):** the gate is cleared on the product's rows. On test document + agentic rows, recall at FPR 1% / 0.5% / 0.1%: TF-IDF 0.908 / 0.867 / 0.843, CNN 0.954 / 0.930 / 0.890, **ensemble (mean) 0.961 / 0.943 / 0.909**. `gate.json` recommends `ensemble_mean`; the package loads it by default. Chat-shaped recall at 1% FPR is 0.64 and out of scope. Phase 4 (chat-shaped holdouts): accuracy 0.889. All numbers: `docs/MODEL_CARD.md`.
- **Phase 5 built 2026-10-10:** `src/prompt_injection_nn` (tokenizer, model, three backends, detector with budget thresholds, CLI), `filter/safety_filter_rag_injection_scan.py` (request hook, shadow mode default, fails open), `tests/` (21 unit and filter tests pass in the container; parity test passes against the v4 artifacts), `.github/workflows/tests.yml`, `scripts/benchmark_latency.py`.
- **Retrieval operating points:** thresholds chosen on validation document + agentic rows are saved as `fpr_budget_thresholds_val_retrieval` in every `thresholds.json`; the filter ships with `retrieval_fpr_0.01`.
- **Model is padding-invariant** (packed BiGRU). The package's batch test caught the leak; never reintroduce the unpacked forward.
- Phase 4 (LoRA held-out sets) on v3: accuracy 0.905, FP 12.4%, FN 7.1%. No LoRA eval exists yet, so no head-to-head.
- **Next action:** install the package and the filter into the Open WebUI container, run shadow mode for a week, set the budget valve from the operator's dropped-benign-chunk budget, then enforce (`docs/PROCESS.md` §7). In parallel: run the LoRA eval in `safety` for the head-to-head; a red-team set of hand-written planted documents; the human label review (`docs/EVALUATION.md` §10).
- Every notebook's last cell writes a per-run report folder under `reports/`; `scripts/export_html.sh` exports executed notebooks to HTML. Superseded reruns of the same version are deleted so there is one report per phase per version.
- Notebooks are executed headlessly with `jupyter nbconvert --to notebook --execute --inplace` inside `unsloth-notebook`; `models/` output folders are root-owned and are removed with `docker exec ... rm -rf` before a rerun of the same fingerprint (Phase 3 refuses to overwrite).
- The dataset recipe to start from is frozen in the LoRA notebook `safety/notebooks/prompt_injection_lora_qwen38_27b.ipynb`: Nemotron Safety Guard v3 `jailbreaking` rows and WildGuardMix adversarial as INJECTION, WildGuardMix benign as SAFE, with the subcategory-to-reason map there. Six of the nine reasons have zero training rows in that recipe; closing that gap is a Phase 4 job.

Full detail: `docs/PLAN.md` (design and Open WebUI integration), `docs/PROCESS.md` (how to run), `docs/EVALUATION.md` (verification criteria).

## Machines

| Machine | Role | Path |
|---|---|---|
| DGX Spark | Primary. Runs training. Holds `data/` and `models/`. | `/home/spark/projects/training/prompt-injection-nn` on host, `/workspace/training/prompt-injection-nn` in the `unsloth-notebook` container (JupyterLab, host port 8889) |
| Mac (Apple Silicon) | Optional fallback. MPS. | any clone |
| Windows laptop | Docs and notebook editing only. CPU-only torch. | `C:\Source\prompt-injection-nn` |

Notebooks will resolve the project root themselves (env var `PI_NN_DIR`, then the Spark paths, then a walk-up search for `docs/PLAN.md`). Copy the `resolve_project_dir()`, `resolve_device()`, `amp_context()`, and `empty_device_cache()` helpers from the email repo's v2 notebooks verbatim, changing only the env var name and the Spark paths.

GitHub is the hub. Pull before editing, push when done, never hold unpushed edits on two machines at once. Commit executed notebooks from the machine that ran them.

## Conventions that matter

- **Output contract is fixed:** `SAFE` or `INJECTION: <reason>` with one of the nine reasons in `docs/PLAN.md` §1. The filter and the LoRA depend on it.
- **Open WebUI integration is a `request`-hook filter, no patch.** The reasoning and the verified pipeline order are in `docs/PLAN.md` §5. Do not propose patching Open WebUI unless the user asks.
- **Notebooks are committed with outputs. LF line endings, `indent=1`.** Use the `notebook-editing` skill.
- **Holdout discipline.** `train` fits; `val` selects; `test` and `adversarial_test` are opened once. The LoRA's held-out sets (WildGuardMix `wildguardtest`, Nemotron held-out `jailbreaking`) are the comparison set and are never trained on.
- **Every run records provenance** in a manifest tied to the dataset fingerprint.
- **The data contains live attack strings.** Never paste them into a real model session outside the notebooks.
- **Do not commit or push unless the user asks.** State the exact change before making it.

## Skills in this repo

- `.claude/skills/pipeline-status/` — snapshot of git state, data versions, model runs, notebook execution state. Run first when resuming.
- `.claude/skills/notebook-editing/` — rules and helper for safe `.ipynb` edits.

## Related repos

- `email-classifier-nn` — the template for everything here.
- `safety` — the LoRAs; source of the dataset recipe, reason taxonomy, system prompt, and eval format.
- `openwebui-safety-filters` — the live filters; the new request-hook filter follows their block-aware pattern and header conventions.
- `open-webui` (local checkout at `C:\Source\open-webui`, 0.11.4) — the pipeline order in `docs/PLAN.md` §5 was read from `backend/open_webui/utils/middleware.py` there.
