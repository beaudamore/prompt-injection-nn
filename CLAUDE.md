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

- Two data builds, v1 (`febe6837be209929`) and v2 (`d3d3fa8d932f3280`); Phases 2-4 run on each; artifacts under `models/`, per-run reports under `reports/`. Numbers and what they mean: README "Progress and findings".
- v1 and v2 are both superseded. v2 fixed the chat-shaped grouping partly (twin rate 43% -> 39%) and added legitimate emails (FPR 0%), but its BIPIA document key was not whitespace-normalized: BIPIA has only 182 base documents, 18 still straddle train and test, and injected rows outnumber clean 40:1 in train. Never promote a v1 or v2 model.
- Phase 3 has **not cleared the promotion gate** (`docs/PLAN.md` §4, recall-at-FPR plus document-shaped FPR): on v2 the TF-IDF baseline ties it at the 1% FPR budget.
- Phase 4 on v2: holdout accuracy 0.901, FP 9.4%, FN 10.3%; tracks the test split. No LoRA eval exists yet, so no head-to-head.
- **Next action:** v3 data build (`docs/PROCESS.md` §2), rerun Phases 2-4, compare the report folders.
- Every notebook's last cell writes a per-run report folder under `reports/`; `scripts/export_html.sh` exports executed notebooks to HTML.
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
