# filter/safety_filter_rag_injection_scan.py

An Open WebUI **request-hook filter**. On every turn it finds the `<source ...>...</source>` blocks that retrieval appended to the context (knowledge bases, uploaded files, web search, tool results all flow through the same list), classifies each chunk with `prompt_injection_nn`, and, in enforce mode, removes flagged chunks from the message text and from `metadata["sources"]` so they neither reach the model nor appear as citations. The turn continues; dropping the chunk is the action. No LLM call. Verified against the `<source>` rendering and request hook of Open WebUI 0.11.4 (`backend/open_webui/utils/middleware.py`, `get_source_context`).

**Shadow mode is the default** (`log_only = True`): the filter scores and reports what it would drop, and leaves the context untouched. Run a week of shadow mode and review the would-drop list before switching to enforce (`docs/EVALUATION.md` sections 10.3 and 12).

## Install

1. Make the package importable inside the Open WebUI container: `pip install /path/to/prompt-injection-nn` in the container image, or mount the repo and set the `package_path` valve to its `src/` folder.
2. Copy a `models/` folder with a `phase2-*/` and a `phase3-cnn-bigru-*/` run into the container (or mount it) and set the `models_dir` valve, or set `PI_NN_MODELS_DIR`.
3. In Open WebUI: Admin, Functions, add the file as a filter, enable it globally or per model.
4. Valves: `backend` (empty = the Phase 3 `gate.json` recommendation), `budget` (`fpr_0.01` default; choose from the operator's dropped-benign-chunk budget), `log_only`, `max_chunks`, `show_status`, `debug`.

## Behaviour

- Scans system and user messages, string or list content.
- Fails open: any exception in loading or scoring logs a traceback and passes the context through unchanged.
- Emits one status event per turn naming dropped (or would-be-dropped) chunks with their reason.
- Chunks beyond `max_chunks` pass through unscanned, and the status line says so.
- Does not touch typed user text; that is the inlet prompt-injection filter's job.

## Tests

`pytest tests/test_filter.py` runs the hook against a stub detector: flagged block removed, other blocks byte-identical, citation entry removed from both `body["metadata"]` and `__metadata__`, status event emitted, shadow mode untouched, list content, fail-open, scan limit. The integration test against a live Open WebUI with a planted document (EVALUATION.md section 9) is still open.
