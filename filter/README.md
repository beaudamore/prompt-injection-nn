# filter/safety_filter_rag_injection_scan.py

An Open WebUI **request-hook filter**. On every turn it finds the `<source ...>...</source>` blocks that retrieval appended to the context (knowledge bases, uploaded files, web search, tool results all flow through the same list) and every native tool result, sends each chunk to the prompt-injection guard service, and, in enforce mode, removes flagged chunks from the message text and from `metadata["sources"]` so they neither reach the model nor appear as citations. The turn continues; dropping the chunk is the action. No LLM call. Verified against the `<source>` rendering and request hook of Open WebUI 0.11.4 (`backend/open_webui/utils/middleware.py`, `get_source_context`).

**Shadow mode is the default** (`log_only = True`): the filter scores and reports what it would drop, and leaves the context untouched. Run a week of shadow mode and review the would-drop list before switching to enforce (`docs/EVALUATION.md` sections 10.3 and 12).

## What scores the chunks

One thing: the guard service. `Horizon-Labs/prompt-injection-guard-base` (a 308M-parameter ModernBERT
classifier, Apache-2.0, trained for documents, emails, web pages and tool outputs; PIArena F1 0.97,
LLMail recall 0.998, NotInject 94% accuracy) served by vLLM with `--runner pooling` as the Portainer
stack `vllm-prompt-injection-guard` (`compose/live/vllm-prompt-injection-guard.yml`, host port 8009).
The filter posts chunk text to its `/classify` endpoint and gets back P(INJECTION) per input. A chunk
longer than 6,000 characters is sent as overlapping pieces and the highest piece score is kept. On the
two false positives the in-house models produced in shadow mode on 2026-10-10 (a 404 page, an API
streaming log) the guard scores 0.007 and 0.05.

The filter is self-contained: one Python file, no package, no model files in the container, nothing
to copy anywhere. The in-house models in this repo (`src/prompt_injection_nn`, `models/`) are for
evaluation and comparison only; the filter does not load them.

## Install

1. **Guard service.** Deploy `compose/live/vllm-prompt-injection-guard.yml` as a Portainer stack. It
   serves the guard from the shared HF cache on host port 8009. Memory: about 3 to 4 GB of the unified
   pool. Check: `curl http://localhost:8009/health` returns 200.
2. **Add the filter**: Admin Panel, Functions, add this file, save, enable it globally or per model.
   The default valves are right for this host: `guard_url` `http://host.docker.internal:8009`,
   `guard_model` `prompt-injection-guard`, `threshold` 0.5, `log_only` on.
3. **First turn with retrieved chunks**: the log shows one `turn` line and one line per chunk with the
   guard's probability. If the service is down, the filter logs a warning, shows a status line saying
   how many chunks passed through unscanned, and lets the turn continue (fail open).

Other deployments need the same two things: the guard served by vLLM (or any server answering
`POST /classify` in vLLM's format) and this file.

## Valves

| Valve | Default | Meaning |
|---|---|---|
| `log_only` | `True` | Shadow mode. `False` drops flagged chunks and blanks flagged tool results. |
| `guard_url` | `http://host.docker.internal:8009` | The guard service. |
| `guard_model` | `prompt-injection-guard` | The served model name the service expects (`--served-model-name` in the stack). |
| `threshold` | `0.5` | Flag at or above this P(INJECTION). 0.5 is the guard's own decision point; raise it to drop fewer benign chunks. |
| `timeout_s` | `30` | Wait per batch of up to 64 pieces. |
| `scan_tool_results` | `True` | Also scan native tool-call results. |
| `show_status`, `debug`, `log_text_chars` | on, on, 160 | Status line in the chat; one log line per chunk; preview length in that line. |

## Behaviour

- Scans the `<source>` blocks in system and user messages, string or list content: knowledge bases, chat uploads (chunked or full context), web search, OCR'd and transcribed files, and non-native tool results.
- Scans native tool-call results too (`tool`-role messages, one chunk each). In enforce mode a flagged result is replaced by `[tool result removed by RAG Injection Scan: ...]`; the message stays so the tool call is still answered.
- Does not see images sent to a vision model; they are not text.
- Fails open and says so: if the guard does not answer, the log gets a warning naming the URL and the error, the chat gets a status line with the number of chunks that passed through unscanned, and the turn continues.
- Emits one status event per turn naming dropped (or would-be-dropped) chunks with their probability.
- Logs, at INFO, one line per turn (`scanned=`, `flagged=`, `mode=`, `guard=`, `model=`, `threshold=`, `ms=`) and one line per chunk (`debug` valve, on by default): kind (source or tool), id, name, size, pieces scored, probability, threshold, verdict, action taken, and a text preview. A would-drop line is enough to explain the verdict without re-scoring the chunk.
- Every chunk is scanned, every turn; there is no cap. The time taken is in the turn log line.
- Every flagged chunk carries the reason `Embedded Instruction` (the contract's indirect-injection reason); the guard gives no finer reason.
- Does not touch typed user text; that is the inlet prompt-injection filter's job.

## Tests

`pytest tests/test_filter.py` runs the hook against a stub scorer and against a stub `/classify` server: flagged block removed, other blocks byte-identical, citation entry removed from both `body["metadata"]` and `__metadata__`, status event emitted, shadow mode untouched, list content, tool results, fail-open with the unscanned notice, no cap, piece splitting, threshold valve. The integration test against a live Open WebUI with a planted document (EVALUATION.md section 9) is still open.
