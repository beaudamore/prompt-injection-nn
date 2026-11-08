# filter/: two Open WebUI filters, one guard service

Two Open WebUI functions (2026-11-05). Each is a single self-contained Python file. Both send text to the same prompt-injection guard service over `POST /classify` and make no LLM call.

| File | Title in Open WebUI | Hook | Scans | On a hit (enforce mode) | Version |
|---|---|---|---|---|---|
| `safety_filter_prompt_injection_retrieved_content.py` | Prompt Injection Protection (Retrieved Content) | `inlet` and `request` | `inlet`, at send: the whole stored text of every file attached to the chat. `request`, just before the model reads them: every retrieved `<source>` chunk and every native tool result | `inlet`: blocks the turn and names each flagged file; deleting the file is a valve, off by default. `request`: drops the chunk from the context and the citation list, or blanks the tool result, and the turn continues. | 0.5.0 |
| `safety_filter_prompt_injection_chat.py` | Prompt Injection Protection (Chat) | `inlet` | The prompt the user typed (the last message, when it is a user message) | Blocks the turn, logs the violation to a knowledge base, counts it against the user, locks the account at the limit. | 3.0.0 |

**Both ship in shadow mode** (`log_only = True`): they score and report what they would have done, and change nothing.

**Which filter catches what, and when.** Files attached to a chat are scanned when a message is sent (Retrieved Content filter, `inlet` hook): the whole stored text of each file, and a flagged file blocks the turn. The typed prompt is scanned at send too (Chat filter): a flagged prompt blocks the turn. Retrieved chunks and tool results are scanned just before the model reads them (Retrieved Content filter, `request` hook): only the flagged text is removed and the turn continues. Web search results, files of a chat folder and, in most cases, a model's knowledge files are not in the list the `inlet` hook reads (see "Which files" under its Behaviour section); they are covered by the `request` hook when they are retrieved. Open WebUI runs no filter at upload time; `inlet` (at send) is the earliest hook there is.

**Not done yet (2026-11-05):**

- The guard has **not been measured on typed (chat-shaped) prompts**. Its published numbers and this repo's v4 evaluation are on documents, emails, web pages, tool outputs and agent messages. That is why the chat filter ships in shadow mode.
- **The retrieved-content filter has run live in Open WebUI 0.13.0 in shadow mode only** (2026-11-05, on the tool-result path; see its section below). Enforce mode has not been run live. It was first verified live on 0.11.4 (2026-10-10).
- **The scan of attached files at send (the `inlet` hook, version 0.5.0, 2026-11-05) has not been run live.** It was checked by reading the Open WebUI `v0.13.0` source only. **The copy installed in Open WebUI is still 0.4.1** (the `request` hook only) until the owner has it updated.
- **The chat filter is not installed anywhere yet.** It was checked on 2026-11-05 by reading the Open WebUI `v0.13.0` source only: every `open_webui` import and call it makes exists there with the signature it uses.

## Install

1. **Guard service.** One of the two must be running. Both answer `POST /classify` in the same format, so either filter works against either one:
   - vLLM on the DGX Spark: Portainer stack `vllm-prompt-injection-guard`, port 8009 (the vLLM stack on the DGX Spark, `compose/live/vllm-prompt-injection-guard.yml`, not in this repo). Check: `curl http://localhost:8009/health` returns 200.
   - The MLX server on the Mac (Apple Silicon): `serve/mlx/server.py`, port 8194, `http://host.docker.internal:8194` as seen from the Open WebUI container. Install, run and check: `serve/README.md`.
2. **Add the filter(s).** Admin Panel, Functions, add the file, save. Do this for one file or both. Enable each globally or per model.
3. **Set `guard_url`** on each filter to the service from step 1. It is the only setting that differs between the two guard services: `http://host.docker.internal:8009` (the default) for vLLM, `http://host.docker.internal:8194` for the MLX server. Leave `guard_model` (`prompt-injection-guard`), `threshold` (0.5) and `log_only` (on) at their defaults.
4. **First turn.** With `log_only` on nothing is blocked or dropped. Retrieved Content: the log shows one `turn` line and one line per chunk with the guard's probability, and for a message with attached files one `upload` line and one `file` line per file; a flagged file shows the status `Would block: 1 attached file(s) flagged as prompt injection: ...`. Chat: a flagged prompt writes one `would block (shadow mode)` line to the log and shows a `would block (shadow mode)` status in the chat.
5. **Enforce.** Review the shadow-mode lines first (`docs/EVALUATION.md` sections 10.3 and 12), then set `log_only` off on that filter.

## What scores the text

One thing: the guard service. `Horizon-Labs/prompt-injection-guard-base` (a 308M-parameter ModernBERT
classifier, Apache-2.0, trained for documents, emails, web pages and tool outputs; PIArena F1 0.97,
LLMail recall 0.998, NotInject 94% accuracy). The filters post text to its `/classify` endpoint
(request `{"model", "input": [...]}`) and get back `data[i].probs = [P(SAFE), P(INJECTION)]` per input.
Text is first normalized with Unicode NFKC (the guard's tokenizer does the same); text longer than
2,000 characters is then sent as overlapping pieces (`PIECE_CHARS` 2,000, `PIECE_OVERLAP` 1,000, at most
`BATCH_SIZE` 64 pieces per guard call) and the highest piece score is kept. Why 2,000 (6,000 before
2026-11-05): see "Chunk size" below.
On the two false positives the in-house models produced in shadow mode on 2026-10-10 (a 404 page, an
API streaming log) the guard scores 0.007 and 0.05.

The code that talks to the guard (`GuardClient`, `Score`, `pieces_of`, `PIECE_CHARS`, `PIECE_OVERLAP`,
`BATCH_SIZE`) is the same in both files. An Open WebUI function is a single file, so the copy is
intentional; `tests/test_filter_chat.py` fails if the two copies differ.

The in-house models in this repo (`src/prompt_injection_nn`, `models/`) are for evaluation and
comparison only; neither filter loads them.

## Prompt Injection Protection (Retrieved Content)

`safety_filter_prompt_injection_retrieved_content.py`: one filter with two hooks. Until 2026-11-05 this file was `safety_filter_rag_injection_scan.py`, titled "RAG Injection Scan"; the rename changed the title, the logger name (`prompt_injection_retrieved_content`) and wording only. Version 0.5.0 (2026-11-05) added the `inlet` hook; the `request` hook's code is unchanged from 0.4.1. The file name, title and logger name are kept for now because the filter is installed in Open WebUI under that id.

| Hook | When | What it scans | On a hit (enforce mode) |
|---|---|---|---|
| `inlet` | When a message is sent, before anything else happens to the turn | The whole stored text of every file attached to the chat | Blocks the turn and tells the user which file and why. Optionally deletes the file. |
| `request` | Just before the model reads the context, and again after each tool call | Every retrieved `<source>` chunk and every native tool result | Removes only the flagged chunk or tool result. The turn continues. |

The `request` hook is described first, then the `inlet` hook.

On every turn it finds the `<source ...>...</source>` blocks that retrieval appended to the context (knowledge bases, uploaded files, web search, tool results all flow through the same list) and every native tool result, sends each chunk to the guard, and, in enforce mode, removes flagged chunks from the message text and from `metadata["sources"]` so they neither reach the model nor appear as citations. The turn continues; dropping the chunk is the action. Verified against the `<source>` rendering and request hook of Open WebUI 0.11.4 (`backend/open_webui/utils/middleware.py`, `get_source_context`).

**Live in Open WebUI 0.13.0 (2026-11-05, shadow mode).** Installed as function id `prompt_injection_protection_retrieved_content` (active, not global, attached to one test model), `guard_url` `http://host.docker.internal:8194` (the MLX server). Uploading `filter/test_files/planted_injection_vendor_guide.md` and asking the model to analyse it produced the status line "Would drop 1 retrieved chunk(s) / tool result(s) flagged as prompt injection: tool result..." and the log line `turn scanned=2 flagged=1 mode=shadow ... ms=138.2`.

- **The file arrived as a tool result, not as chunks.** On that model (built-in tools on, native tool calling) Open WebUI 0.13.0 sent the `<source>` block with an empty body (`chars=0`) and the model fetched the file with the built-in `view_file` tool. The whole file (4,982 characters) was scanned as one tool result and scored 1.000. In enforce mode that blanks the whole tool result, not only the planted section. Chunk-level dropping applies only when content arrives through retrieval as `<source>` chunks.
- **Silent when there is nothing to scan.** An earlier turn ("this is a test..") showed no status line: retrieval returned zero chunks (relevance threshold 0.2).
- The chat model (Gemma 4 12B) noticed and refused the planted instruction by itself in shadow mode. That is the model's behaviour, not the filter's.
- Enforce mode (`log_only` off) has not been run live.

### Valves (Retrieved Content)

| Valve | Default | Meaning |
|---|---|---|
| `log_only` | `True` | Shadow mode, for both hooks. `False` drops flagged chunks, blanks flagged tool results and blocks a turn that carries a flagged attached file. |
| `guard_url` | `http://host.docker.internal:8009` | The guard service: vLLM (port 8009) or the MLX server (`http://host.docker.internal:8194`). |
| `guard_model` | `prompt-injection-guard` | The served model name the service expects (`--served-model-name` in the stack). |
| `threshold` | `0.5` | Flag at or above this P(INJECTION). 0.5 is the guard's own decision point; raise it to drop fewer benign chunks. |
| `timeout_s` | `30` | Wait per batch of up to 64 pieces. |
| `scan_tool_results` | `True` | `request` hook: also scan native tool-call results. |
| `scan_uploaded_files` | `True` | `inlet` hook: scan the files attached to a message when it is sent. `False` turns the upload scan off; the `request` hook is not affected. |
| `block_mode` | `message` | `inlet` hook. `message`: show `block_message` as the reply and end the turn without calling the model. `error`: raise an error on the message. |
| `block_message` | `⛔ This message was blocked by a safety filter and was not sent to the model. You can continue the conversation.` | The reply shown for a blocked turn; the flagged files and what to do next are listed under it. Also the marker used to scrub blocked turns from later history. The same default as the Chat filter and the antivirus filter. |
| `delete_flagged_files` | `False` | `inlet` hook, enforce mode only: also delete each flagged file (its file record, then the stored file), only when the sending user is an admin, owns the file or has write access to it. Its knowledge-base links and vectors are not removed. Off by default: one flagged paragraph would delete the whole document, and the guard has false positives. |
| `show_status`, `debug`, `log_text_chars` | on, on, 160 | Status line in the chat; one log line per chunk and per attached file; preview length in that line. |
| `priority`, `enabled` | -50, on | Order among filters, for both hooks; master switch for both hooks. |

### Behaviour (Retrieved Content)

- Scans the `<source>` blocks in system and user messages, string or list content: knowledge bases, chat uploads (chunked or full context), web search, OCR'd and transcribed files, and non-native tool results.
- Scans native tool-call results too (`tool`-role messages, one chunk each). In enforce mode a flagged result is replaced by a notice written to the model: the content was withheld by this filter as a possible prompt injection (reason and score), do not call the tool again for it, tell the user it could not be used and that an administrator can review it. The message stays so the tool call is still answered. Since 0.4.1 (2026-11-05): with the earlier bare `[tool result removed ...]` line the chat model (Gemma 4 12B) returned an empty reply and the user saw only the grey status line. The new wording has not been re-run live yet.
- Does not see images sent to a vision model; they are not text.
- Fails open and says so: if the guard does not answer, the log gets a warning naming the URL and the error, the chat gets a status line with the number of chunks that passed through unscanned, and the turn continues.
- Emits one status event per turn naming dropped (or would-be-dropped) chunks with their probability.
- Logs, at INFO under the logger `prompt_injection_retrieved_content`, one line per turn (`scanned=`, `flagged=`, `mode=`, `guard=`, `model=`, `threshold=`, `ms=`) and one line per chunk (`debug` valve, on by default): kind (source or tool), id, name, size, pieces scored, probability, threshold, verdict, action taken, and a text preview. A would-drop line is enough to explain the verdict without re-scoring the chunk.
- Every chunk is scanned, every turn; there is no cap. The time taken is in the turn log line.
- Every flagged chunk carries the reason `Embedded Instruction` (the contract's indirect-injection reason); the guard gives no finer reason.
- Does not touch typed user text; that is the chat filter's job.
- Never blocks a turn. Only the `inlet` hook does.

### Behaviour (attached files at send, the `inlet` hook)

Added in 0.5.0 (2026-11-05). Not run live; every statement about Open WebUI below was read from the `v0.13.0` source.

- **What is scanned.** Every item of type `file` in `body["metadata"]["files"]`. The same file id listed twice is scanned once. For each one the filter loads the file record (`Files.get_file_by_id`) and scores `data["content"]`, the text Open WebUI stored when it processed the upload. The whole text is scored: over 2,000 characters (after NFKC) it goes to the guard in overlapping pieces and the highest piece score is the file's score. Each file is scored on its own, one guard call per group of up to 64 pieces.
- **Which files (current behaviour, an open decision for the owner).** Per a reading of the `v0.13.0` source (not run): the list always holds the files the user attached to the chat; a model's knowledge files are in it only when the model uses legacy function calling and the user attached at least one file; files of a chat folder are not in it. Whatever is not scanned here is still scanned by the `request` hook when it is retrieved.
- **Access.** A file is read and scored only when the sending user is an admin, owns it, or has read access by Open WebUI's own `has_access_to_file(id, 'read', user)`; the user record is loaded with `Users.get_user_by_id`. Any other file is skipped and logged as `no read access for this user`, and nothing about it is shown to the user (the status treats it the same as a file that does not exist). With no user on the request, a user that is not found, or the Open WebUI helpers missing, the scan is not active for that turn: it fails open with the warning and status below.
- **The name shown** for a file that was read, in the status, the log and the block message, is its stored filename, made one line, with backticks turned into apostrophes, and cut to 120 characters (a skipped file is logged under the name the request gave, made one line the same way). The block message shows it as a code span (in backticks), so markdown in a stored file name is not rendered. File ids are made one line wherever they are logged.
- **What is not scanned.** Knowledge collections, notes, chats, folders, pasted text and web search items are skipped; what they retrieve is covered by the `request` hook. A file with no stored text (an image, a file still being processed, an empty file, a file record that no longer exists) is skipped and logged with the reason.
- **Flagged** means the file's score is at or above `threshold`.
- **Shadow mode** (`log_only` on, the default): one INFO `upload` line per turn, one INFO `file` line per file (`debug` valve, on by default), the status `Would block: N attached file(s) flagged as prompt injection: <name> (p=0.99), ...`, and nothing else. No block, no delete.
- **Enforce mode** (`log_only` off), with at least one flagged file: the turn is blocked before the model is called. `block_mode` `message` replaces the reply with `block_message` followed by one line per flagged file and one line of next steps; `error` raises an error with the same text. For one flagged file the user sees:

  ```
  ⛔ This message was blocked by a safety filter and was not sent to the model. You can continue the conversation.

  - Attached file `guide.md` was flagged as a possible prompt injection (guard score 0.990, threshold 0.5). The file was not deleted and stays attached to this chat, so later messages in this chat are blocked until it is removed.
  - What to do next: remove the flagged file from this chat or start a new chat. An administrator can review it.
  ```

  With `delete_flagged_files` on, the second sentence of the file line reads `The file was deleted from Open WebUI.` (or the not-deleted sentence if the delete failed or the user has no write access to the file; either is logged as a warning).
- **With `delete_flagged_files` off (the default) the file stays attached to the chat, so every later turn in that chat is blocked again until the user removes the file or starts a new chat.** In Open WebUI 0.13.0 the file is removed in the Controls panel, under Files (`src/lib/components/chat/Controls/Controls.svelte`; read from the source, not seen live).
- **The files of earlier messages are sent again with every later message.** The 0.13.0 chat page sends, as `files`, every non-image attachment of the messages in the current branch plus the new message's own (`src/lib/components/chat/Chat.svelte`), and `main.py` copies that list into `metadata["files"]`. So a file is scored again on every turn of the chat, whole. There is no cache; the time taken is in the `upload` log line (`ms=`).
- **Deleting** (`delete_flagged_files` on) needs the sending user to be an admin, the file's owner, or to have `'write'` access (`has_access_to_file`); otherwise the file is not deleted and a warning is logged. The order is the one Open WebUI's own delete route uses: the file record first (`Files.delete_file_by_id`), then the stored file (`os.remove` on the record's `path`, best effort: a failure there is a warning and the file still counts as deleted). A record already gone counts as deleted. It does less than Open WebUI's own delete (`DELETE /api/v1/files/{id}`): knowledge-base links, knowledge-base chunks and the file's vector collection `file-<id>` are not removed (known limit), a stored object on storage that is not local disk stays behind, no file-deleted event is published (`EVENTS.FILE_DELETED`), and the chat's messages still list the file. After a delete the next turn is not blocked: the file record is gone, so the item is skipped as `file record not found`.
- **No violation count, no lockout, no webhook, no violation knowledge base.** The person who uploads a poisoned document is usually its victim.
- **If the guard does not answer: fails open and says so.** Each guard call stands alone: a failed call loses only its own pieces, a score already found still blocks, and a guard answer that scores fewer pieces than it was sent counts as a failed call. The seconds spent in failed guard calls are added up over the turn; once they reach `FAILED_CALL_TIMEOUTS` (2) x `timeout_s` (60 s at the default), the remaining pieces of that turn (the rest of the current file and every later file) are not sent and pass unscanned. A call that fails at once (guard not running, an HTTP error) uses up next to nothing, so the remaining calls are still tried. The first failed call of a file is a WARNING with the guard URL and the error (`guard call failed for file id=... name=... (N piece(s) UNSCANNED, guard=...): <error>`); when that file has more than one failed call or pieces that were not sent, one summary WARNING follows (`file id=... name=...: X of Y piece(s) UNSCANNED (F failed guard call(s); K piece(s) not sent, S s already spent in failed calls this turn)`). When no file got a score, the chat gets the status `Prompt injection scan of attached files is not active: <error>. N file(s) passed through unscanned.`; otherwise files not fully scored are added to the result status as `Not fully scanned: N file(s), because the guard did not answer (<error>).` The turn continues and nothing is deleted for an unscored file.
- **The hook's own errors never end a turn.** An exception in the scan code is logged, a closing status says the attached files passed through unscanned, and the turn continues. The two intended blocks (`block_mode` `message` and `error`) are not affected.
- **Log lines**, at INFO under the same logger: one per turn, `upload user= files= scanned= skipped= flagged= incomplete= mode= guard= model= threshold= ms=`, and one per file, `file id= name= chars= pieces= p= threshold= verdict= action= text=` (the preview is capped by `log_text_chars`; no other file text is logged). The verdict is `INJECTION`, `SAFE`, `PARTLY SCANNED`, `UNSCANNED` or `SKIPPED`.
- **Blocked turns are scrubbed from later history** on every request, in shadow mode too, as in the Chat filter: an assistant message that starts with `block_message`, and the user message before it.
- **Block-aware since 0.5.0** (`FILTER_BLOCK_AWARE = True`), the same coordination as the Chat filter and the antivirus filter: a block is recorded in `metadata["filter_block"]` and the last block-aware filter in the chain shows the message and ends the turn. The inlet order is by `priority`, lowest first: antivirus (-200), Chat (-100), this filter (-50), then the content-safety (-1) and policy (0) filters of `openwebui-safety-filters` when they are active on the model. When this filter is the last block-aware one it ends the turn for blocks the earlier filters recorded, with its own `block_message`, in shadow mode too, and also when `enabled` or `scan_uploaded_files` is off. Open WebUI itself does not read `FILTER_BLOCK_AWARE`; only these filters do, in their `inlet` hooks, so the `request` hook is not affected.
- **In legacy function-calling mode** Open WebUI appends the model's attached knowledge items to the same list when the user attached at least one file, so a model-knowledge item of type `file` is scanned too in that case (read from the `v0.13.0` source, not run; see "Which files" above).

### The two hooks on the same file

- **Enforce mode:** a flagged attached file blocks the turn at `inlet`, so the `request` hook never runs for that turn.
- **Shadow mode:** nothing is blocked, so both hooks report the same file: once as an attached file (`Would block ...`) and once more when its text reaches the model, as a retrieved chunk or as a tool result (`Would drop ...`). The double report is expected.
- A file whose whole text scores below `threshold` passes the `inlet` hook; its chunks are still scored one by one by the `request` hook.

## Prompt Injection Protection (Chat)

`safety_filter_prompt_injection_chat.py`, an `inlet`-hook filter. It is `safety_filter_prompt_injection_v2.py` 2.1.0 from `openwebui-safety-filters` with the detector swapped: the LLM call (`injection_detection_model_id`) is replaced by the guard. Everything else that filter did is kept.

### Valves (Chat)

| Valve | Default | Meaning |
|---|---|---|
| `log_only` | `True` | Shadow mode: score and log what would have been blocked; block nothing, count nothing, lock nobody out, send no webhook. `False` enforces. |
| `guard_url` | `http://host.docker.internal:8009` | The guard service: vLLM (port 8009) or the MLX server (`http://host.docker.internal:8194`). |
| `guard_model` | `prompt-injection-guard` | The served model name the service expects. |
| `threshold` | `0.5` | Flag at or above this P(INJECTION). Raise it to block fewer benign prompts. |
| `timeout_s` | `30` | Wait per batch of up to 64 pieces. |
| `block_on_unsafe` | `True` | From v2. `False` with `log_only` off: a flagged prompt is not blocked and the status line reads `Prompt injection check FAILED (not enforced): ...`. |
| `block_mode` | `message` | `message`: show `block_message` as the reply and end the turn without calling the model. `error`: raise an error on the message. |
| `block_message` | `⛔ This message was blocked by a safety filter ...` | The reply shown for a blocked prompt. Also the marker used to scrub blocked turns from later history. |
| `violation_kb` | `Prompt Injection Violations` | Knowledge base that receives one report file per violation; created if missing. `none` disables it. |
| `max_violations_count` | `3` | Violations before the account is locked (role set to `pending`). |
| `enable_webhook_notifications` | `False` | Send a webhook when a user is locked out. |
| `notification_webhook_url_env` | `PROMPT_INJECTION_WEBHOOK_URL` | Environment variable (in the Open WebUI container) holding the webhook URL. |
| `notification_webhook_subject`, `notification_webhook_timeout` | `Prompt injection lockout`, 10 | Subject in the payload; request timeout in seconds. |
| `enable_step_debug`, `enable_full_debug` | off, off | Step-by-step and full debug prints to the container log. |
| `priority`, `enabled` | -100, on | Order among inlet filters; master switch. |

Removed from v2: `injection_detection_model_id`. There is no `show_status`, `debug` or `log_text_chars` valve on this filter; the shadow-mode log preview is fixed at 160 characters.

### Behaviour (Chat)

- Scores the last message of the request when its role is `user`, the same selection as v2. For a multimodal message the text parts are joined and scored; images are not text and are not scored. A prompt longer than 2,000 characters (after NFKC) is scored in pieces and the highest score decides.
- The scored text is the message as Open WebUI hands it to inlet filters. In 0.13.0 `add_file_context` (`backend/open_webui/utils/middleware.py`) can prepend an `<attached_files>` block to a user message before inlet filters run; when it does, that block is scored, logged and reported with the prompt. Read from the source, not observed live.
- Flagged means P(INJECTION) at or above `threshold`. The guard gives no reason, so the reason text everywhere is `Prompt injection (guard score 0.987, threshold 0.5)`.
- **Shadow mode** (`log_only` on): writes one INFO line under the logger `prompt_injection_chat` (`would block (shadow mode)`, user id, size, pieces, score, threshold, a 160-character preview), shows the status `Prompt injection check complete: would block (shadow mode): ...`, and returns the request unchanged. No block, no violation report, no count, no lockout, no webhook.
- **Enforce mode** (`log_only` off), on a flagged prompt, in this order:
  1. Writes a violation report file and adds it to the `violation_kb` knowledge base.
  2. For a non-admin user: adds 1 to `violation_count` in the user's `info`; at `max_violations_count` sets the user's role to `pending` and, if enabled, posts the lockout webhook (JSON fields `event`, `subject`, `user_id`, `user_name`, `user_email`, `reason`, `timestamp`, `content_preview`). Admins are never counted or locked out; their prompt is still blocked.
  3. Shows the status `Content blocked - prompt injection detected: ...`.
  4. Blocks: `block_mode` `message` replaces the reply with `block_message` plus the reason and ends the turn without calling the model; `error` raises an error instead.
- Blocked turns are scrubbed from later history on every request, in both modes: an assistant message that starts with `block_message`, and the user message before it, are removed before the model sees the history.
- Coordination with other block-aware filters (`FILTER_BLOCK_AWARE = True`), unchanged from v2: a block is recorded in `metadata["filter_block"]`, and the last block-aware filter in the chain shows the message and ends the turn. This also applies in shadow mode to a block that an earlier filter recorded: if this filter is the last block-aware one, it ends the turn for that filter.
- **If the guard does not answer: fails open**, as v2 did when its detector call failed. The turn continues, nothing is counted, and the status line reads `Prompt injection check FAILED (not enforced): check failed: <error>`. The error is printed to the container log only when `enable_step_debug` or `enable_full_debug` is on.

## Chunk size

Open WebUI's Chunk Size (default 1,000; this instance 1,200 with 300 overlap) does not need to match the filters' piece size (`PIECE_CHARS` 2,000, `PIECE_OVERLAP` 1,000). The piece size is only a ceiling that keeps one text inside the model's 8,192-token limit; a 1,200-character chunk is one piece and gets one score. The ceiling matters for the attached-file scan, full-context uploads, long web pages and long tool results. With a 300 overlap, a planted instruction that falls in the overlap appears in two neighbouring chunks, so both are flagged. Chunk size does affect speed (tokens per chunk).

**Why 2,000 (changed from 6,000 on 2026-11-05).** One input over 8,192 tokens makes the guard refuse the whole request with HTTP 400, and the filters fail open, so text built to exceed the limit would push everything in that request through unscanned. The guard's tokenizer falls back to one token per UTF-8 byte (at most four per character) for characters it does not know, and it applies NFKC, which expands some characters into several. `pieces_of` therefore applies NFKC itself before splitting. Measured live against the MLX guard on 2026-11-05 (M1 Ultra, other services idle):

- 6,000 characters of U+10FFFD: 24,003 tokens, HTTP 400. 2,000 of them: 8,003 tokens, HTTP 200, about 0.81 to 0.86 s for that one piece.
- Sent raw, 2,000 x U+1D160 is 18,003 tokens and 2,000 x U+FDFA is 10,001 tokens, both HTTP 400. Through `pieces_of` (NFKC first) the same texts become pieces of at most 2,000 characters that the guard accepts (6,002 and 558 tokens for one piece), and an injection wrapped in 1,800 of those characters scored 0.9997 and 1.0000.
- Cost on ordinary text: 60,000 characters 691 ms at 6,000 against 803 ms at 2,000 (12 against 59 pieces); 300,000 characters 3,153 ms against 3,752 ms (60 against 299 pieces). The test file below (4,760 characters) is 4 pieces, highest score 1.0000.

Known limits of the piece size:

- The Open WebUI container's Python (3.12.16) has Unicode 15.0.0 tables. A character that the tokenizer expands but those tables do not know would not be expanded by the filter first. Not observed: five Unicode 16 characters with decompositions (U+16D68, U+16D69, U+16D6A, U+105C9, U+105E4), 2,000 of each through `pieces_of`, were accepted by the guard at 8,003 tokens each (measured 2026-11-05).
- The cost of NFKC: the scan's cost follows the normalized length, and NFKC can expand text up to 18 times (U+FDFA), so a file made of such characters costs up to 18 times its size in pieces. This is bounded only by Open WebUI's upload size limit. Not measured beyond the 2,000-character cases above.
- Not fixed, an open decision for the owner: one guard call carries up to 64 pieces, and 64 worst-case pieces (about 0.81 s each) need about 52 s against the 30 s `timeout_s`, so a file padded with such characters can still time out and pass unscanned. A measured alternative, not applied: 16 pieces per call costs about 2% more on ordinary text (300,000 characters: 3,775 ms at 64 per call, 3,844 ms at 16) and keeps the worst call near 13 s.
- Not fixed: the MLX server keeps working on a request after the client gave up (`serve/mlx/server.py` has no disconnect check).

## Test file

`filter/test_files/planted_injection_vendor_guide.md` (4,760 characters): a fictional vendor guide with one planted instruction in section 4. Scored through the guard on 2026-11-05:

- By section: section 4 1.0000, the title block 0.0518, every other section 0.0000.
- Cut the way Open WebUI would without the header splitter (1,200 characters, 300 overlap): the two windows containing the planted text 1.0000 and 0.9993, the other three 0.0055 or below.
- Whole file as one text: 0.9999.

To exercise the retrieval path, upload it and ask a question the document answers, for example "What notes does the guide have for automated readers, and what are the payment terms?". On a model with native tool calling the file may arrive as a tool result instead (see the live result above).

The EICAR antivirus test string (`openwebui-safety-filters/antivirus/test_files/eicar.txt`, 69 characters) scores 0.9992 INJECTION. It is not a prompt injection, so this is a false positive; once the retrieval filter enforces, uploading that file to test the antivirus filter will also get it dropped by this filter. Since 0.5.0 the `inlet` hook scores the same text at send, so in enforce mode it also blocks the turn for that file.

## Tests

`pytest tests/test_filter.py tests/test_filter_chat.py tests/test_filter_uploads.py`. All run against a stub scorer and against a stub `/classify` server on a local port; no Open WebUI, model or guard service is needed. 181 passed on 2026-11-05 (20 + 27 + 134).

- `tests/test_filter.py` (Retrieved Content, the `request` hook): flagged block removed, other blocks byte-identical, citation entry removed from both `body["metadata"]` and `__metadata__`, status event emitted, shadow mode untouched, list content, tool results, fail-open with the unscanned notice, no cap, piece splitting, threshold valve.
- `tests/test_filter_chat.py` (Chat), with stand-ins for the `open_webui` modules the filter imports: safe prompt untouched, empty prompt never sent to the guard, scoring off the event loop, shadow mode leaves the body untouched with no side effects, enforce mode blocks in both block modes, violation report written, count increments and lockout at the limit, admin exempt, webhook payload on lockout, history scrub, deferral to a later block-aware filter, threshold valve, long prompt split with the maximum deciding, fail-open when the guard is down or answers HTTP 500, and the guard code being identical in both filter files.
- `tests/test_filter_uploads.py` (Retrieved Content, the `inlet` hook), with stand-ins for the `open_webui` modules the hook uses: clean file untouched, non-file items and files with no stored text skipped, shadow mode changes and deletes nothing, enforce mode blocks in both block modes, the block message names the flagged file and its score and not the clean one, `delete_flagged_files` off makes no delete call and on deletes only the flagged file, a failed delete is reported as not deleted, the turn after a delete, long file split with the maximum deciding, threshold valve, fail-open with the warning and the unscanned status (scorer error, guard down, HTTP 500, outside Open WebUI), no user, no event emitter, history scrub, deferral to a later block-aware filter, an earlier filter's block enforced in every valve state, the valve list and defaults, the `request` hook unchanged by a recorded block, and the double report in shadow mode. Added on 2026-11-05 after the review: read access (another user's file skipped, read grant, admin, role taken from the user record, user missing or not found), a file id listed twice, delete order and write access, one guard call per file and per group of 64 pieces, a failed call losing only its own pieces, the limit on time spent in failed calls, partly scanned and unscanned files reported, a status emitter that raises, errors in the hook's own code, and a 2,000-character text of four-byte characters being one piece.

Still open: an enforce-mode run of the retrieval filter, any run of its `inlet` hook (attached files), and any run of the chat filter in a live Open WebUI (`docs/EVALUATION.md` section 9). The retrieval filter's shadow-mode live run is above.
