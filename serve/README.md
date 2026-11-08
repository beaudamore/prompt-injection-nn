# serve/: the prompt-injection guard service

The two Open WebUI filters in `filter/` send text to one guard service over `POST /classify` and need nothing else.
This folder holds the MLX server that runs that service on a Mac (`serve/mlx/server.py`, added 2026-11-05) and a
checker that tests any `/classify` server (`serve/check_guard_server.py`). The model is
`Horizon-Labs/prompt-injection-guard-base` (ModernBERT, 307,531,778 parameters, labels `SAFE` / `INJECTION`). The other
server is vLLM on the DGX Spark (see "The vLLM side" below). Both answer in the same format under the same served name
(`prompt-injection-guard`), so the filters' `guard_model` valve stays as it is and only `guard_url` differs.

## Install and run on a Mac (Apple Silicon)

1. **Python 3.11 venv** with `pip install -r serve/mlx/requirements.txt` (`mlx-embeddings` 0.1.0, `mlx` 0.32.4,
   `fastapi`, `uvicorn`; verified with Python 3.11.14). No torch.
2. **Fetch the model into a Hugging Face cache.** The server never downloads; it reads the cache offline. With the
   venv active: `HF_HUB_CACHE=<cache dir> python -c 'from mlx_embeddings.utils import get_model_path; print(get_model_path("Horizon-Labs/prompt-injection-guard-base"))'`
   (the same call `guardctl fetch-model` makes).
3. **Environment variables** (defaults are the ones in `server.py`):

   | Variable | Default | Meaning |
   |---|---|---|
   | `GUARD_MODEL` | `Horizon-Labs/prompt-injection-guard-base` | Repo id as it sits in the cache |
   | `GUARD_CACHE` | unset (Hugging Face default cache) | Cache folder; sets `HF_HUB_CACHE` |
   | `GUARD_SERVED_NAME` | `prompt-injection-guard` | Name requests must carry in `"model"` |
   | `GUARD_HOST` | `127.0.0.1` | Bind address; `0.0.0.0` so Docker containers can reach it |
   | `PORT_GUARD` | `8194` | Port |
   | `GUARD_MAX_TOKENS` | `8192` | Longest input, special tokens included; longer is HTTP 400 |
   | `GUARD_BATCH_BUDGET` | `4194304` (16 x 512^2) | Sequences x longest length^2 per forward pass |
   | `GUARD_CACHE_LIMIT_MB` | `1024` | `mx.set_cache_limit` at startup |

4. **Run** `python serve/mlx/server.py` in a normal logged-in user session. MLX cannot see the GPU in a sandboxed or
   headless session ("No Metal device available"). The startup log shows `exact_gelu=23`.
5. **Check** `python serve/check_guard_server.py --url http://127.0.0.1:8194`. Add
   `--torch-model <local model folder>` for parity against Hugging Face transformers (needs torch and transformers in
   the interpreter running the checker, not in the server's venv), and `--chunk-chars` set to your Open WebUI Chunk
   Size. Exit status 0 means every check passed.
6. **Point the filters at it:** `guard_url` = `http://host.docker.internal:8194` from the Open WebUI container.

On steppenwolf (the owner's Mac) the server is run by a launcher in another repo,
`~/Source/local-llm-config/prompt-injection-guard-mlx/` (`guardctl`, `guard.env`, its own venv):

- `llmctl start guard` (an opt-in `llmctl` service, not a login service); ready about 7 s later.
- `guardctl status | test | logs | fetch-model`.
- Port 8194; model cache `/Volumes/evotwin/model_cache/hf-cache`.

## The /classify contract

Request (`"model"` may be omitted; any other name is a 404; other fields are ignored, nothing is truncated):

```json
{"model": "prompt-injection-guard", "input": ["The quarterly report is attached.", "second text"]}
```

Response (`probs` is the softmax over the model's labels in `id2label` order, `[P(SAFE), P(INJECTION)]`;
`label` is the larger one):

```json
{"id": "classify-...", "object": "list", "created": <unix time>, "model": "prompt-injection-guard",
 "data": [{"index": 0, "label": "SAFE", "probs": [<P(SAFE)>, <P(INJECTION)>], "num_classes": 2},
          {"index": 1, "label": "...", "probs": [<P(SAFE)>, <P(INJECTION)>], "num_classes": 2}],
 "usage": {"prompt_tokens": <tokens>, "total_tokens": <tokens>}}
```

Also `GET /health` (model, served name, labels, limits, versions) and `GET /v1/models`. Input text is never logged;
each request writes one `req N` line with counts, time and the highest P(INJECTION).

## Three things a plain mlx-embeddings load gets wrong

1. **Rotary base.** The model's `config.json` is in transformers 5 layout:
   `rope_parameters {full_attention: {rope_theta: 160000}, sliding_attention: {rope_theta: 160000}}`.
   `mlx-embeddings` 0.1.0 reads only `global_rope_theta` / `local_rope_theta` and would default the local one to
   10000, so every sliding-attention layer would be wrong and the scores would be wrong with no error.
   `rope_overrides()` maps the values.
2. **GELU.** `mlx-embeddings` 0.1.0 builds every GELU as the tanh approximation; Hugging Face uses exact GELU. The
   server switches all 23 GELU modules (22 MLPs and the head) to exact after loading (log line `exact_gelu=23`).
3. **Truncation.** `tokenizer.json` carries a built-in truncation at 1,024 tokens. The server tokenizes with
   `truncation=False` and clears that setting once at startup, because changing it at request time is not
   thread-safe.

## Batching and concurrency

One forward pass at a time. Inside one request, texts are sorted by length and run in sub-batches sized so that
sequences x longest length^2 stays under `GUARD_BATCH_BUDGET`; ten 300-token chunks go through in one pass, a long
piece may run alone. The lock is taken per pass, so separate requests take turns pass by pass instead of one
request waiting for all of another.

## Measured

Mac Studio M1 Ultra (48 GPU cores, 128 GB), 2026-11-05, chat, task, embedding, reranker, TTS, STT and PII services
loaded but idle; `serve/check_guard_server.py --torch-model <snapshot> --chunk-chars 1200`. All checks passed
(contract, 15 verdicts at 0.5, batch versus single, parity, token count).

| Measure | Result |
|---|---|
| Parity with transformers 5.20.2 / torch 2.14.1, float32, largest difference in P(INJECTION) over 15 texts | 1.71e-07 on CPU, 2.33e-07 on MPS |
| Token count | Identical: 6,476 for the 15 texts; longest 1,545 (past the 1,024 truncation) |
| Same text in a batch versus alone, largest difference | 3.54e-08 |
| Ten chunks of 1,200 characters, one request, 30 runs: MLX server over HTTP | p50 74.3 ms, p95 75.0 ms (second run: p50 76.7, p95 85.3) |
| Same, torch on MPS in process | p50 83.6 ms, p95 85.6 ms |
| Same, torch on CPU in process | p50 336.2 ms, p95 367.2 ms |
| One short input | p50 10.8 ms to 13.1 ms |
| Eight simultaneous requests of ten chunks | per request p50 370 ms, p95 980 ms; all eight answered after 584 ms (median of 5 rounds) |
| Eight pieces of 6,000 characters, one request (8,992 tokens, 3 passes; the filters' piece size until 2026-11-05) | p50 379 ms |
| Memory and start | RSS 1.72 to 1.77 GB after load; MLX active about 1.23 GB; load 3.0 s |

Sample scores (P(INJECTION)): benign business paragraph 0.0000; benign text about prompt injection 0.0040; a short
Python snippet 0.0213; plain override instruction 1.0000 in English and in German; an instruction hidden in a JSON
tool result 1.0000; an attack after token 1,024 of a 1,544-token text 1.0000.

Not measured: the guard on typed (chat-shaped) prompts; speed on any other Mac; this server against the Spark's vLLM
directly.

## Known limits

- **Inputs over 8,192 tokens** are an HTTP 400 for the whole request (vLLM does the same). The tokenizer has
  `byte_fallback` (one token per UTF-8 byte for characters outside its vocabulary) and applies NFKC, which expands some
  characters into several. Since 2026-11-05 the filters apply NFKC themselves and split text at 2,000 characters
  (6,000 before): measured on this server that day, 2,000 x U+10FFFD is 8,003 tokens (HTTP 200) where 6,000 of them
  were 24,003 (HTTP 400). Measurements and the remaining limits: `filter/README.md`, "Chunk size". The filters fail
  open on any guard error.
- **No disconnect check.** The server keeps working on a request after the client gave up (a filter's `timeout_s`).
  Not fixed.
- **Trained at 1,024 tokens.** The model card's training code truncated at 1,024 tokens (`code/train/train.py`,
  `training/train_log.json` in the model snapshot). A 2,000-character piece can still be well over 1,024 tokens (8,003
  measured for characters the tokenizer splits into bytes) and is scored in one pass. The check texts with an attack after token 1,024 scored 1.0000; nothing broader was measured.
- **EICAR false positive.** The EICAR antivirus test string
  (`openwebui-safety-filters/antivirus/test_files/eicar.txt`, 69 characters) scores 0.9992 INJECTION. It is not a
  prompt injection. Once the retrieval filter enforces, uploading that file to test the antivirus filter will also
  get it dropped by this filter.
- **GPU access.** MLX needs a normal user session; the server fails to start in a sandboxed or headless session
  ("No Metal device available"). The checker itself only makes HTTP calls.

## The vLLM side

The vLLM stack is not in this repo. It lives on the DGX Spark as `compose/live/vllm-prompt-injection-guard.yml`
(Portainer stack `vllm-prompt-injection-guard`, vLLM 0.30, `--runner pooling`, port 8009,
`--served-model-name prompt-injection-guard`). From the Open WebUI container: `guard_url`
`http://host.docker.internal:8009`, the filters' default.

## Comparing the two servers

`python serve/check_guard_server.py --url http://127.0.0.1:8194 --reference-url http://<spark>:8009` sends the same 15
texts to both and fails if P(INJECTION) differs by more than `--tolerance` (default 0.01). Not run yet.
