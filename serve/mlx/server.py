"""Prompt-injection guard server on MLX (Apple Silicon): the same POST /classify the vLLM stack answers.

Serves a Hugging Face ModernBERT sequence-classification guard (built for Horizon-Labs/prompt-injection-guard-base,
labels SAFE / INJECTION) with mlx-embeddings, so the Open WebUI filters in ../../filter work against this server or
against vLLM (`--runner pooling`, the stack on the DGX Spark) by changing only their `guard_url` valve.

  - POST /classify  {"model": "<served name>", "input": "text" | ["text", ...]}
      -> {"id", "object": "list", "created", "model", "data": [{"index", "label", "probs": [P(SAFE), P(INJECTION)],
          "num_classes"}], "usage": {"prompt_tokens", "total_tokens"}}
    probs are the softmax over the model's labels in id2label order, as vLLM returns them. "model" may be omitted;
    a name other than GUARD_SERVED_NAME is a 404. Other request fields (vLLM's truncate_prompt_tokens, ...) are
    ignored: nothing is truncated here.
  - An input longer than GUARD_MAX_TOKENS tokens (special tokens included) is a 400 for the whole request, not a
    silent truncation; vLLM does the same. The tokenizer applies NFKC and falls back to one token per byte for
    characters outside its vocabulary, so a short text can cost many tokens. Since 2026-11-05 the filters apply NFKC
    themselves and split text into pieces of at most 2,000 characters (6,000 before), which kept every piece measured
    under the limit (8,003 tokens for 2,000 x U+10FFFD). The filters fail open on any guard error.
  - GET /health     {"status": "ok", model, served name, labels, limits, versions}
  - GET /v1/models  the served name, in the OpenAI list shape.
  - The model is resolved offline from the Hugging Face cache in GUARD_CACHE and loaded at startup; the process
    never reaches the network.
  - One forward pass at a time (one GPU job at a time). Inputs are sorted by length and run in sub-batches sized so
    that sequences x longest_length^2 stays under GUARD_BATCH_BUDGET: attention memory grows with the square of the
    length, so ten 300-token chunks go through in one pass while a 2,000-token piece runs alone. The lock is taken
    per pass, so the passes of two simultaneous requests alternate instead of one request waiting for all of another.
  - Scores follow Hugging Face transformers: exact GELU and the model's own rotary settings (see Engine and
    rope_overrides); serve/check_guard_server.py --torch-model measures the difference.
  - Logging to stderr: one "req N" line per request with counts, time and the highest P(INJECTION). Input text is
    never logged.
Settings come from the environment (see serve/README.md); on steppenwolf they are exported by
local-llm-config/prompt-injection-guard-mlx/guardctl from guard.env.
"""

import itertools
import logging
import os
import sys
import threading
import time
import uuid
from importlib.metadata import version as package_version

MODEL_ID = os.environ.get("GUARD_MODEL", "Horizon-Labs/prompt-injection-guard-base")
# Offline, from the given Hugging Face cache; must be set before huggingface_hub / transformers are imported.
if os.environ.get("GUARD_CACHE"):
    os.environ["HF_HUB_CACHE"] = os.environ["GUARD_CACHE"]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from mlx_embeddings.utils import get_model_path, load, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("prompt-injection-guard")

SERVED_NAME = os.environ.get("GUARD_SERVED_NAME", "prompt-injection-guard")
MAX_TOKENS = int(os.environ.get("GUARD_MAX_TOKENS", "8192"))               # the model's max_position_embeddings
BATCH_BUDGET = int(os.environ.get("GUARD_BATCH_BUDGET", str(16 * 512 * 512)))  # sequences x longest_length^2 per forward pass
CACHE_LIMIT_MB = int(os.environ.get("GUARD_CACHE_LIMIT_MB", "1024"))
MB = 1e6


def _mem() -> str:
    return f"active={mx.get_active_memory() / MB:.0f}MB cache={mx.get_cache_memory() / MB:.0f}MB"


def rope_overrides(config: dict) -> dict:
    """Rotary settings for mlx-embeddings from a transformers 5 config.

    transformers 5 writes them as rope_parameters {"full_attention": {"rope_theta"}, "sliding_attention": {"rope_theta"}};
    mlx-embeddings 0.1.0 reads only the older global_rope_theta / local_rope_theta keys and otherwise falls back to
    160000 / 10000. This model uses 160000 for both, so without the mapping every sliding-attention layer would
    rotate with the wrong base and the scores would be wrong without any error.
    """
    if "rope_parameters" not in config or "global_rope_theta" in config:
        return {}
    try:
        return {"global_rope_theta": float(config["rope_parameters"]["full_attention"]["rope_theta"]),
                "local_rope_theta": float(config["rope_parameters"]["sliding_attention"]["rope_theta"])}
    except (KeyError, TypeError) as e:
        raise SystemExit(f"{MODEL_ID}: config.json has rope_parameters without full_attention / sliding_attention rope_theta ({e!r})")


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str, param: str | None = None):
        self.status, self.message, self.kind, self.param = status, message, kind, param


class Engine:
    """The loaded classifier and its tokenizer."""

    def __init__(self):
        t0 = time.time()
        model_path = get_model_path(MODEL_ID)
        self.model, wrapper = load(str(model_path), model_config=rope_overrides(load_config(model_path)))
        # mlx-embeddings 0.1.0 builds every GELU as the tanh approximation (nn.GELU(approx="precise")); Hugging Face
        # ModernBERT uses the exact erf GELU in all 22 MLPs and in the head. Switch them to exact so the scores are
        # the reference's, not an approximation of them. nn.GELU reads _approx on every call.
        gelus = [m for _, m in self.model.named_modules() if isinstance(m, nn.GELU)]
        for m in gelus:
            m._approx = "none"
        # The Hugging Face fast tokenizer; its normalizer is part of tokenizer.json. That file also carries a
        # truncation setting (max_length 1024 for this model); truncation=False clears it. Clearing it changes the
        # shared Rust tokenizer, which is not safe while another thread is encoding, so it is done once here: every
        # later call passes the same options and changes nothing.
        self.tokenizer = wrapper._tokenizer
        self.tokenizer(["warm-up"], add_special_tokens=True, truncation=False, padding=False)
        config = self.model.config
        # mlx-embeddings builds the classification head only for this architecture; anything else would answer
        # with embeddings or nothing where the scores should be. Refuse instead.
        if config.architectures != ["ModernBertForSequenceClassification"]:
            raise SystemExit(f"{MODEL_ID} is {config.architectures}, not a ModernBertForSequenceClassification model")
        id2label = {int(k): str(v) for k, v in (config.id2label or {}).items()}
        self.labels = [id2label[i] for i in range(len(id2label))]
        if len(self.labels) < 2:
            raise SystemExit(f"{MODEL_ID} has labels {self.labels}; a classifier with at least two labels is required")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            raise SystemExit(f"the tokenizer of {MODEL_ID} has no pad token")
        self.lock = threading.Lock()
        log.info("loaded %s in %.1fs labels=%s exact_gelu=%d %s", MODEL_ID, time.time() - t0, self.labels, len(gelus), _mem())

    def _batches(self, lengths: list[int]) -> list[list[int]]:
        """Input positions grouped for one forward pass each: shortest first, so padding is small, and
        sequences x longest_length^2 within BATCH_BUDGET (a sequence longer than that runs alone)."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches: list[list[int]] = []
        current: list[int] = []
        for i in order:
            if current and (len(current) + 1) * lengths[i] ** 2 > BATCH_BUDGET:
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
        return batches

    def classify(self, texts: list[str]) -> tuple[list[list[float]], int, int, int]:
        """probs per text in input order, total tokens, longest input in tokens, forward passes."""
        ids = self.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        lengths = [len(x) for x in ids]
        for i, n in enumerate(lengths):
            if n > MAX_TOKENS:
                raise ApiError(400, f"input {i} is {n} tokens; this model's maximum context length is {MAX_TOKENS} tokens", "BadRequestError", "input")
        probs: list[list[float]] = [[] for _ in texts]
        batches = self._batches(lengths)
        for batch in batches:
            width = max(lengths[i] for i in batch)
            with self.lock:  # one forward pass at a time on the GPU; taken per pass so a long request does not hold up a short one
                input_ids = mx.array([ids[i] + [self.pad_id] * (width - lengths[i]) for i in batch])
                mask = mx.array([[1.0] * lengths[i] + [0.0] * (width - lengths[i]) for i in batch])
                out = self.model(input_ids, attention_mask=mask).pooler_output  # softmax over the labels
                mx.eval(out)
                rows = out.tolist()
            for i, row in zip(batch, rows):
                probs[i] = [float(p) for p in row]
        return probs, sum(lengths), max(lengths, default=0), len(batches)


class ClassifyRequest(BaseModel):
    model: str | None = None
    input: str | list[str]


app = FastAPI(title="prompt-injection-guard-mlx")
engine: Engine | None = None
_req_seq = itertools.count(1)


@app.exception_handler(ApiError)
async def api_error(_: Request, e: ApiError):
    return JSONResponse(status_code=e.status, content={"error": {"message": e.message, "type": e.kind, "param": e.param, "code": e.status}})


@app.post("/classify")
async def classify(payload: ClassifyRequest, request: Request):
    rid = next(_req_seq)
    client = request.client.host if request.client else "?"
    t0 = time.time()
    try:
        if payload.model and payload.model != SERVED_NAME:
            raise ApiError(404, f"The model `{payload.model}` does not exist.", "NotFoundError", "model")
        texts = [payload.input] if isinstance(payload.input, str) else list(payload.input)
        if not texts:
            probs, tokens, longest, passes = [], 0, 0, 0
        else:
            probs, tokens, longest, passes = await run_in_threadpool(engine.classify, texts)
    except ApiError as e:
        log.warning("req %d from %s rejected %d: %s", rid, client, e.status, e.message)
        raise
    except Exception as e:
        log.exception("req %d failed", rid)
        raise ApiError(500, f"{type(e).__name__}: {e}", "InternalServerError")
    data = [{"index": i, "label": engine.labels[max(range(len(p)), key=p.__getitem__)], "probs": p, "num_classes": len(p)}
            for i, p in enumerate(probs)]
    injection = [p[engine.labels.index(label)] for p in probs for label in engine.labels if label.upper().startswith("INJ")]
    log.info("req %d from %s: inputs=%d tokens=%d longest=%d passes=%d ms=%.0f max_p_injection=%s %s", rid, client, len(texts), tokens,
             longest, passes, (time.time() - t0) * 1000, f"{max(injection):.3f}" if injection else "n/a", _mem())
    return {"id": f"classify-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()), "model": SERVED_NAME, "data": data,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "created": 0, "owned_by": "system", "root": MODEL_ID}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "served_name": SERVED_NAME, "labels": engine.labels, "max_tokens": MAX_TOKENS,
            "batch_budget": BATCH_BUDGET, "mlx": mx.__version__, "mlx_embeddings": package_version("mlx-embeddings")}


if __name__ == "__main__":
    previous = mx.set_cache_limit(CACHE_LIMIT_MB * 2**20)
    log.info("guard starting: mlx-embeddings %s, mlx %s, model %s (cache %s), served as %s, max %d tokens, batch budget %d, MLX cache limit %d MB (was %.0f MB)",
             package_version("mlx-embeddings"), mx.__version__, MODEL_ID, os.environ.get("HF_HUB_CACHE", "default"), SERVED_NAME, MAX_TOKENS,
             BATCH_BUDGET, CACHE_LIMIT_MB, previous / MB)
    engine = Engine()
    uvicorn.run(app, host=os.environ.get("GUARD_HOST", "127.0.0.1"), port=int(os.environ.get("PORT_GUARD", "8194")), workers=1,
                loop="asyncio", log_config=None)
