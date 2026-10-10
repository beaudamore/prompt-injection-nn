"""
title: RAG Injection Scan
author: Beau D'Amore
author_url: https://www.damore.ai
description: Scans every retrieved <source> chunk (knowledge bases, uploaded files, web search, OCR'd and transcribed files) and every native tool result for prompt injection before it reaches the model, using the prompt-injection guard served by the vllm-prompt-injection-guard stack; drops flagged chunks from the context and the citation list and blanks flagged tool results. No LLM call. Shadow (log-only) mode by default.
version: 0.3.0
requirements: 
note: self-contained. Nothing to install in the container; the only dependency is the guard service (vLLM, POST /classify) at guard_url.
openwebui: 0.11.4 (verified 2026-10-10, image ghcr.io/open-webui/open-webui:0.11.4; request hook and <source> rendering read from backend/open_webui/utils/middleware.py)
"""

import asyncio
import html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

log = logging.getLogger("rag_injection_scan")

# A retrieved chunk as Open WebUI 0.11.4 renders it (get_source_context in utils/middleware.py):
#   <source id="1" name="file.pdf" resource-type="file" resource-id="..." ...>chunk text</source>\n
# Attributes are html-escaped; the body is the raw chunk. One file's chunks share one id.
#
# Parsing is by opening tags, not by the first closing tag, for two reasons seen live on 2026-10-10:
# the RAG template's own prose contains a bare "<source>" and an example '<source id="1">' with no
# closing tag, and a chunk body is raw text that may itself contain "</source>" or "<source id=". So a
# block is: a real opening tag (numeric id) up to the LAST "</source>" before the next opening tag.
# Prose tags produce no block; a chunk containing fake tags stays one block and is scanned whole.
SOURCE_OPEN_RE = re.compile(r'<source\s+id="(\d+)"([^>]*)>')
SOURCE_CLOSE = "</source>"
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')

FILTER_BLOCK_AWARE = False   # this filter never blocks a turn; it edits the context and continues

# The guard classifies a chunk as SAFE or INJECTION and gives no finer reason. Every flagged chunk is
# reported with the indirect-injection reason from the prompt-injection-nn contract (docs/PLAN.md section 1).
REASON = "Embedded Instruction"

# A chunk longer than this is scored as overlapping pieces and the highest piece score is kept (the guard's
# windowed-max rule). 6,000 characters stays inside the service's 8,192-token limit for any text.
PIECE_CHARS = 6000
PIECE_OVERLAP = 1000
BATCH_SIZE = 64

# Native function calling puts each tool's output in a `tool`-role message and renders its <source> tag
# with an empty body, so the text is only ever in that message. It is scanned as one chunk and, when
# flagged in enforce mode, replaced by this notice: the message must stay (the model expects a reply
# for every tool call id), only its content goes.
TOOL_RESULT_REMOVED = "[tool result removed by RAG Injection Scan: flagged as prompt injection ({reason})]"


class Score(NamedTuple):
    probability: float   # P(INJECTION), the maximum over the chunk's pieces
    pieces: int          # how many pieces the chunk was scored as


def parse_attrs(attr_text: str) -> Dict[str, str]:
    return {k: html.unescape(v) for k, v in ATTR_RE.findall(attr_text or "")}


def find_source_blocks(text: str) -> List[Tuple[int, int, Dict[str, str], str]]:
    """(start, end, attrs, body) for every <source> block in `text`, in document order.

    `end` covers the closing tag and one trailing newline, so removing [start, end) removes the block."""
    text = text or ""
    opens = list(SOURCE_OPEN_RE.finditer(text))
    blocks = []
    for i, m in enumerate(opens):
        last = i + 1 == len(opens)
        region_end = len(text) if last else opens[i + 1].start()
        region = text[m.end(): region_end]
        close = region.rfind(SOURCE_CLOSE)
        if close < 0:
            continue   # an opening tag with no closing tag before the next one: template prose, not a chunk
        if last:
            # Only the template suffix (</context> ...) follows the real closer of the last block.
            body = region[:close]
            end = m.end() + close + len(SOURCE_CLOSE)
            if end < len(text) and text[end] == "\n":
                end += 1
        else:
            # Everything up to the next opening tag belongs to this chunk, fake closers included, so
            # text an attacker places after a fake "</source>" is still scanned and still removed.
            end = region_end
            body = region[: -len(SOURCE_CLOSE) - 1] if region.endswith(SOURCE_CLOSE + "\n") else (region[: -len(SOURCE_CLOSE)] if region.endswith(SOURCE_CLOSE) else region)
        blocks.append((m.start(), end, {"id": m.group(1), **parse_attrs(m.group(2))}, body))
    return blocks


def remove_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """Remove the given (start, end) spans from text; the rest is byte-identical."""
    out, cursor = [], 0
    for start, end in sorted(spans):
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def drop_from_sources(sources: Optional[List[dict]], bodies: List[str]) -> int:
    """Remove flagged chunk bodies from Open WebUI's `sources` list in place, so they are not cited.

    Each entry holds parallel lists `document`, `metadata` and (optionally) `distances`. The chunk is
    matched by exact body text. An entry whose document list becomes empty is removed."""
    if not sources or not bodies:
        return 0
    wanted = set(bodies)
    removed = 0
    for entry in list(sources):
        documents = entry.get("document")
        if not isinstance(documents, list):
            continue
        keep = [i for i, doc in enumerate(documents) if doc not in wanted]
        if len(keep) == len(documents):
            continue
        removed += len(documents) - len(keep)
        for key in ("document", "metadata", "distances"):
            values = entry.get(key)
            if isinstance(values, list) and len(values) == len(documents):
                entry[key] = [values[i] for i in keep]
        if not entry["document"]:
            sources.remove(entry)
    return removed


def _iter_text_parts(content: Any):
    """Yield (setter, text) for the text parts of a message content, which is a str or a list of parts."""
    if isinstance(content, str):
        yield (lambda holder, new: holder.__setitem__("content", new)), content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                yield (lambda holder, new, part=part: part.__setitem__("text", new)), part["text"]


def pieces_of(text: str, size: int = PIECE_CHARS, overlap: int = PIECE_OVERLAP) -> List[str]:
    """Overlapping character pieces covering the whole text; one piece when it fits."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    starts = list(range(0, len(text) - size + 1, step))
    if starts[-1] + size < len(text):
        starts.append(len(text) - size)
    return [text[s: s + size] for s in starts]


class GuardClient:
    """The guard service: vLLM serving a sequence-classification model with `--runner pooling`, POST /classify
    with {"model", "input": [...]} answering data[i].probs = [P(SAFE), P(INJECTION)] and data[i].label."""

    def __init__(self, url: str, model: str, timeout: float):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._injection_index: Optional[int] = None

    def _post(self, inputs: List[str]) -> List[dict]:
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        req = urllib.request.Request(self.url + "/classify", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"guard at {self.url} answered HTTP {exc.code} for a batch of {len(inputs)} (longest {max(len(t) for t in inputs)} chars): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"guard at {self.url} unreachable: {exc}") from exc
        data = sorted(payload["data"], key=lambda d: d["index"])
        if self._injection_index is None:
            self._injection_index = 1   # the guard's id2label is {0: SAFE, 1: INJECTION}
            for d in data:
                if str(d.get("label") or "").upper().startswith("INJ") and len(d["probs"]) >= 2:
                    self._injection_index = max(range(len(d["probs"])), key=lambda k: d["probs"][k])
                    break
        return data

    def score(self, texts: Sequence[str]) -> List[Score]:
        """One Score per text, in order. Empty or whitespace-only text is 0.0 with no pieces: nothing to inject."""
        probabilities = [0.0] * len(texts)
        pieces = [0] * len(texts)
        queue, owner = [], []
        for i, t in enumerate(texts):
            if isinstance(t, str) and t.strip():
                for piece in pieces_of(t, PIECE_CHARS, PIECE_OVERLAP):
                    queue.append(piece)
                    owner.append(i)
        for start in range(0, len(queue), BATCH_SIZE):
            for r, d in enumerate(self._post(queue[start: start + BATCH_SIZE])):
                i = owner[start + r]
                p = float(d["probs"][self._injection_index]) if len(d["probs"]) > 1 else float(d["probs"][0])
                pieces[i] += 1
                probabilities[i] = max(probabilities[i], p)
        return [Score(p, n) for p, n in zip(probabilities, pieces)]


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=-50, description="Run before other request filters so they see the cleaned context.")
        enabled: bool = True
        log_only: bool = Field(default=True, description="Shadow mode: score and report, but leave the context untouched. Set False to drop flagged chunks. Start in shadow mode for at least a week (docs/EVALUATION.md section 12).")
        guard_url: str = Field(default="http://host.docker.internal:8009", description="Base URL of the guard service: Portainer stack vllm-prompt-injection-guard, host port 8009. The Open WebUI container reaches the host through host.docker.internal.")
        guard_model: str = Field(default="prompt-injection-guard", description="The served model name the guard service expects in requests (--served-model-name in the stack).")
        threshold: float = Field(default=0.5, description="A chunk is flagged when the guard's P(INJECTION) is at or above this. 0.5 is the guard's own decision point. Raise it to drop fewer benign chunks, lower it to catch more.")
        timeout_s: float = Field(default=30.0, description="Seconds to wait for one batch of up to 64 pieces from the guard service.")
        scan_tool_results: bool = Field(default=True, description="Also scan native tool-call results (tool-role messages). In enforce mode a flagged result is replaced by a short notice; the message itself stays so the tool call is still answered.")
        show_status: bool = Field(default=True, description="Emit a status event naming dropped (or, in shadow mode, would-be-dropped) chunks, and one when the guard could not be reached and the context passed through unscanned.")
        debug: bool = Field(default=True, description="Log one line per scanned chunk at INFO: kind, source, size, pieces, score, threshold, verdict, action, text preview. On by default; this is the shadow-mode record.")
        log_text_chars: int = Field(default=160, description="Characters of each chunk's text to include in its log line (0 = none).")

    def __init__(self):
        self.valves = self.Valves()
        self._scorer: Optional[Callable[[Sequence[str]], List[Score]]] = None
        self._scorer_key = None

    # ---------------- scoring ----------------
    def _get_scorer(self) -> Callable[[Sequence[str]], List[Score]]:
        key = (self.valves.guard_url, self.valves.guard_model, self.valves.timeout_s)
        if self._scorer is None or self._scorer_key != key:
            self._scorer = GuardClient(self.valves.guard_url, self.valves.guard_model, self.valves.timeout_s).score
            self._scorer_key = key
        return self._scorer

    def set_scorer(self, scorer: Callable[[Sequence[str]], List[Score]]) -> None:
        """For tests and embedding: a function texts -> [Score, ...] used instead of the guard service."""
        self._scorer = scorer
        self._scorer_key = (self.valves.guard_url, self.valves.guard_model, self.valves.timeout_s)

    # ---------------- logging ----------------
    def _chunk_line(self, f: dict, s: Score, flagged: bool, mode: str) -> str:
        """One self-contained line per chunk, enough to explain the verdict without re-scoring it."""
        attrs = f["attrs"]
        parts = [f"rag_injection_scan: chunk kind={f['kind']}", f"id={attrs.get('id')}", f"name={attrs.get('name') or '-'}",
                 f"chars={len(f['chunk'])}", f"pieces={s.pieces}", f"p={s.probability:.3f}", f"threshold={self.valves.threshold:.3f}",
                 f"verdict={'INJECTION' if flagged else 'SAFE'}"]
        if flagged:
            parts.append(f'reason="{REASON}"')
            parts.append("action=" + ("would-drop (shadow mode)" if mode == "shadow" else ("blanked tool result" if f["kind"] == "tool" else "dropped from context and citations")))
        else:
            parts.append("action=none")
        n = max(0, int(self.valves.log_text_chars))
        if n:
            preview = " ".join(f["chunk"].split())[:n]
            parts.append(f'text="{preview}"')
        return " ".join(parts)

    # ---------------- the hook ----------------
    async def request(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not self.valves.enabled:
            return body
        messages = body.get("messages") or []
        # Collect every <source> block in system and user messages, with where it lives, and every
        # native tool result (a tool-role message) as one chunk.
        found: List[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role in ("system", "user"):
                for setter, text in _iter_text_parts(message.get("content")):
                    for start, end, attrs, chunk in find_source_blocks(text):
                        found.append({"kind": "source", "message": message, "setter": setter, "text": text, "span": (start, end), "attrs": attrs, "chunk": chunk})
            elif role == "tool" and self.valves.scan_tool_results:
                for setter, text in _iter_text_parts(message.get("content")):
                    if text.strip():
                        attrs = {"name": f"tool result {message.get('tool_call_id') or ''}".strip(), "id": message.get("tool_call_id")}
                        found.append({"kind": "tool", "message": message, "setter": setter, "text": text, "span": (0, len(text)), "attrs": attrs, "chunk": text})
        if not found:
            return body

        # Every chunk is scanned, every turn. There is no cap: a cap would let content through unscanned,
        # which is the one thing this filter exists to prevent. The time taken is in the turn log line.
        started = time.perf_counter()
        try:
            scorer = self._get_scorer()
            scores = await asyncio.to_thread(scorer, [f["chunk"] for f in found])
            if len(scores) != len(found):
                raise RuntimeError(f"scorer returned {len(scores)} scores for {len(found)} chunks")
        except Exception as exc:
            # The guard did not answer: say so in the log and the UI, and pass the context through unscanned.
            log.warning("rag_injection_scan: %d chunk(s) passed through UNSCANNED: %s", len(found), exc)
            if __event_emitter__ and self.valves.show_status:
                await __event_emitter__({"type": "status", "data": {"description": f"RAG injection scan is not active: {exc}. {len(found)} chunk(s) passed through unscanned.", "done": True}})
            return body
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            return await self._apply(body, found, scores, elapsed_ms, __event_emitter__, __metadata__)
        except Exception:
            # Nothing in this filter's own code may end a turn. Log it, pass the context through.
            log.exception("rag_injection_scan failed after scoring; passing the context through unchanged")
            return body

    async def _apply(self, body, scanned, scores, elapsed_ms, __event_emitter__, __metadata__) -> dict:
        threshold = float(self.valves.threshold)
        flagged = [(f, s) for f, s in zip(scanned, scores) if s.probability >= threshold]
        mode = "shadow" if self.valves.log_only else "enforce"
        log.info("rag_injection_scan: turn scanned=%d flagged=%d mode=%s guard=%s model=%s threshold=%.3f ms=%.1f",
                 len(scanned), len(flagged), mode, self.valves.guard_url, self.valves.guard_model, threshold, elapsed_ms)
        if self.valves.debug:
            for f, s in zip(scanned, scores):
                log.info(self._chunk_line(f, s, s.probability >= threshold, mode))

        if flagged and not self.valves.log_only:
            # Remove flagged <source> blocks from each text they live in, then from the citation sources;
            # replace flagged tool results with the notice.
            by_text: Dict[int, dict] = {}
            for f, _ in flagged:
                if f["kind"] == "tool":
                    f["setter"](f["message"], TOOL_RESULT_REMOVED.format(reason=REASON))
                    continue
                holder = by_text.setdefault(id(f["text"]), {"setter": f["setter"], "message": f["message"], "text": f["text"], "spans": []})
                holder["spans"].append(f["span"])
            for holder in by_text.values():
                holder["setter"](holder["message"], remove_spans(holder["text"], holder["spans"]))
            bodies = [f["chunk"] for f, _ in flagged if f["kind"] == "source"]
            for container in (body.get("metadata"), __metadata__):
                if isinstance(container, dict):
                    drop_from_sources(container.get("sources"), bodies)

        if __event_emitter__ and self.valves.show_status and flagged:
            names = ", ".join(f"{f['attrs'].get('name') or 'source ' + str(f['attrs'].get('id'))} ({REASON}, p={s.probability:.2f})" for f, s in flagged[:5])
            more = f" and {len(flagged) - 5} more" if len(flagged) > 5 else ""
            verb = "Would drop" if self.valves.log_only else "Dropped"
            what = "retrieved chunk(s) / tool result(s)" if any(f["kind"] == "tool" for f, _ in flagged) else "retrieved chunk(s)"
            await __event_emitter__({"type": "status", "data": {"description": f"{verb} {len(flagged)} {what} flagged as prompt injection: {names}{more}", "done": True}})
        return body
