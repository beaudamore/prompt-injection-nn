"""
title: RAG Injection Scan
author: Beau D'Amore
author_url: https://www.damore.ai
description: Scans every retrieved <source> chunk (knowledge bases, uploaded files, web search, tool results) for prompt injection with the prompt-injection-nn classifier before it reaches the model, and drops flagged chunks from the context and the citation list. No LLM call. Shadow (log-only) mode by default.
version: 0.1.0
requirements: prompt-injection-nn
openwebui: 0.11.4 (verified 2026-10-10, image ghcr.io/open-webui/open-webui:0.11.4; request hook and <source> rendering read from backend/open_webui/utils/middleware.py)
"""

import asyncio
import html
import logging
import re
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

log = logging.getLogger("rag_injection_scan")

# A retrieved chunk as Open WebUI 0.11.4 renders it (get_source_context in utils/middleware.py):
#   <source id="1" name="file.pdf" resource-type="file" resource-id="..." ...>chunk text</source>\n
# Attributes are html-escaped; the body is the raw chunk. One file's chunks share one id.
SOURCE_BLOCK_RE = re.compile(r"<source\b([^>]*)>(.*?)</source>\n?", re.DOTALL)
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')

FILTER_BLOCK_AWARE = False   # this filter never blocks a turn; it edits the context and continues


def parse_attrs(attr_text: str) -> Dict[str, str]:
    return {k: html.unescape(v) for k, v in ATTR_RE.findall(attr_text or "")}


def find_source_blocks(text: str) -> List[Tuple[int, int, Dict[str, str], str]]:
    """(start, end, attrs, body) for every <source> block in `text`, in document order."""
    return [(m.start(), m.end(), parse_attrs(m.group(1)), m.group(2)) for m in SOURCE_BLOCK_RE.finditer(text or "")]


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


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=-50, description="Run before other request filters so they see the cleaned context.")
        enabled: bool = True
        log_only: bool = Field(default=True, description="Shadow mode: score and report, but leave the context untouched. Set False to drop flagged chunks. Start in shadow mode for at least a week (docs/EVALUATION.md section 12).")
        backend: str = Field(default="", description="prompt-injection-nn backend: '' (the Phase 3 gate.json recommendation), 'cnn_bigru', 'tfidf_logreg' or 'ensemble_mean'.")
        budget: str = Field(default="retrieval_fpr_0.01", description="Operating point. 'retrieval_fpr_0.01' / '_0.005' / '_0.001': thresholds chosen on validation document and agentic rows, which is what this filter sees. 'fpr_*' variants are chosen on all validation rows including chat. Pick from the operator's dropped-benign-chunk budget, never 'f1'.")
        models_dir: str = Field(default="", description="Directory holding the phase2-*/ and phase3-cnn-bigru-*/ run folders. Empty: PI_NN_MODELS_DIR or the package's own resolution.")
        package_path: str = Field(default="", description="Optional path to add to sys.path so `prompt_injection_nn` imports (for example the repo's src/ folder mounted into the container).")
        max_chunks: int = Field(default=64, description="Scan at most this many chunks per turn; the rest pass through and a status line says so.")
        max_windows: int = Field(default=4, description="For a chunk longer than the model window (512 tokens), score at most this many overlapping windows (first, last, evenly spread). Bounds latency on full-context uploads.")
        torch_threads: int = Field(default=0, description="CPU threads for the CNN; 0 = package default (half the cores, at most 8).")
        show_status: bool = Field(default=True, description="Emit a status event naming dropped (or, in shadow mode, would-be-dropped) chunks.")
        debug: bool = Field(default=False, description="Log every chunk's probability at INFO level.")

    def __init__(self):
        self.valves = self.Valves()
        self._detector = None
        self._detector_key = None

    # ---------------- model loading ----------------
    def _get_detector(self):
        key = (self.valves.backend, self.valves.budget, self.valves.models_dir, self.valves.package_path, self.valves.max_windows, self.valves.torch_threads)
        if self._detector is not None and self._detector_key == key:
            return self._detector
        if self.valves.package_path and self.valves.package_path not in sys.path:
            sys.path.insert(0, self.valves.package_path)
        from prompt_injection_nn import load_detector  # imported lazily so the filter loads without the package

        self._detector = load_detector(models_dir=self.valves.models_dir or None, backend=self.valves.backend or None, budget=self.valves.budget,
                                       max_windows=self.valves.max_windows or None, torch_threads=self.valves.torch_threads or None)
        self._detector_key = key
        log.info("rag_injection_scan loaded backend=%s budget=%s threshold=%.4f", self._detector.backend.name, self._detector.budget, self._detector.threshold)
        return self._detector

    def set_detector(self, detector) -> None:
        """For tests and embedding: inject a detector with classify_batch(texts) -> verdicts."""
        self._detector = detector
        self._detector_key = (self.valves.backend, self.valves.budget, self.valves.models_dir, self.valves.package_path, self.valves.max_windows, self.valves.torch_threads)

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
        # Collect every <source> block in system and user messages, with where it lives.
        found: List[dict] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user"):
                continue
            for setter, text in _iter_text_parts(message.get("content")):
                for start, end, attrs, chunk in find_source_blocks(text):
                    found.append({"message": message, "setter": setter, "text": text, "span": (start, end), "attrs": attrs, "chunk": chunk})
        if not found:
            return body

        scanned = found[: self.valves.max_chunks]
        skipped = len(found) - len(scanned)
        started = time.perf_counter()
        try:
            detector = await asyncio.to_thread(self._get_detector)
            verdicts = await asyncio.to_thread(detector.classify_batch, [f["chunk"] for f in scanned])
        except Exception:
            log.exception("rag_injection_scan failed; passing the context through unchanged")
            return body
        elapsed_ms = (time.perf_counter() - started) * 1000

        flagged = [(f, v) for f, v in zip(scanned, verdicts) if v.is_injection]
        if self.valves.debug:
            for f, v in zip(scanned, verdicts):
                log.info("chunk id=%s name=%s p=%.4f -> %s", f["attrs"].get("id"), f["attrs"].get("name"), v.probability, v.contract)
        log.info("rag_injection_scan: %d chunk(s) scanned in %.1f ms, %d flagged, mode=%s", len(scanned), elapsed_ms, len(flagged), "shadow" if self.valves.log_only else "enforce")

        if flagged and not self.valves.log_only:
            # Remove flagged blocks from each text they live in, then from the citation sources.
            by_text: Dict[int, dict] = {}
            for f, _ in flagged:
                holder = by_text.setdefault(id(f["text"]), {"setter": f["setter"], "message": f["message"], "text": f["text"], "spans": []})
                holder["spans"].append(f["span"])
            for holder in by_text.values():
                holder["setter"](holder["message"], remove_spans(holder["text"], holder["spans"]))
            bodies = [f["chunk"] for f, _ in flagged]
            for container in (body.get("metadata"), __metadata__):
                if isinstance(container, dict):
                    drop_from_sources(container.get("sources"), bodies)

        if __event_emitter__ and self.valves.show_status and (flagged or skipped):
            names = ", ".join(f"{f['attrs'].get('name') or 'source ' + str(f['attrs'].get('id'))} ({v.reason})" for f, v in flagged[:5])
            more = f" and {len(flagged) - 5} more" if len(flagged) > 5 else ""
            if flagged:
                verb = "Would drop" if self.valves.log_only else "Dropped"
                description = f"{verb} {len(flagged)} retrieved chunk(s) flagged as prompt injection: {names}{more}"
            else:
                description = ""
            if skipped:
                description = (description + "; " if description else "") + f"{skipped} chunk(s) over the scan limit passed through unscanned"
            await __event_emitter__({"type": "status", "data": {"description": description, "done": True}})
        return body
