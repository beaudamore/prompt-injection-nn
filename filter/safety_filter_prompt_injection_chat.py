"""
title: Prompt Injection Protection (Chat)
author: Beau D'Amore
author_url: https://www.damore.ai
description: Scores the prompt the user typed (the last user message) for prompt injection with the prompt-injection guard service before the model is called. On a hit in enforce mode it blocks the turn, logs the violation to a knowledge base, counts it against the user and locks the account at max_violations_count (admins exempt), with an optional lockout webhook. No LLM call. Shadow (log-only) mode by default.
version: 3.0.0
requirements:
note: self-contained. Nothing to install in the container; the only dependency is the guard service at guard_url: vLLM (the vllm-prompt-injection-guard stack on the DGX Spark) or the MLX server (serve/mlx, Apple Silicon), both answering POST /classify. The guard has not been measured on typed (chat-shaped) prompts, which is why this filter ships in shadow mode. Version 3.0.0 is safety_filter_prompt_injection_v2.py 2.1.0 (openwebui-safety-filters) with the LLM detector replaced by the guard.
openwebui: 0.13.0 (checked 2026-11-05 by reading the v0.13.0 source: every open_webui import and every call on the normal path exists there with the signature used here; the one exception is the older-version fallback in log_violation kept from v2, Knowledges.update_knowledge_data_by_id, which does not exist in 0.13.0 and is not reached there; NOT run in a live instance)
"""

import asyncio
import datetime
import inspect
import json
import logging
import os
import unicodedata
import urllib.error
import urllib.request
from tempfile import SpooledTemporaryFile
from typing import Any, Awaitable, Callable, List, NamedTuple, Optional, Sequence

from pydantic import BaseModel, Field

# Local Open WebUI imports (guarded for environments outside runtime)
try:
    from open_webui.models.knowledge import Knowledges, KnowledgeForm  # type: ignore
    from open_webui.models.users import Users  # type: ignore
    from open_webui.routers.files import upload_file_handler  # type: ignore
    from open_webui.routers.retrieval import process_file, ProcessFileForm  # type: ignore
    from open_webui.internal.db import AsyncSessionLocal  # type: ignore
    from fastapi import UploadFile  # type: ignore
    from fastapi.concurrency import run_in_threadpool  # type: ignore
except ImportError:  # pragma: no cover - guarded optional runtime deps
    Knowledges = None
    KnowledgeForm = None
    AsyncSessionLocal = None
    Users = None
    upload_file_handler = None
    process_file = None
    ProcessFileForm = None
    UploadFile = None
    run_in_threadpool = None

log = logging.getLogger("prompt_injection_chat")

FILTER_BLOCK_AWARE = True  # this filter records blocks in the shared record and defers to later block-aware filters

try:
    from open_webui.utils.filter import resolve_filter_pipeline  # type: ignore
    from open_webui.utils.plugin import get_function_module_from_cache  # type: ignore
except ImportError:  # pragma: no cover
    resolve_filter_pipeline = None
    get_function_module_from_cache = None

# Characters of the prompt included in the shadow-mode log line.
LOG_TEXT_CHARS = 160

# Shared with safety_filter_prompt_injection_retrieved_content.py: the three constants, Score, pieces_of and
# GuardClient below are the same code in both files (an Open WebUI function is a single file, so the copy is
# intentional). Change both together; tests/test_filter_chat.py compares them.

# A chunk longer than this is scored as overlapping pieces and the highest piece score is kept (the guard's
# windowed-max rule). One input over the service's 8,192-token limit makes it refuse the whole request
# (HTTP 400), which would let every chunk in that request through unscanned, so a piece must stay under the
# limit for any text. The guard's tokenizer applies NFKC first, which can turn one character into several
# (U+FDFA becomes 18), and then falls back to one token per UTF-8 byte for characters it does not know, at
# most four per character. pieces_of therefore applies NFKC itself before splitting: a 2,000-character piece
# is then at most 8,000 tokens plus three (measured 8,003 on 2026-11-05).
PIECE_CHARS = 2000
PIECE_OVERLAP = 1000
BATCH_SIZE = 64


class Score(NamedTuple):
    probability: float   # P(INJECTION), the maximum over the chunk's pieces
    pieces: int          # how many pieces the chunk was scored as


def pieces_of(text: str, size: int = PIECE_CHARS, overlap: int = PIECE_OVERLAP) -> List[str]:
    """Overlapping character pieces covering the whole text, NFKC-normalized as the guard's tokenizer does;
    one piece when it fits."""
    text = unicodedata.normalize("NFKC", text)
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    starts = list(range(0, len(text) - size + 1, step))
    if starts[-1] + size < len(text):
        starts.append(len(text) - size)
    return [text[s: s + size] for s in starts]


class GuardClient:
    """The guard service: vLLM (`--runner pooling`, the stack on the DGX Spark) or the MLX server (serve/mlx). Both take POST /classify
    with {"model", "input": [...]} and answer data[i].probs = [P(SAFE), P(INJECTION)], data[i].label and data[i].index."""

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


def prompt_text(content: Any) -> str:
    """The text of a message content: the string itself, or the text parts of a multimodal list joined by
    newlines. Image parts are not text and are not scored."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str))
    return ""


async def _call_openwebui(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    if run_in_threadpool:
        result = await run_in_threadpool(func, *args, **kwargs)
    else:
        result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class Filter:
    """
    Open WebUI Filter implementation for prompt injection protection of typed prompts.
    """

    # Also on the class: Open WebUI's function cache hands back the Filter instance, not the module.
    FILTER_BLOCK_AWARE = True

    class Valves(BaseModel):
        priority: int = -100
        enabled: bool = True
        log_only: bool = Field(default=True, description="Shadow mode: score the prompt and log what would have been blocked, but block nothing, count no violation, lock nobody out and send no webhook. Set False to enforce. On by default because the guard has not been measured on typed prompts.")
        guard_url: str = Field(default="http://host.docker.internal:8009", description="Base URL of the guard service: vLLM (DGX Spark; Portainer stack vllm-prompt-injection-guard, port 8009) or the MLX server (serve/mlx, Apple Silicon, port 8194), both answering POST /classify. The Open WebUI container reaches its own host through host.docker.internal.")
        guard_model: str = Field(default="prompt-injection-guard", description="The served model name the guard service expects in requests (--served-model-name in the stack).")
        threshold: float = Field(default=0.5, description="A prompt is flagged when the guard's P(INJECTION) is at or above this. 0.5 is the guard's own decision point. Raise it to block fewer benign prompts, lower it to catch more.")
        timeout_s: float = Field(default=30.0, description="Seconds to wait for one batch of up to 64 pieces from the guard service.")
        block_on_unsafe: bool = True
        block_mode: str = Field(
            default="message",
            description="'message': show block_message as the reply and end the turn without calling the model, so the chat stays usable. 'error': raise an error on the message (old behaviour).",
        )
        block_message: str = Field(
            default="⛔ This message was blocked by a safety filter and was not sent to the model. You can continue the conversation.",
            description="Static text shown as the assistant reply when a prompt is blocked (block_mode='message').",
        )
        enable_full_debug: bool = Field(
            default=False,
            description="Enable heavy debugging logs, including payloads/results (masked & truncated).",
        )
        enable_step_debug: bool = Field(
            default=False,
            description="Enable step-by-step progress logs (concise, truncated).",
        )
        violation_kb: str = Field(
            default="Prompt Injection Violations",
            description="Knowledge base name for logging violations.",
        )
        max_violations_count: int = Field(
            default=3,
            description="Maximum number of violations before user lockout.",
        )
        enable_webhook_notifications: bool = Field(
            default=False,
            description="Send a webhook notification when a user is locked out.",
        )
        notification_webhook_url_env: str = Field(
            default="PROMPT_INJECTION_WEBHOOK_URL",
            description="Environment variable containing the webhook URL.",
        )
        notification_webhook_subject: str = Field(
            default="Prompt injection lockout",
            description="Subject/title included in webhook notification payloads.",
        )
        notification_webhook_timeout: float = Field(
            default=10.0,
            description="Webhook request timeout in seconds.",
        )

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

    # Debugging helpers
    def _is_step_debug(self) -> bool:
        return bool(getattr(self.valves, "enable_step_debug", False) or getattr(self.valves, "enable_full_debug", False))

    def _is_full_debug(self) -> bool:
        return bool(getattr(self.valves, "enable_full_debug", False))

    def _truncate(self, text: Any, n: int = 200) -> Any:
        try:
            s = str(text)
        except Exception:
            return text
        if len(s) <= n:
            return s
        return s[: n - 1] + "…"

    def _print_safely(self, *parts: Any) -> None:
        safe_parts: List[Any] = []
        for p in parts:
            # Truncate long strings
            if isinstance(p, str):
                p = self._truncate(p)
            safe_parts.append(p)
        try:
            print(*safe_parts)
        except Exception:
            # Best-effort: avoid crashing due to logging
            pass

    def _dbg_step(self, *parts: Any) -> None:
        if self._is_step_debug():
            self._print_safely(*parts)

    def _dbg_full(self, *parts: Any) -> None:
        if self._is_full_debug():
            self._print_safely(*parts)

    async def _block_turn(self, __event_emitter__, body: Optional[dict] = None) -> None:
        """Show the static block message plus every recorded reason as the reply and end the
        turn without calling the model."""
        record = ((body or {}).get("metadata") or {}).get("filter_block") or {}
        reasons = [str(r) for r in record.get("reasons", []) if r]
        text = self.valves.block_message
        if reasons:
            text += "\n\n" + "\n".join(f"- {r[:1].upper()}{r[1:]}" for r in reasons)
        if __event_emitter__:
            await __event_emitter__({"type": "replace", "data": {"content": text}})
        raise asyncio.CancelledError("blocked by filter")

    def _record_block(self, body: dict, reason: str) -> None:
        """Record a violation in the shared block record (request metadata, never sent to the model).
        The prompt is left untouched so later filters scan the same original text."""
        meta = body.setdefault("metadata", {})
        record = meta.setdefault("filter_block", {"reasons": []})
        record.setdefault("reasons", []).append(reason)
        self._dbg_step(f"Block recorded: {reason}")

    async def _enforce_block_if_last(
        self, body, __event_emitter__, __request__, __model__, __metadata__, __id__
    ) -> None:
        """If anything was recorded and no later block-aware filter will run, show the static
        block message and end the turn. If the chain cannot be resolved, block immediately."""
        record = (body.get("metadata") or {}).get("filter_block")
        if not record or not record.get("reasons"):
            return
        try:
            if not (resolve_filter_pipeline and get_function_module_from_cache and __request__ and __model__):
                raise RuntimeError("filter chain helpers unavailable")
            enabled_ids = (__metadata__ or body.get("metadata") or {}).get("filter_ids", []) or []
            chain, _ = await resolve_filter_pipeline(__request__, __model__, enabled_ids)
            later = chain[chain.index(__id__) + 1 :] if __id__ in chain else []
            for fid in later:
                module, _, _ = await get_function_module_from_cache(__request__, fid)
                if getattr(module, "FILTER_BLOCK_AWARE", False):
                    self._dbg_step(f"Block recorded; deferring final block to later filter '{fid}'")
                    return
        except Exception as e:
            self._dbg_step(f"Could not resolve filter chain ({e}); blocking now")
        await self._block_turn(__event_emitter__, body)

    def _scrub_blocked_history(self, messages: list) -> list:
        """Drop earlier blocked user messages and their block replies so blocked content never reaches the model as history."""
        blocked = self.valves.block_message.strip()
        out: list = []
        for m in messages:
            content = m.get("content", "")
            if (
                m.get("role") == "assistant"
                and isinstance(content, str)
                and content.strip().startswith(blocked)
            ):
                if out and out[-1].get("role") == "user":
                    out.pop()
                continue
            out.append(m)
        return out

    def _safe_status_text(self, reason: str) -> str:
        """Status line for the UI. A failed check must not read as a pass."""
        if reason:
            return f"Prompt injection check FAILED (not enforced): {reason}"
        return "Prompt injection check complete: ✓ Safe"

    async def _get_or_create_kb(self, kb_name: str, user_id: str) -> Optional[str]:
        """Return the id of the knowledge base named kb_name, creating it if missing.
        the per-user knowledge lookup was removed from Open WebUI; list all and match by id or name."""
        if not Knowledges or not kb_name:
            return None
        try:
            for kb in await _call_openwebui(Knowledges.get_knowledge_bases):
                if kb.id == kb_name or kb.name == kb_name:
                    return kb.id
            if not user_id or not KnowledgeForm:
                self._dbg_step(f"Cannot create KB '{kb_name}': no user id or KnowledgeForm")
                return None
            created = await _call_openwebui(
                Knowledges.insert_new_knowledge,
                user_id,
                KnowledgeForm(
                    name=kb_name,
                    description="Auto-created by Prompt Injection Protection Filter for violation logging",
                ),
            )
            if created:
                self._dbg_step(f"Created KB '{kb_name}' ({created.id})")
                return created.id
            self._dbg_step(f"Failed to create KB '{kb_name}'")
        except Exception as e:
            self._dbg_step(f"KB lookup/create error for '{kb_name}': {e}")
        return None

    async def _disable_user(self, user_id: str) -> bool:
        """
        Disable a user account by updating their role to 'pending'.
        """
        self._dbg_step(f"Attempting to disable user {user_id}")
        try:
            # Try to update user role to 'pending'
            # Assuming Users.update_user_role_by_id exists
            if hasattr(Users, "update_user_role_by_id"):
                await _call_openwebui(Users.update_user_role_by_id, user_id, "pending")
                self._dbg_step(f"User {user_id} disabled via update_user_role_by_id")
                return True
            # Fallback: try update_user_by_id
            elif hasattr(Users, "update_user_by_id"):
                await _call_openwebui(Users.update_user_by_id, user_id, {"role": "pending"})
                self._dbg_step(f"User {user_id} disabled via update_user_by_id")
                return True
            else:
                self._dbg_step("Could not find method to disable user (Users.update_user_role_by_id or Users.update_user_by_id)")
                return False
        except Exception as e:
            self._dbg_step(f"Error disabling user: {e}")
            return False

    def _user_value(self, user: Optional[dict], key: str, default: str = "unknown") -> str:
        if not user:
            return default
        value = user.get(key, default) if isinstance(user, dict) else getattr(user, key, default)
        if value is None or value == "":
            return default
        return str(value)

    async def _send_webhook_notification(self, user: Optional[dict], content: str, reason: str) -> None:
        """
        Send a lockout notification to a configured webhook.
        """
        if not self.valves.enable_webhook_notifications:
            self._dbg_step("Webhook notifications disabled, skipping notification")
            return

        webhook_url = ""
        webhook_url_env = self.valves.notification_webhook_url_env.strip()
        if webhook_url_env:
            webhook_url = os.getenv(webhook_url_env, "").strip()

        if not webhook_url:
            self._dbg_step("No webhook URL found in configured environment variable, skipping notification")
            return

        try:
            user_id = self._user_value(user, "id")
            user_name = self._user_value(user, "name")
            user_email = self._user_value(user, "email")
            generic_payload = {
                "event": "prompt_injection_lockout",
                "subject": self.valves.notification_webhook_subject,
                "user_id": user_id,
                "user_name": user_name,
                "user_email": user_email,
                "reason": reason,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "content_preview": self._truncate(content, 1000),
            }
            body = json.dumps(generic_payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}

            def send():
                req = urllib.request.Request(
                    webhook_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.valves.notification_webhook_timeout) as response:
                    return response.status

            status = await run_in_threadpool(send) if run_in_threadpool else send()
            self._dbg_step(f"Webhook notification sent successfully: HTTP {status}")
        except Exception as e:
            self._dbg_step(f"Error sending webhook notification: {e}")

    async def _increment_violation_count(self, user_id: str) -> int:
        """
        Increments the persistent violation count for a user.
        Returns the new count.
        """
        if not Users:
            self._dbg_step("Users module not available, cannot persist violation count")
            return 0

        try:
            # 1. Get the current user object
            user = await _call_openwebui(Users.get_user_by_id, user_id)
            if not user:
                self._dbg_step(f"User {user_id} not found")
                return 0

            # 2. Access the existing 'info' dictionary
            # Handle cases where info might be None or missing
            current_info = getattr(user, "info", {}) or {}
            if not isinstance(current_info, dict):
                current_info = {}

            # 3. Increment the count
            current_count = current_info.get("violation_count", 0)
            new_count = current_count + 1
            current_info["violation_count"] = new_count

            # 4. Persist the changes
            await _call_openwebui(Users.update_user_by_id, user_id, {"info": current_info})
            self._dbg_step(f"Persisted violation count for {user_id}: {new_count}")

            return new_count

        except Exception as e:
            self._dbg_step(f"Error updating violation count for {user_id}: {e}")
            return 0

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
    ) -> dict:
        body = await self._inlet_impl(
            body,
            __user__=__user__,
            __event_emitter__=__event_emitter__,
            __request__=__request__,
        )
        await self._enforce_block_if_last(
            body, __event_emitter__, __request__, __model__, __metadata__, __id__
        )
        return body

    async def _inlet_impl(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
    ) -> dict:
        """
        Filter incoming user messages for prompt injection attempts.
        """
        self._dbg_full("Inlet called with body:", body)
        if not self.valves.enabled:
            self._dbg_step("Inlet skipped: Disabled")
            return body

        messages = self._scrub_blocked_history(body.get("messages", []))
        body["messages"] = messages
        if not messages:
            self._dbg_step("Inlet skipped: No messages")
            return body

        # Check the last user message
        last_message = messages[-1]
        if last_message.get("role") == "user":
            user_content = last_message.get("content", "")
            self._dbg_step("Checking user content:", self._truncate(user_content, 100))

            if __event_emitter__:
                self._dbg_step("Emitting status: Checking for prompt injection...")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Checking for prompt injection...",
                            "done": False,
                        },
                    }
                )

            # Prompt injection detection by the guard service (no LLM call)
            is_injection_safe, injection_reason, score = await self._detect_prompt_injection_guard(user_content)
            self._dbg_step(f"Injection check result: is_safe={is_injection_safe} reason={injection_reason}")

            if not is_injection_safe and self.valves.log_only:
                # Shadow mode: say what would have been blocked and change nothing.
                preview = " ".join(prompt_text(user_content).split())[:LOG_TEXT_CHARS]
                log.info(
                    'prompt_injection_chat: would block (shadow mode) user=%s chars=%d pieces=%d p=%.3f threshold=%.3f text="%s"',
                    __user__.get("id") if __user__ else "unknown",
                    len(prompt_text(user_content)),
                    score.pieces,
                    score.probability,
                    float(self.valves.threshold),
                    preview,
                )
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Prompt injection check complete: would block (shadow mode): {injection_reason}",
                                "done": True,
                            },
                        }
                    )
                return body

            if not is_injection_safe and self.valves.block_on_unsafe:
                model_name = body.get("model", "unknown")
                await self.log_violation(__user__, user_content, injection_reason, __request__, model_name=model_name)

                # Lockout Logic - Skip for admin users
                user_id = __user__.get("id") if __user__ else "unknown"
                user_role = __user__.get("role") if __user__ else "user"

                if user_id != "unknown" and user_role != "admin":
                    # Use persistent counter
                    violation_count = await self._increment_violation_count(user_id)
                    self._dbg_step(f"User {user_id} violation count: {violation_count}/{self.valves.max_violations_count}")

                    if violation_count >= self.valves.max_violations_count:
                        self._dbg_step(f"User {user_id} reached max violations. Locking out.")
                        if await self._disable_user(user_id):
                            await self._send_webhook_notification(__user__, user_content, injection_reason)
                        else:
                            self._dbg_step(f"Failed to disable user {user_id}")
                elif user_role == "admin":
                    self._dbg_step(f"Skipping lockout for admin user {user_id}")

                self._dbg_step(f"Blocking input content: Prompt Injection detected - {injection_reason}")
                if __event_emitter__:
                    self._dbg_step("Emitting status: Content blocked - prompt injection detected")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Content blocked - prompt injection detected: {injection_reason}",
                                "done": True,
                            },
                        }
                    )
                if self.valves.block_mode == "error":
                    raise ValueError(f"Content blocked - prompt injection detected: {injection_reason}")
                self._record_block(body, f"prompt injection: {injection_reason}")
                return body

            if __event_emitter__:
                self._dbg_step("Emitting status: Prompt injection check complete")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": self._safe_status_text(injection_reason),
                            "done": True,
                        },
                    }
                )

        return body

    async def log_violation(
        self, user: Optional[dict], content: str, reason: str, request: Optional[Any] = None, model_name: Optional[str] = None
    ) -> None:
        """
        Log violation to the configured Knowledge Base.
        """
        self._dbg_step("log_violation called")

        # 1. Basic local logging (always do this)
        try:
            timestamp = datetime.datetime.now().isoformat()
            record = {
                "user_id": user.get("id", "unknown") if user else "unknown",
                "datetime": timestamp,
                "model": model_name or "unknown",
                "reason": reason,
                "content": self._truncate(content, 400),
            }
            self._dbg_full("Violation Record:", record)
        except Exception as e:
            self._dbg_step(f"log_violation local logging error: {e}")

        # 2. Remote KB logging
        if not self.valves.violation_kb or self.valves.violation_kb.lower() == "none":
            self._dbg_step("Violation KB logging disabled (violation_kb not set)")
            return

        if not all([upload_file_handler, process_file, Knowledges, Users, AsyncSessionLocal]):
            self._dbg_step("Required OpenWebUI modules not available for KB logging")
            return

        if not request or not user:
            self._dbg_step("Missing request or user context for KB logging")
            return

        try:
            # Resolve User object
            user_obj = await _call_openwebui(Users.get_user_by_id, str(user["id"]))
            if not user_obj:
                self._dbg_step("Could not resolve User object")
                return

            # Find KB
            kb_name = self.valves.violation_kb.strip()
            kb_id = None

            # Find by id or name, creating it if missing
            kb_id = await self._get_or_create_kb(kb_name, user_obj.id)
            if not kb_id:
                self._dbg_step(f"Violation KB '{kb_name}' not found and could not be created")
                return

            # Prepare content
            full_log_content = (
                f"--- Prompt Injection Violation Report ---\n"
                f"Timestamp: {timestamp}\n"
                f"User ID: {user.get('id', 'unknown')}\n"
                f"User Name: {user.get('name', 'unknown')}\n"
                f"User Email: {user.get('email', 'unknown')}\n"
                f"Model: {model_name or 'unknown'}\n"
                f"Reason: {reason}\n"
                f"--- Content ---\n"
                f"{content}\n"
                f"-------------------------------\n"
            )

            filename = f"injection_violation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            # Upload File
            upload = UploadFile(
                filename=filename,
                file=SpooledTemporaryFile(max_size=1024 * 1024),
                headers={"content-type": "text/plain"},
            )
            upload.file.write(full_log_content.encode("utf-8"))
            upload.file.seek(0)

            try:
                file_data = await _call_openwebui(
                    upload_file_handler,
                    request,
                    upload,
                    {"source": "prompt_injection_filter", "type": "violation_report"}, # metadata
                    False, # process
                    False, # process_in_background
                    user_obj,
                    None,
                )
            finally:
                await upload.close()

            # Handle Pydantic vs Dict
            file_id = getattr(file_data, "id", None)
            if file_id is None and isinstance(file_data, dict):
                file_id = file_data.get("id")

            if not file_id:
                self._dbg_step("Failed to upload violation report file")
                return

            # Attach to KB
            try:
                await _call_openwebui(
                    Knowledges.add_file_to_knowledge_by_id,
                    kb_id,
                    file_id,
                    user_obj.id,
                )
            except AttributeError:
                 # Fallback for older versions
                knowledge = await _call_openwebui(Knowledges.get_knowledge_by_id, id=kb_id)
                if knowledge:
                    data = getattr(knowledge, "data", None) or {}
                    file_ids = data.get("file_ids", [])
                    if file_id not in file_ids:
                        file_ids.append(file_id)
                        data["file_ids"] = file_ids
                        await _call_openwebui(Knowledges.update_knowledge_data_by_id, id=kb_id, data=data)

            # Process (index) the file. process_file needs an explicit session now that
            # FastAPI's Depends is not injecting one for direct calls.
            async with AsyncSessionLocal() as db:
                await process_file(
                    request,
                    ProcessFileForm(file_id=file_id, collection_name=kb_id, content=full_log_content),
                    user_obj,
                    db=db,
                )

            self._dbg_step(f"Violation logged to KB '{kb_name}' (File ID: {file_id})")

        except Exception as e:
            self._dbg_step(f"Error logging violation to KB: {e}")

    async def _detect_prompt_injection_guard(self, content: Any) -> tuple[bool, str, Optional[Score]]:
        """
        Prompt injection detection by the guard service. The blocking HTTP call runs in a worker thread.

        Returns:
            tuple: (is_safe: bool, reason: str, score: Score or None when the check failed)
        """
        self._dbg_step(f"Starting guard injection detection for content: '{content}'")
        self._dbg_step(f"Using guard: '{self.valves.guard_url}' model '{self.valves.guard_model}'")

        try:
            scorer = self._get_scorer()
            scores = await asyncio.to_thread(scorer, [prompt_text(content)])
            if len(scores) != 1:
                raise RuntimeError(f"scorer returned {len(scores)} scores for 1 prompt")
            score = scores[0]
            threshold = float(self.valves.threshold)
            self._dbg_step(f"Guard score: p={score.probability:.3f} pieces={score.pieces} threshold={threshold}")

            if score.probability >= threshold:
                # The guard gives no reason, only a score.
                return False, f"Prompt injection (guard score {score.probability:.3f}, threshold {threshold})", score
            return True, "", score

        except Exception as e:
            self._dbg_step(f"Injection detection exception: {e}")
            # Fail-open: do not block on unexpected exceptions, but report it
            return True, f"check failed: {e}", None
