"""Model backends. Each loads one run directory written by a notebook and scores texts on CPU.

- CnnBackend:      models/phase3-cnn-bigru-<version>-<fingerprint>/   (model_state.pt, vocab.json, tokenizer.json, config.json, thresholds.json)
- TfidfBackend:    models/phase2-<version>-<fingerprint>/             (tfidf_word.joblib, tfidf_char.joblib, logreg.joblib, logreg_calibrator.joblib, thresholds.json)
- EnsembleBackend: mean of the two calibrated probabilities; reasons from the CNN. Thresholds from the
                   Phase 3 thresholds.json "candidates"["ensemble_mean"] entry.
- TransformerBackend: a Hugging Face SAFE/INJECTION classifier from a local directory (models/hf/<name>/),
                   such as Horizon-Labs/prompt-injection-guard-base; no reason head, so the Detector borrows
                   the CNN's reason head for flagged rows.

Every backend exposes `name`, `thresholds` ({"f1": t, "fpr_0.01": t, ...}), `has_reasons`, and
`predict(texts) -> (probabilities, reason_logits or None, windows_per_text, components)` where components
maps each underlying model's name to its own probabilities, so a log line can show how a score was made.
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contract import REASONS
from .tokenizer import Tokenizer


def _budget_thresholds(table: dict, prefix: str = "") -> Dict[str, float]:
    return {prefix + budget: float(row["threshold"]) for budget, row in table.items()}


def _all_thresholds(f1: float, all_rows: dict, retrieval: Optional[dict]) -> Dict[str, float]:
    # "fpr_0.01": chosen on all validation rows. "retrieval_fpr_0.01": chosen on validation document and
    # agentic rows only, the retrieval filter's input, and the filter's default.
    out = {"f1": float(f1), **_budget_thresholds(all_rows)}
    if retrieval:
        out.update(_budget_thresholds(retrieval, "retrieval_"))
    return out


class CnnBackend:
    name = "cnn_bigru"
    has_reasons = True

    def __init__(self, run_dir: Path, window_stride: Optional[int] = None, torch_threads: Optional[int] = None, max_windows: Optional[int] = None):
        import torch
        from .model import TextCnnBiGru

        self.run_dir = Path(run_dir)
        config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        assert tuple(config["reasons"]) == REASONS, "reason taxonomy in the run differs from the package contract"
        self.tokenizer = Tokenizer.from_run_dir(self.run_dir)
        self.model = TextCnnBiGru(len(self.tokenizer.vocab), self.tokenizer.pad_index, len(REASONS), config["text"])
        state = torch.load(self.run_dir / "model_state.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.window_stride = int(window_stride) if window_stride else max(1, self.tokenizer.max_tokens // 2)
        # Cap on windows per text (None = score every window). Open WebUI chunks are about 1,000 characters,
        # a single window; the cap bounds latency on unusually long chunks such as full-context uploads.
        self.max_windows = int(max_windows) if max_windows else None
        thresholds = json.loads((self.run_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = _all_thresholds(thresholds["threshold"], thresholds["fpr_budget_thresholds_val"], thresholds.get("fpr_budget_thresholds_val_retrieval"))
        self.run_manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8")) if (self.run_dir / "run_manifest.json").exists() else {}
        # Thread count: the benchmark in the Phase 3 notebook used half the cores. Too many threads on a
        # small batch costs more in contention than it gains; PI_NN_TORCH_THREADS overrides.
        threads = torch_threads or os.environ.get("PI_NN_TORCH_THREADS") or min(8, max(1, (os.cpu_count() or 2) // 2))
        torch.set_num_threads(int(threads))
        self._lock = threading.Lock()
        self._torch = torch

    def predict(self, texts: Sequence[str], batch_size: int = 64) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        torch = self._torch
        windows: List[List[int]] = []
        owner: List[int] = []
        for i, text in enumerate(texts):
            text_windows = self.tokenizer.windows(text or "", self.window_stride)
            if self.max_windows and len(text_windows) > self.max_windows:
                # keep the first and last windows and an even spread between them
                keep = sorted({0, len(text_windows) - 1, *[round(k * (len(text_windows) - 1) / (self.max_windows - 1)) for k in range(self.max_windows)]})[: self.max_windows]
                text_windows = [text_windows[k] for k in keep]
            for w in text_windows:
                windows.append(w)
                owner.append(i)
        probabilities = np.zeros(len(texts), dtype=np.float64)
        reason_logits = np.zeros((len(texts), len(REASONS)), dtype=np.float64)
        best = np.full(len(texts), -np.inf)
        with self._lock, torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch = windows[start: start + batch_size]
                token_ids = self.tokenizer.pad(batch)
                injection_logit, reason_logit = self.model(token_ids)
                probs = torch.sigmoid(injection_logit.float()).numpy()
                logits = reason_logit.float().numpy()
                for j, p in enumerate(probs):
                    i = owner[start + j]
                    if p > best[i]:
                        best[i] = p
                        probabilities[i] = float(p)
                        reason_logits[i] = logits[j]
        counts = np.bincount(np.asarray(owner, dtype=np.int64), minlength=len(texts)).tolist() if texts else []
        return probabilities, reason_logits, counts, {self.name: probabilities}


class TfidfBackend:
    name = "tfidf_logreg"
    has_reasons = False

    def __init__(self, run_dir: Path):
        import joblib
        import scipy.sparse as sp

        self.run_dir = Path(run_dir)
        self.word = joblib.load(self.run_dir / "tfidf_word.joblib")
        self.char = joblib.load(self.run_dir / "tfidf_char.joblib")
        self.logreg = joblib.load(self.run_dir / "logreg.joblib")
        self.calibrator = joblib.load(self.run_dir / "logreg_calibrator.joblib")
        thresholds = json.loads((self.run_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = _all_thresholds(thresholds["thresholds"]["tfidf_logreg"], thresholds["fpr_budget_thresholds_val"]["tfidf_logreg"],
                                          thresholds.get("fpr_budget_thresholds_val_retrieval", {}).get("tfidf_logreg"))
        self.run_manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8")) if (self.run_dir / "run_manifest.json").exists() else {}
        self._sp = sp
        self._lock = threading.Lock()

    def predict(self, texts: Sequence[str], batch_size: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        texts = [t or "" for t in texts]
        if not texts:
            return np.zeros(0), None, [], {self.name: np.zeros(0)}
        with self._lock:
            matrix = self._sp.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
            raw = self.logreg.predict_proba(matrix)[:, 1]
        clipped = np.clip(raw, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        probabilities = self.calibrator.predict_proba(logits)[:, 1].astype(np.float64)
        return probabilities, None, [1] * len(texts), {self.name: probabilities}


class EnsembleBackend:
    name = "ensemble_mean"
    has_reasons = True

    def __init__(self, cnn: CnnBackend, tfidf: TfidfBackend):
        self.cnn = cnn
        self.tfidf = tfidf
        self.run_dir = cnn.run_dir
        thresholds = json.loads((cnn.run_dir / "thresholds.json").read_text(encoding="utf-8"))
        entry = thresholds.get("candidates", {}).get("ensemble_mean")
        if entry is None:
            raise FileNotFoundError("the Phase 3 run has no ensemble thresholds; rerun phase_3_text_cnn_bigru after phase_2_baselines")
        self.thresholds = _all_thresholds(entry["threshold_f1_val"], entry["fpr_budget_thresholds_val"], entry.get("fpr_budget_thresholds_val_retrieval"))
        self.run_manifest = cnn.run_manifest

    def predict(self, texts: Sequence[str], batch_size: int = 64):
        p_cnn, reasons, windows, _ = self.cnn.predict(texts, batch_size)
        p_tfidf, _, _, _ = self.tfidf.predict(texts)
        return 0.5 * (p_cnn + p_tfidf), reasons, windows, {self.cnn.name: p_cnn, self.tfidf.name: p_tfidf}


class TransformerBackend:
    """A Hugging Face sequence-classification guard loaded from a local directory (never fetched at runtime).

    Built for Horizon-Labs/prompt-injection-guard-base (Apache-2.0, mmBERT, 8k context, normaliser inside the
    tokenizer) but any SAFE/INJECTION classifier with the same head works. Long inputs are scored in overlapping
    windows and the maximum is kept, as the model's own card prescribes. Default 512-token windows with 256
    overlap: measured 2026-10-10, a one-sentence instruction planted in 2,400 tokens of benign text scores
    0.999+ at any window size once windows are built correctly, while the benign-only text scores 0.10 at 2,048
    tokens and 0.04 at 512, so smaller windows cost nothing on recall and lower the benign floor. Thresholds come from a thresholds.json
    written beside the model by scripts/evaluate_hf_guard.py (budgets chosen on our validation rows); without
    one, the card's default of 0.5 is used for every budget.
    """
    name = "hf_guard"
    has_reasons = False

    def __init__(self, model_dir: Path, window: int = 512, stride: int = 256, batch_size: int = 8,
                 torch_threads: Optional[int] = None, max_windows: Optional[int] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.run_dir = self.model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval()
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.injection_index = next((i for i, l in id2label.items() if l.upper().startswith("INJ")), 1)
        self.window, self.stride, self.batch_size = int(window), int(stride), int(batch_size)
        self.max_windows = int(max_windows) if max_windows else None
        threads = torch_threads or os.environ.get("PI_NN_TORCH_THREADS") or min(8, max(1, (os.cpu_count() or 2) // 2))
        torch.set_num_threads(int(threads))
        thresholds_path = self.model_dir / "thresholds.json"
        if thresholds_path.exists():
            t = json.loads(thresholds_path.read_text(encoding="utf-8"))
            self.thresholds = _all_thresholds(t["threshold"], t["fpr_budget_thresholds_val"], t.get("fpr_budget_thresholds_val_retrieval"))
        else:
            self.thresholds = {"f1": 0.5, "fpr_0.01": 0.5, "fpr_0.005": 0.5, "fpr_0.001": 0.5,
                               "retrieval_fpr_0.01": 0.5, "retrieval_fpr_0.005": 0.5, "retrieval_fpr_0.001": 0.5}
        self.run_manifest = json.loads((self.model_dir / "run_manifest.json").read_text(encoding="utf-8")) if (self.model_dir / "run_manifest.json").exists() else {"model": self.model_dir.name}
        self._lock = threading.Lock()
        self._torch = torch

    def _windows(self, text: str):
        """Token windows of `window` length with `stride` overlap, built by hand: transformers 5's
        return_overflowing_tokens yields only two windows regardless of length, which silently skips the
        middle of long inputs. Special tokens are added per window."""
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        inner = self.window - (2 if cls_id is not None and sep_id is not None else 0)
        if len(ids) <= inner:
            starts = [0]
        else:
            step = max(1, inner - self.stride)
            starts = list(range(0, len(ids) - inner + 1, step))
            if starts[-1] + inner < len(ids):
                starts.append(len(ids) - inner)
        if self.max_windows and len(starts) > self.max_windows:
            keep = sorted({0, len(starts) - 1, *[round(k * (len(starts) - 1) / (self.max_windows - 1)) for k in range(self.max_windows)]})[: self.max_windows]
            starts = [starts[k] for k in keep]
        out = []
        for st in starts:
            piece = ids[st: st + inner]
            if cls_id is not None and sep_id is not None:
                piece = [cls_id] + piece + [sep_id]
            out.append(piece)
        return out

    def predict(self, texts: Sequence[str], batch_size: Optional[int] = None):
        torch = self._torch
        texts = [t or "" for t in texts]
        probabilities = np.zeros(len(texts), dtype=np.float64)
        windows = [0] * len(texts)
        if not texts:
            return probabilities, None, windows, {self.name: probabilities}
        pieces, owner = [], []
        for i, text in enumerate(texts):
            for piece in self._windows(text):
                pieces.append(piece)
                owner.append(i)
        bs = batch_size or self.batch_size
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        with self._lock, torch.inference_mode():
            for start in range(0, len(pieces), bs):
                batch = pieces[start: start + bs]
                width = max(len(p) for p in batch)
                input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for r, piece in enumerate(batch):
                    input_ids[r, : len(piece)] = torch.tensor(piece, dtype=torch.long)
                    attention[r, : len(piece)] = 1
                probs = torch.softmax(self.model(input_ids=input_ids, attention_mask=attention).logits, -1)[:, self.injection_index].numpy()
                for r, p in enumerate(probs):
                    i = owner[start + r]
                    windows[i] += 1
                    if p > probabilities[i]:
                        probabilities[i] = float(p)
        return probabilities, None, windows, {self.name: probabilities}


class HttpGuardBackend:
    """The transformer guard served by vLLM (`--runner pooling`, POST /classify) or any server with the same
    contract, reached over HTTP. No model in this process. Thresholds come from the local models/hf/<name>/
    thresholds.json written by scripts/evaluate_hf_guard.py when present, else 0.5 for every budget.

    Chunks are split by characters into overlapping pieces of at most max_chars (6,000 by default, well inside
    the server's 8,192-token limit even for punctuation-heavy text) and the maximum is kept (the card's
    windowed-max rule). The server returns P(SAFE), P(INJECTION) per input.
    """
    name = "hf_guard"
    has_reasons = False

    def __init__(self, url: str, model: str = "prompt-injection-guard", thresholds_dir: Optional[Path] = None,
                 timeout: float = 30.0, max_chars: int = 6000, overlap_chars: int = 1000, batch_size: int = 64):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_chars, self.overlap_chars, self.batch_size = int(max_chars), int(overlap_chars), int(batch_size)
        self.run_dir = Path(thresholds_dir) if thresholds_dir else None
        tpath = (self.run_dir / "thresholds.json") if self.run_dir else None
        if tpath and tpath.exists():
            t = json.loads(tpath.read_text(encoding="utf-8"))
            self.thresholds = _all_thresholds(t["threshold"], t["fpr_budget_thresholds_val"], t.get("fpr_budget_thresholds_val_retrieval"))
        else:
            self.thresholds = {k: 0.5 for k in ("f1", "fpr_0.01", "fpr_0.005", "fpr_0.001", "retrieval_fpr_0.01", "retrieval_fpr_0.005", "retrieval_fpr_0.001")}
        self.run_manifest = {"model": self.model, "url": self.url}
        self._injection_index: Optional[int] = None

    def _pieces(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]
        step = max(1, self.max_chars - self.overlap_chars)
        starts = list(range(0, len(text) - self.max_chars + 1, step))
        if starts[-1] + self.max_chars < len(text):
            starts.append(len(text) - self.max_chars)
        return [text[s: s + self.max_chars] for s in starts]

    def _post(self, inputs: List[str]) -> List[dict]:
        import urllib.request
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        req = urllib.request.Request(self.url + "/classify", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            longest = max(len(t) for t in inputs)
            raise RuntimeError(f"guard service returned HTTP {exc.code} for a batch of {len(inputs)} (longest {longest} chars): {detail}") from exc
        data = sorted(payload["data"], key=lambda d: d["index"])
        if self._injection_index is None:
            # Prefer the label the server reports; the guard's id2label is {0: SAFE, 1: INJECTION}.
            labels = [d.get("label") for d in data]
            self._injection_index = 1
            for d in data:
                if d.get("label") and str(d["label"]).upper().startswith("INJ") and len(d["probs"]) >= 2:
                    self._injection_index = int(np.argmax(d["probs"]))
                    break
        return data

    def predict(self, texts: Sequence[str], batch_size: Optional[int] = None):
        texts = [t or "" for t in texts]
        probabilities = np.zeros(len(texts), dtype=np.float64)
        windows = [0] * len(texts)
        if not texts:
            return probabilities, None, windows, {self.name: probabilities}
        pieces, owner = [], []
        for i, t in enumerate(texts):
            for piece in self._pieces(t):
                pieces.append(piece)
                owner.append(i)
        bs = batch_size or self.batch_size
        for start in range(0, len(pieces), bs):
            data = self._post(pieces[start: start + bs])
            for r, d in enumerate(data):
                i = owner[start + r]
                p = float(d["probs"][self._injection_index]) if len(d["probs"]) > 1 else float(d["probs"][0])
                windows[i] += 1
                if p > probabilities[i]:
                    probabilities[i] = p
        return probabilities, None, windows, {self.name: probabilities}
