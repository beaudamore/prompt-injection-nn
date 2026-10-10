"""Model backends. Each loads one run directory written by a notebook and scores texts on CPU.

- CnnBackend:      models/phase3-cnn-bigru-<version>-<fingerprint>/   (model_state.pt, vocab.json, tokenizer.json, config.json, thresholds.json)
- TfidfBackend:    models/phase2-<version>-<fingerprint>/             (tfidf_word.joblib, tfidf_char.joblib, logreg.joblib, logreg_calibrator.joblib, thresholds.json)
- EnsembleBackend: mean of the two calibrated probabilities; reasons from the CNN. Thresholds from the
                   Phase 3 thresholds.json "candidates"["ensemble_mean"] entry.

Every backend exposes `name`, `thresholds` ({"f1": t, "fpr_0.01": t, ...}), `has_reasons`, and
`predict(texts) -> (probabilities, reason_logits or None, windows_per_text)`.
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
        return probabilities, reason_logits, counts


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
            return np.zeros(0), None, []
        with self._lock:
            matrix = self._sp.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
            raw = self.logreg.predict_proba(matrix)[:, 1]
        clipped = np.clip(raw, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1].astype(np.float64), None, [1] * len(texts)


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
        p_cnn, reasons, windows = self.cnn.predict(texts, batch_size)
        p_tfidf, _, _ = self.tfidf.predict(texts)
        return 0.5 * (p_cnn + p_tfidf), reasons, windows
