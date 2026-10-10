"""Detector: a backend plus an operating point, with the public classify API."""

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .backends import CnnBackend, EnsembleBackend, TfidfBackend
from .contract import INJECTION, REASONS, SAFE, Verdict

BUDGETS = ("f1", "fpr_0.01", "fpr_0.005", "fpr_0.001", "retrieval_fpr_0.01", "retrieval_fpr_0.005", "retrieval_fpr_0.001")
DEFAULT_BUDGET = "retrieval_fpr_0.01"


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class Detector:
    def __init__(self, backend, budget: str = DEFAULT_BUDGET):
        if budget not in backend.thresholds and budget.startswith("retrieval_"):
            budget = budget[len("retrieval_"):]   # older runs without retrieval thresholds fall back to the all-rows budget
        if budget not in backend.thresholds:
            raise ValueError(f"budget {budget!r} not in {sorted(backend.thresholds)}")
        self.backend = backend
        self.budget = budget
        self.threshold = float(backend.thresholds[budget])

    def set_budget(self, budget: str) -> None:
        if budget not in self.backend.thresholds:
            raise ValueError(f"budget {budget!r} not in {sorted(self.backend.thresholds)}")
        self.budget = budget
        self.threshold = float(self.backend.thresholds[budget])

    def classify_batch(self, texts: Sequence[str]) -> List[Verdict]:
        """Order-preserving. Empty or whitespace-only text is SAFE with probability 0: nothing to inject."""
        texts = list(texts)
        verdicts: List[Optional[Verdict]] = [None] * len(texts)
        live = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        for i, t in enumerate(texts):
            if i not in live:
                verdicts[i] = Verdict(SAFE, 0.0, self.threshold, self.budget, None, {}, self.backend.name, 0)
        if live:
            probabilities, reason_logits, windows = self.backend.predict([texts[i] for i in live])
            for k, i in enumerate(live):
                p = float(probabilities[k])
                flagged = p >= self.threshold
                reason, scores = None, {}
                if reason_logits is not None:
                    dist = _softmax(np.asarray(reason_logits[k], dtype=np.float64))
                    scores = {r: float(s) for r, s in zip(REASONS, dist)}
                    reason = REASONS[int(dist.argmax())]
                if flagged and reason is None:
                    reason = "Embedded Instruction"   # a backend without a reason head reports the indirect-injection reason
                verdicts[i] = Verdict(INJECTION if flagged else SAFE, p, self.threshold, self.budget,
                                      reason if flagged else None, scores, self.backend.name, int(windows[k]))
        return verdicts  # type: ignore[return-value]

    def classify(self, text: str) -> Verdict:
        return self.classify_batch([text])[0]


def _newest(models_dir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted((p for p in models_dir.glob(f"{prefix}*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def resolve_models_dir(models_dir: Optional[os.PathLike] = None) -> Path:
    if models_dir:
        return Path(models_dir)
    env = os.environ.get("PI_NN_MODELS_DIR")
    if env:
        return Path(env)
    for candidate in ("/workspace/training/prompt-injection-nn/models", "/home/spark/projects/training/prompt-injection-nn/models"):
        if Path(candidate).is_dir():
            return Path(candidate)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "models").is_dir() and (parent / "docs" / "PLAN.md").exists():
            return parent / "models"
    raise FileNotFoundError("models directory not found; pass models_dir or set PI_NN_MODELS_DIR")


def load_detector(models_dir: Optional[os.PathLike] = None, backend: Optional[str] = None, budget: str = DEFAULT_BUDGET,
                  phase3_run: Optional[os.PathLike] = None, phase2_run: Optional[os.PathLike] = None,
                  window_stride: Optional[int] = None, torch_threads: Optional[int] = None, max_windows: Optional[int] = None) -> Detector:
    """Load the recommended backend (from the newest Phase 3 run's gate.json) or the one named.

    backend: None | "cnn_bigru" | "tfidf_logreg" | "ensemble_mean"
    budget:  "retrieval_fpr_0.01" (default) | "retrieval_fpr_0.005" | "retrieval_fpr_0.001": thresholds chosen on validation
             document + agentic rows, the retrieval filter's input; "fpr_0.01" | "fpr_0.005" | "fpr_0.001": chosen on all
             validation rows (chat included); "f1": the notebooks' F1-max point, for development only
    """
    models = resolve_models_dir(models_dir)
    p3 = Path(phase3_run) if phase3_run else _newest(models, "phase3-cnn-bigru-")
    p2 = Path(phase2_run) if phase2_run else _newest(models, "phase2-")
    if backend is None:
        gate_path = (p3 / "gate.json") if p3 else None
        backend = json.loads(gate_path.read_text(encoding="utf-8")).get("recommended_backend", "cnn_bigru") if gate_path and gate_path.exists() else "cnn_bigru"
    if backend == "cnn_bigru":
        if p3 is None:
            raise FileNotFoundError(f"no phase3-cnn-bigru-* run under {models}")
        chosen = CnnBackend(p3, window_stride=window_stride, torch_threads=torch_threads, max_windows=max_windows)
    elif backend == "tfidf_logreg":
        if p2 is None:
            raise FileNotFoundError(f"no phase2-* run under {models}")
        chosen = TfidfBackend(p2)
    elif backend == "ensemble_mean":
        if p3 is None or p2 is None:
            raise FileNotFoundError(f"ensemble needs a phase2-* and a phase3-cnn-bigru-* run under {models}")
        chosen = EnsembleBackend(CnnBackend(p3, window_stride=window_stride, torch_threads=torch_threads, max_windows=max_windows), TfidfBackend(p2))
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if p3 is not None and p2 is not None and backend == "ensemble_mean":
        f3 = json.loads((p3 / "run_manifest.json").read_text()).get("phase1_config_fingerprint")
        f2 = json.loads((p2 / "run_manifest.json").read_text()).get("phase1_config_fingerprint")
        if f3 and f2 and f3 != f2:
            raise ValueError(f"ensemble runs were trained on different data builds: {f3[:16]} vs {f2[:16]}")
    return Detector(chosen, budget=budget)
