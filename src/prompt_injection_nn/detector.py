"""Detector: a backend plus an operating point, with the public classify API."""

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .backends import CnnBackend, EnsembleBackend, HttpGuardBackend, TfidfBackend, TransformerBackend
from .contract import INJECTION, REASONS, SAFE, Verdict

BUDGETS = ("f1", "fpr_0.01", "fpr_0.005", "fpr_0.001", "retrieval_fpr_0.01", "retrieval_fpr_0.005", "retrieval_fpr_0.001")
DEFAULT_BUDGET = "retrieval_fpr_0.01"


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class Detector:
    def __init__(self, backend, budget: str = DEFAULT_BUDGET, reason_backend=None):
        if budget not in backend.thresholds and budget.startswith("retrieval_"):
            budget = budget[len("retrieval_"):]   # older runs without retrieval thresholds fall back to the all-rows budget
        if budget not in backend.thresholds:
            raise ValueError(f"budget {budget!r} not in {sorted(backend.thresholds)}")
        self.backend = backend
        self.budget = budget
        self.threshold = float(backend.thresholds[budget])
        # A backend without a reason head (the transformer guard, TF-IDF) borrows this one's reason head for
        # flagged rows only, so the "INJECTION: <reason>" contract still carries a real reason.
        self.reason_backend = reason_backend if (reason_backend is not None and not backend.has_reasons) else None

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
            probabilities, reason_logits, windows, components = self.backend.predict([texts[i] for i in live])
            if reason_logits is None and self.reason_backend is not None:
                flagged_positions = [k for k in range(len(live)) if float(probabilities[k]) >= self.threshold]
                if flagged_positions:
                    _, borrowed, _, _ = self.reason_backend.predict([texts[live[k]] for k in flagged_positions])
                    reason_logits = np.full((len(live), len(REASONS)), np.nan)
                    for row, k in enumerate(flagged_positions):
                        reason_logits[k] = borrowed[row]
            for k, i in enumerate(live):
                p = float(probabilities[k])
                flagged = p >= self.threshold
                reason, scores = None, {}
                if reason_logits is not None and not np.isnan(np.asarray(reason_logits[k], dtype=np.float64)).any():
                    dist = _softmax(np.asarray(reason_logits[k], dtype=np.float64))
                    scores = {r: float(s) for r, s in zip(REASONS, dist)}
                    reason = REASONS[int(dist.argmax())]
                if flagged and reason is None:
                    reason = "Embedded Instruction"   # a backend without a reason head reports the indirect-injection reason
                verdicts[i] = Verdict(INJECTION if flagged else SAFE, p, self.threshold, self.budget,
                                      reason if flagged else None, scores, self.backend.name, int(windows[k]),
                                      {name: float(values[k]) for name, values in components.items()})
        return verdicts  # type: ignore[return-value]

    def classify(self, text: str) -> Verdict:
        return self.classify_batch([text])[0]


def _newest(models_dir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted((p for p in models_dir.glob(f"{prefix}*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _deploy_choice(models: Path) -> dict:
    """models/deploy.json names the backend to ship (written by scripts/evaluate_hf_guard.py when a guard clears
    the gate, or by hand). It takes precedence over the Phase 3 gate.json recommendation."""
    path = models / "deploy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
                  window_stride: Optional[int] = None, torch_threads: Optional[int] = None, max_windows: Optional[int] = None,
                  guard_url: Optional[str] = None, guard_model: str = "prompt-injection-guard") -> Detector:
    """Load the recommended backend (from the newest Phase 3 run's gate.json) or the one named.

    backend: None | "cnn_bigru" | "tfidf_logreg" | "ensemble_mean" | "hf_guard" | "hf_guard_http"
             "hf_guard_http" (or "hf_guard" with guard_url set, or PI_NN_GUARD_URL in the environment) posts to a
             vLLM /classify endpoint serving the guard on the GPU and loads no weights here; thresholds are read from
             models/hf/<hf_model>/thresholds.json when present.
             None: models/deploy.json's choice if present, else the newest Phase 3 run's gate.json recommendation, else
             (no run folders under models/ at all) "hf_guard_http" when a guard_url is known.
             The guard backends need no Phase 2/3 run folders; without a Phase 3 run, a flagged chunk's reason is
             "Embedded Instruction" (the indirect-injection reason) instead of a CNN reason-head reading.
             "hf_guard" loads models/hf/<hf_model> (deploy.json names it; default horizon-labs-prompt-injection-guard-base)
             and borrows the CNN's reason head when a Phase 3 run exists.
    budget:  "retrieval_fpr_0.01" (default) | "retrieval_fpr_0.005" | "retrieval_fpr_0.001": thresholds chosen on validation
             document + agentic rows, the retrieval filter's input; "fpr_0.01" | "fpr_0.005" | "fpr_0.001": chosen on all
             validation rows (chat included); "f1": the notebooks' F1-max point, for development only
    """
    models = resolve_models_dir(models_dir)
    p3 = Path(phase3_run) if phase3_run else _newest(models, "phase3-cnn-bigru-")
    p2 = Path(phase2_run) if phase2_run else _newest(models, "phase2-")
    deploy = _deploy_choice(models)
    guard_url = guard_url or deploy.get("guard_url") or os.environ.get("PI_NN_GUARD_URL")
    if backend is None:
        if deploy.get("backend"):
            backend = deploy["backend"]
        else:
            gate_path = (p3 / "gate.json") if p3 else None
            if gate_path and gate_path.exists():
                backend = json.loads(gate_path.read_text(encoding="utf-8")).get("recommended_backend", "cnn_bigru")
            else:
                # No in-house run folders at all: the only thing that can score is the guard service, when configured.
                backend = "hf_guard_http" if guard_url else "cnn_bigru"
    if backend == "hf_guard_http" or (backend == "hf_guard" and guard_url):
        if not guard_url:
            raise ValueError("backend hf_guard_http needs guard_url (or PI_NN_GUARD_URL / deploy.json guard_url)")
        hf_dir = models / "hf" / deploy.get("hf_model", "horizon-labs-prompt-injection-guard-base")
        reasons = CnnBackend(p3, window_stride=window_stride, torch_threads=torch_threads, max_windows=max_windows) if p3 is not None else None
        return Detector(HttpGuardBackend(guard_url, model=deploy.get("guard_model", guard_model), thresholds_dir=hf_dir if hf_dir.is_dir() else None), budget=budget, reason_backend=reasons)
    if backend == "hf_guard":
        hf_dir = models / "hf" / deploy.get("hf_model", "horizon-labs-prompt-injection-guard-base")
        if not (hf_dir / "config.json").exists():
            raise FileNotFoundError(f"no transformer guard at {hf_dir}")
        reasons = CnnBackend(p3, window_stride=window_stride, torch_threads=torch_threads, max_windows=max_windows) if p3 is not None else None
        return Detector(TransformerBackend(hf_dir, torch_threads=torch_threads, max_windows=max_windows), budget=budget, reason_backend=reasons)
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
