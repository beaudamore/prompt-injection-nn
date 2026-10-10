"""Parity: the package must reproduce the notebooks' locked-test numbers from the saved artifacts.

Skipped unless the v4 data and runs are present (they are not in git). On the DGX Spark:
    pytest tests/test_parity.py -s
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("PI_NN_MODELS_DIR", ROOT / "models"))
DATA = ROOT / "data"


def _newest(prefix):
    runs = sorted((p for p in MODELS.glob(f"{prefix}*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _test_frame(run):
    manifest = json.loads((run / "run_manifest.json").read_text())
    version = manifest["dataset_version"]
    path = DATA / f"source-clean-{version}" / "test.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    import pandas as pd
    return pd.read_parquet(path, columns=["text", "label", "shape"])


@pytest.mark.parametrize("backend_name,prefix,metrics_key", [("cnn_bigru", "phase3-cnn-bigru-", None), ("tfidf_logreg", "phase2-", "tfidf_logreg")])
def test_package_reproduces_locked_test_metrics(backend_name, prefix, metrics_key):
    run = _newest(prefix)
    if run is None:
        pytest.skip(f"no {prefix}* run under {MODELS}")
    from prompt_injection_nn import load_detector
    frame = _test_frame(run)
    metrics = json.loads((run / "metrics.json").read_text())
    expected = metrics["test"][metrics_key] if metrics_key else metrics["test"]
    detector = load_detector(models_dir=MODELS, backend=backend_name, budget="f1")
    # Parity is checked with truncation, exactly as the notebook scored: one window per text.
    if backend_name == "cnn_bigru":
        detector.backend.window_stride = detector.backend.tokenizer.max_tokens * 10  # single window
        # windows() still emits one window for texts within max_tokens and a tail window otherwise;
        # score truncated texts explicitly to mirror the notebook's encode().
        tok = detector.backend.tokenizer
        texts = [" ".join(tok.tokenize(t)[: tok.max_tokens]) for t in frame["text"].fillna("").astype(str)]
    else:
        texts = frame["text"].fillna("").astype(str).tolist()
    verdicts = detector.classify_batch(texts)
    probs = np.array([v.probability for v in verdicts])
    y = frame["label"].to_numpy()
    pred = probs >= detector.threshold
    recall = pred[y == 1].mean()
    fpr = pred[y == 0].mean()
    print(f"\n{backend_name}: recall {recall:.4f} (notebook {expected['recall']:.4f}), FPR {fpr:.4f} (notebook {expected['false_positive_rate']:.4f})")
    tolerance = 0.01 if backend_name == "cnn_bigru" else 0.002   # re-joining tokens with spaces is not byte-identical input for the CNN
    assert abs(recall - expected["recall"]) < tolerance
    assert abs(fpr - expected["false_positive_rate"]) < tolerance
