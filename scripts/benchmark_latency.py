"""Latency of the three backends on real test chunks, CPU. Run on the Spark after the notebooks:
    docker exec unsloth-notebook python /workspace/training/prompt-injection-nn/scripts/benchmark_latency.py
Prints p50 and p95 for ten real document chunks (as stored, and cut to 1,000 characters like an
Open WebUI chunk), with the package's default thread count and window cap."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from prompt_injection_nn import load_detector  # noqa: E402


def bench(detector, rows, label, calls=200, warmup=20):
    for _ in range(warmup):
        detector.classify_batch(rows)
    timings = []
    for _ in range(calls):
        started = time.perf_counter()
        detector.classify_batch(rows)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    print(f"{label:58s} p50 {timings[len(timings) // 2]:6.1f} ms   p95 {timings[int(len(timings) * 0.95)]:6.1f} ms")


def main():
    data = ROOT / "data" / "source-clean-v4" / "test.parquet"
    frame = pd.read_parquet(data, columns=["text", "shape"]).query("shape == 'document'")
    as_stored = frame.head(10)["text"].tolist()
    typical = [t[:1000] for t in frame.sample(10, random_state=1)["text"].tolist()]
    for backend in ("tfidf_logreg", "cnn_bigru", "ensemble_mean"):
        detector = load_detector(backend=backend)
        tok = getattr(detector.backend, "tokenizer", None) or getattr(getattr(detector.backend, "cnn", None), "tokenizer", None)
        if tok is not None and backend == "cnn_bigru":
            print("tokens per stored chunk:", [len(tok.tokenize(t)) for t in as_stored], "| per 1,000-char chunk:", [len(tok.tokenize(t)) for t in typical])
        print(f"{backend}: budget {detector.budget}, threshold {detector.threshold:.4f}, torch threads {torch.get_num_threads()}, cpus {os.cpu_count()}")
        bench(detector, typical, f"  ten 1,000-char document chunks (Open WebUI default size)")
        bench(detector, as_stored, f"  ten stored document chunks (up to several windows each)")


if __name__ == "__main__":
    main()
