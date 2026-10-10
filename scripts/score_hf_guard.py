#!/usr/bin/env python3
"""Score a Hugging Face sequence-classification guard over the v4 splits on CPU and save per-row
probabilities beside the model, aligned to the split files by source_id, so it can be judged by the same
gate and budget thresholds as the in-house models. Windowed: max over overlapping windows.

    python3 scripts/score_hf_guard.py --model models/hf/horizon-labs-prompt-injection-guard-base --version v4
    python3 scripts/score_hf_guard.py --model models/hf/horizon-labs-prompt-injection-guard-base --url http://localhost:8009   # via the vLLM guard stack (GPU, minutes)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--version", default="v4")
    ap.add_argument("--window", type=int, default=512); ap.add_argument("--stride", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--splits", default="val,test,adversarial_test"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--url", default=None, help="score through a vLLM /classify server instead of loading the model here")
    ap.add_argument("--served-name", default="prompt-injection-guard")
    a = ap.parse_args()
    mdir = Path(a.model)
    sys.path.insert(0, str(ROOT / "src"))
    if a.url:
        from prompt_injection_nn.backends import HttpGuardBackend
        backend = HttpGuardBackend(a.url, model=a.served_name, batch_size=a.batch)
        print("scoring via", a.url, flush=True)
    else:
        import torch
        from prompt_injection_nn.backends import TransformerBackend
        backend = TransformerBackend(mdir, window=a.window, stride=a.stride, batch_size=a.batch, torch_threads=a.threads)
        print("scoring locally on CPU, window", a.window, "stride", a.stride, flush=True)

    def score(texts):
        probs, _, _, _ = backend.predict(texts)
        return probs

    data = ROOT / "data" / f"source-clean-{a.version}"
    for split in a.splits.split(","):
        df = pd.read_parquet(data / f"{split}.parquet", columns=["source_id", "label", "shape", "source", "text"])
        texts = df["text"].fillna("").astype(str).tolist(); probs = np.zeros(len(texts)); t0 = time.time()
        for i in range(0, len(texts), a.batch):
            probs[i:i + a.batch] = score(texts[i:i + a.batch])
            if (i // a.batch) % 50 == 0: print(f"{split}: {i}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
        out = df[["source_id", "label", "shape", "source"]].copy(); out["hf_guard"] = probs
        out.to_parquet(mdir / f"probabilities_{split}.parquet", index=False)
        print(f"{split}: done {len(texts)} rows in {time.time()-t0:.0f}s -> {mdir / f'probabilities_{split}.parquet'}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
