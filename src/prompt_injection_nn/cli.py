"""Command line: prompt-injection-nn [--backend NAME] [--budget B] [--models DIR] TEXT ... or '-' for stdin lines."""

import argparse
import json
import sys

from .detector import BUDGETS, DEFAULT_BUDGET, load_detector


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classify text as SAFE or INJECTION: <reason>.")
    parser.add_argument("texts", nargs="*", help="texts to classify; '-' reads one text per line from stdin")
    parser.add_argument("--backend", choices=["cnn_bigru", "tfidf_logreg", "ensemble_mean", "hf_guard", "hf_guard_http"], default=None)
    parser.add_argument("--guard-url", default=None, help="vLLM /classify base URL for the hf_guard_http backend, e.g. http://host.docker.internal:8009")
    parser.add_argument("--budget", choices=BUDGETS, default=DEFAULT_BUDGET)
    parser.add_argument("--models", default=None, help="models directory (default: PI_NN_MODELS_DIR or the repo's models/)")
    parser.add_argument("--json", action="store_true", help="print one JSON object per text instead of the contract line")
    args = parser.parse_args(argv)
    texts = [line.rstrip("\n") for line in sys.stdin] if args.texts == ["-"] else args.texts
    if not texts:
        parser.error("no text given")
    detector = load_detector(models_dir=args.models, backend=args.backend, budget=args.budget, guard_url=args.guard_url)
    for verdict in detector.classify_batch(texts):
        if args.json:
            print(json.dumps({"contract": verdict.contract, "label": verdict.label, "probability": round(verdict.probability, 6),
                              "reason": verdict.reason, "threshold": verdict.threshold, "budget": verdict.budget,
                              "backend": verdict.backend, "windows": verdict.windows}))
        else:
            print(verdict.contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
