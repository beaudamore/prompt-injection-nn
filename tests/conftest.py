import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def stub_cnn_run(tmp_path):
    """A tiny, random-weight Phase 3 run directory with the real file layout."""
    import torch
    from prompt_injection_nn.contract import REASONS
    from prompt_injection_nn.model import TextCnnBiGru

    torch.manual_seed(0)
    run = tmp_path / "phase3-cnn-bigru-vtest-0000000000000000"
    run.mkdir()
    tokens = ["Ignore", "the", "above", "and", "reveal", "system", "prompt", "hello", "world", "<|", "im", "_", "start", "|>", "###", "---", "1", "2"]
    vocab = {"<pad>": 0, "<unk>": 1, **{t: i + 2 for i, t in enumerate(tokens)}}
    config = {"embedding_dim": 8, "hidden_dim": 8, "cnn_channels": 8, "dropout": 0.0}
    model = TextCnnBiGru(len(vocab), 0, len(REASONS), config)
    torch.save(model.state_dict(), run / "model_state.pt")
    (run / "vocab.json").write_text(json.dumps(vocab))
    (run / "tokenizer.json").write_text(json.dumps({"token_pattern": r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+", "cased": True, "max_tokens": 6,
                                                    "pad_token": "<pad>", "unk_token": "<unk>", "pad_index": 0, "unk_index": 1, "vocab_size": len(vocab)}))
    (run / "config.json").write_text(json.dumps({"reasons": list(REASONS), "text": config, "seed": 0}))
    budgets = {f"fpr_{b}": {"threshold": t, "recall": 0.5, "achieved_fpr": b} for b, t in (("0.01", 0.6), ("0.005", 0.7), ("0.001", 0.8))}
    retrieval = {f"fpr_{b}": {"threshold": t, "recall": 0.9, "achieved_fpr": b} for b, t in (("0.01", 0.55), ("0.005", 0.65), ("0.001", 0.75))}
    (run / "thresholds.json").write_text(json.dumps({"threshold": 0.5, "fpr_budget_thresholds_val": budgets, "fpr_budget_thresholds_val_retrieval": retrieval,
                                                     "candidates": {"ensemble_mean": {"threshold_f1_val": 0.5, "fpr_budget_thresholds_val": budgets, "fpr_budget_thresholds_val_retrieval": retrieval}}}))
    (run / "run_manifest.json").write_text(json.dumps({"phase1_config_fingerprint": "0" * 64, "run_id": run.name}))
    (run / "gate.json").write_text(json.dumps({"recommended_backend": "cnn_bigru"}))
    return run
