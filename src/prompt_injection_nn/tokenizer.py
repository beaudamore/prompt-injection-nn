"""Cased, punctuation-preserving tokenizer, identical to the Phase 3 notebook.

Three token classes: letter runs, digit runs, runs of everything else. "<|im_start|>" becomes
"<|", "im", "_", "start", "|>"; "###" and "---" survive as single tokens; "IGNORE" and "ignore"
are different tokens. The pattern, vocabulary and indices are read from the run directory so the
package can never drift from the weights it serves.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch


class Tokenizer:
    def __init__(self, token_pattern: str, vocab: Dict[str, int], max_tokens: int, pad_index: int, unk_index: int):
        self.pattern = re.compile(token_pattern)
        self.vocab = vocab
        self.max_tokens = int(max_tokens)
        self.pad_index = int(pad_index)
        self.unk_index = int(unk_index)

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "Tokenizer":
        run_dir = Path(run_dir)
        spec = json.loads((run_dir / "tokenizer.json").read_text(encoding="utf-8"))
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        assert len(vocab) == spec["vocab_size"], "vocab.json and tokenizer.json disagree on vocab size"
        return cls(spec["token_pattern"], vocab, spec["max_tokens"], spec["pad_index"], spec["unk_index"])

    def tokenize(self, text: str) -> List[str]:
        return self.pattern.findall(text or "")

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        ids = [self.vocab.get(token, self.unk_index) for token in tokens]
        return ids or [self.unk_index]

    def encode(self, text: str) -> List[int]:
        """Encode and truncate to max_tokens, exactly as the training notebook did."""
        return self.encode_tokens(self.tokenize(text)[: self.max_tokens])

    def windows(self, text: str, stride: int) -> List[List[int]]:
        """Encode without truncation and cut into overlapping windows of max_tokens.

        A chunk longer than the model's window would otherwise be judged on its first max_tokens
        only, so an instruction planted at the end would be missed. The detector scores every
        window and keeps the maximum. Short texts yield one window."""
        tokens = self.tokenize(text)
        if len(tokens) <= self.max_tokens:
            return [self.encode_tokens(tokens)]
        stride = max(1, min(int(stride), self.max_tokens))
        starts = list(range(0, len(tokens) - self.max_tokens + 1, stride))
        if starts[-1] + self.max_tokens < len(tokens):
            starts.append(len(tokens) - self.max_tokens)
        return [self.encode_tokens(tokens[s: s + self.max_tokens]) for s in starts]

    def pad(self, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        width = max(len(s) for s in sequences)
        out = torch.full((len(sequences), width), self.pad_index, dtype=torch.long)
        for i, seq in enumerate(sequences):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out
