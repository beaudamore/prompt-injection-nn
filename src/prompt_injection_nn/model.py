"""The Phase 3 architecture, byte-for-byte the notebook's module so saved weights load unchanged."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCnnBiGru(nn.Module):
    """1D-CNN over token embeddings, bidirectional GRU, masked max-pooling, two heads:
    a binary injection logit and a nine-way reason logit vector."""

    def __init__(self, vocab_size: int, pad_index: int, num_reasons: int, config: dict):
        super().__init__()
        self.pad_index = pad_index
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=pad_index)
        self.cnn = nn.Conv1d(config["embedding_dim"], config["cnn_channels"], kernel_size=3, padding=1)
        self.gru = nn.GRU(config["cnn_channels"], config["hidden_dim"], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config["dropout"])
        self.shared = nn.Sequential(nn.Linear(config["hidden_dim"] * 2, config["hidden_dim"]), nn.ReLU(), nn.Dropout(config["dropout"]))
        self.injection_head = nn.Linear(config["hidden_dim"], 1)
        self.reason_head = nn.Linear(config["hidden_dim"], num_reasons)

    def forward(self, token_ids: torch.Tensor):
        # Padding-invariant: the convolution's bleed into pad positions is zeroed and the GRU runs on
        # packed sequences, so a text scores identically alone or in any batch. (An earlier version let
        # the backward GRU read the padding, which made scores depend on batch composition.)
        mask = token_ids.ne(self.pad_index)
        lengths = mask.sum(dim=1).clamp(min=1)
        embedded = self.embedding(token_ids).transpose(1, 2)
        features = torch.relu(self.cnn(embedded)).transpose(1, 2)
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=token_ids.size(1))
        masked = sequence.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = self.dropout(masked.max(dim=1).values)
        hidden = self.shared(pooled)
        return self.injection_head(hidden).squeeze(1), self.reason_head(hidden)
