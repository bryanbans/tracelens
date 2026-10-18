"""Small causal sequence models. Outputs at t predict the event at t+1."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(F.gelu(self.conv(F.pad(x, (self.left, 0)))))


class SequenceModel(nn.Module):
    def __init__(self, vocab_size: int, architecture: str = "lstm", embedding: int = 32,
                 hidden: int = 64, dropout: float = .15):
        super().__init__()
        if architecture not in {"lstm", "tcn"}:
            raise ValueError("Architecture must be lstm or tcn.")
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, embedding)
        self.dropout = nn.Dropout(dropout)
        if architecture == "lstm":
            self.encoder = nn.LSTM(embedding, hidden, batch_first=True)
        else:
            self.project = nn.Conv1d(embedding, hidden, 1)
            self.encoder = nn.Sequential(*[CausalBlock(hidden, d, dropout) for d in (1, 2, 4)])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        x = self.dropout(self.embedding(tokens))
        if self.architecture == "lstm":
            x, _ = self.encoder(x)
        else:
            x = self.encoder(self.project(x.transpose(1, 2))).transpose(1, 2)
        return self.head(self.dropout(x))


def event_losses(model, sequences):
    logits = model(sequences[:, :-1])
    return F.cross_entropy(logits.transpose(1, 2), sequences[:, 1:], reduction="none")
