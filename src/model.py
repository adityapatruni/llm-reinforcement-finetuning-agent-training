import math
import torch
from torch import nn
from transformers import AutoModel


class FeedbackPolicy(nn.Module):
    def __init__(self, model_name: str, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.policy = nn.Linear(hidden, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        logits = self.policy(pooled)
        return logits


class ScratchEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        max_len: int = 512,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.policy = nn.Linear(d_model, num_actions)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        logits = self.policy(pooled)
        return logits


def build_model(
    model_name: str,
    num_actions: int,
    dropout: float,
    from_scratch: bool,
    vocab_size: int,
    max_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
):
    if from_scratch:
        return ScratchEncoder(
            vocab_size=vocab_size,
            num_actions=num_actions,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
    return FeedbackPolicy(model_name=model_name, num_actions=num_actions, dropout=dropout)
