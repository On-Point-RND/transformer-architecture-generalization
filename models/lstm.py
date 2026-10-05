"""Stacked causal LSTM language model with the standard model config.

The LSTM has no normalization layers, so ``model.norm`` is ignored here.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import ModelConfig
from optimizers import get_optimizer


class LSTMModel(nn.Module):
    """Token embedding -> stacked LSTM -> vocabulary logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be filled from the task")
        if config.n_layer < 1 or config.n_embd < 1:
            raise ValueError("n_layer and n_embd must be positive")

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout if config.n_layer > 1 else 0.0,
            bias=config.bias,
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.xavier_uniform_(gate)
            elif "weight_hh" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(parameter)
                if "bias_ih" in name:
                    hidden = self.config.n_embd
                    with torch.no_grad():
                        parameter[hidden:2 * hidden].fill_(1.0)

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {length}, block size is only "
                f"{self.config.block_size}"
            )
        x = self.dropout(self.embedding(idx))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def param_report(self):
        return {
            "positional_encoding": "recurrent",
            "positional_parameters": {"trainable": 0, "frozen": 0, "total": 0},
        }

    def get_num_params(self, non_embedding=True):
        return sum(parameter.numel() for parameter in self.parameters())

    def crop_block_size(self, block_size):
        if block_size > self.config.block_size:
            raise ValueError("crop_block_size cannot increase block_size")
        self.config.block_size = block_size

    def configure_optimizers(self, optimizer, device_type):
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if optimizer.decay == "all":
            groups = [{"params": params, "weight_decay": optimizer.weight_decay}]
        elif optimizer.decay == "matrices":
            groups = [
                {
                    "params": [parameter for parameter in params if parameter.dim() >= 2],
                    "weight_decay": optimizer.weight_decay,
                },
                {
                    "params": [parameter for parameter in params if parameter.dim() < 2],
                    "weight_decay": 0.0,
                },
            ]
        else:
            raise ValueError(
                "optimizer.decay must be 'matrices' or 'all', "
                f"got {optimizer.decay!r}"
            )
        return get_optimizer(optimizer.name)(groups, optimizer, device_type)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        config = self.config
        recurrent = 48 * config.n_layer * config.n_embd**2
        output = 6 * config.n_embd * config.vocab_size
        flops = (recurrent + output) * config.block_size * fwdbwd_per_iter
        return flops / dt / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def build_model(config: ModelConfig):
    return LSTMModel(config)
