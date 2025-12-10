"""
Minimal probe definitions (linear or 1-layer MLP) plus a handy hidden-size lookup.
"""

import torch.nn as nn

HS_DICT = {
    "DeepSeek-R1-Distill-Qwen-32B": 5120,
    "DeepSeek-R1-Distill-Qwen-1.5B": 1536,
    "DeepSeek-R1-Distill-Qwen-7B": 3584,
    "DeepSeek-R1-Distill-Llama-8B": 4096,
    "DeepSeek-R1-Distill-Llama-70B": 8192,
    "QwQ-32B": 5120,
}


class MLPProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)  # logits
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


class LinearProbe(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.output = nn.Linear(input_size, output_size)  # logits

    def forward(self, x):
        return self.output(x)


def load_probe(input_size: int, hidden_size: int = 0, output_size: int = 1, ckpt_weights=None):
    if hidden_size and hidden_size > 0:
        model = MLPProbe(input_size, hidden_size, output_size)
    else:
        model = LinearProbe(input_size, output_size)
    if ckpt_weights is not None:
        model.load_state_dict(ckpt_weights)
    return model
