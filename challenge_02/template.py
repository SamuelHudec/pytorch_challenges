import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    one of the most popular parts if you will understand self attention than you understand the heard of transformers
    """
    def __init__(self):
        super().__init__()
        self.W = None

    def forward(self, x):
        q, k, v = self.W
        # norm_attention = F.softmax(q*k)
        # (norm_attention * v).sum(-1)
        return


x = torch.randint(0, 500, (10, 50))  # tokenizer(text)
model = SelfAttention()
model.forward(x)