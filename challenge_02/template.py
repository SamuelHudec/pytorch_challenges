import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    One of the most popular part in NLP.
    If you understand self attention, you will understand the heard of transformers.
    hint: https://www.youtube.com/watch?v=yGTUuEx3GkA
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