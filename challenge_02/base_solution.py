import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    One of the most popular part in NLP.
    If you understand self attention, you will understand the heard of transformers.
    """
    def __init__(self):
        super().__init__()
        self.W = None

    def forward(self, x):
        q, k, v = self.W
        # norm_attention = F.softmax(q*k)
        # (norm_attention * v).sum(-1)
        return

if __name__ == "__main__":
    x = torch.randint(0, 500, (10, 50))  # tokenizer(text)
    model = SelfAttention()
    print(model.forward(x))
