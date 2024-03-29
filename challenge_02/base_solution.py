from typing import Tuple

import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    One of the most popular part in NLP.
    If you understand self attention, you will understand the heard of transformers.
    """
    def __init__(self, embedding_length: int) -> None:
        super().__init__()
        self.e_length = embedding_length
        self.TW = self._trainable_weights()

    def _trainable_weights(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        q = nn.Linear(self.e_length, self.e_length, bias=False)
        k = nn.Linear(self.e_length, self.e_length, bias=False)
        v = nn.Linear(self.e_length, self.e_length, bias=False)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaled Dot-Product Attention
        q, k, v = self.TW
        s = torch.matmul(q(x), torch.t(k(x)))
        w = s / math.sqrt(self.e_length) # F.normalize(s)
        w = F.softmax(w)
        return torch.matmul(w, v(x))

if __name__ == "__main__":
    embdd_len = 50
    x = torch.randint(0, 500, (10, embdd_len), dtype=torch.float32)  # tokenizer(text)
    model = SelfAttention(embdd_len)
    print(model(x))
