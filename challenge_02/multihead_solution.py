from typing import Tuple

import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    One of the most popular part in NLP.
    If you understand self attention, you will understand the heard of transformers.
    """

    def __init__(self, embedding_length: int, heads: int, mask=None) -> None:
        super().__init__()
        self.e_length = embedding_length
        self.heads = heads
        self.mask = mask
        self.one_head = int(embedding_length / self.heads)
        self.TW = self._trainable_weights()

    def _trainable_weights(self) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        q = nn.Linear(self.one_head, self.one_head, bias=False)
        k = nn.Linear(self.one_head, self.one_head, bias=False)
        v = nn.Linear(self.one_head, self.one_head, bias=False)
        out_cnct = nn.Linear(self.e_length, self.e_length)
        return q, k, v, out_cnct

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # Scaled Dot-Product Attention
        q, k, v, out_cnct = self.TW

        batch_size = key.size(0)
        seq_len = key.size(1)
        seq_len_query = query.size(1)  # for decoder

        query = query.view(batch_size, seq_len_query, self.heads, self.one_head)
        key = key.view(batch_size, seq_len, self.heads, self.one_head)
        value = value.view(batch_size, seq_len, self.heads, self.one_head)

        query = q(query)
        key = k(key)
        value = v(value)

        # transpose to get right dimensions, this almost blow my mind
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        s = torch.matmul(query, key.transpose(-1, -2))

        if self.mask is not None:
            s = s.masked_fill(self.mask == 0, float("-1e20"))

        w = s / math.sqrt(self.one_head)
        w = F.softmax(w, dim=-1)
        to_concat = torch.matmul(w, value)

        # compress it together
        to_concat = (
            to_concat.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_query, self.e_length)
        )

        return out_cnct(to_concat)


if __name__ == "__main__":
    embdd_len = 512
    heads = 8
    x = torch.randint(0, 500, (32, 10, embdd_len), dtype=torch.float32)
    model = MultiHeadAttention(embdd_len, heads)
    print(model(x, x, x))
