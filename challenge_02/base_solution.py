import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    One of the most popular part in NLP.
    If you understand self attention, you will understand the heard of transformers.
    """
    def __init__(self, token_length: int) -> None:
        super().__init__()
        self.token_length = token_length
        self.TW = self._trainable_weights()

    def _trainable_weights(self):
        k = nn.Linear(self.token_length, self.token_length)
        q = nn.Linear(self.token_length, self.token_length)
        v = nn.Linear(self.token_length, self.token_length)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaled Dot-Product Attention
        q, k, v = self.TW
        s = torch.matmul(q(x), torch.t(k(x)))
        w = nn.functional.normalize(s)
        return torch.matmul(w, v(x))

if __name__ == "__main__":
    token_len = 50
    x = torch.randint(0, 500, (10, token_len), dtype=torch.float32)  # tokenizer(text)
    model = SelfAttention(token_len)
    print(model.forward(x))
