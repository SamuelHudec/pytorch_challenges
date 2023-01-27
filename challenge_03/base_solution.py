from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphProd2Vec(nn.Module):
    """
    p-companion - main stage
    https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf
    """
    def __init__(self, in_embedding_len: int, out_embedding_len: int):
        super().__init__()
        self.in_embedding_len = in_embedding_len
        self.out_embedding_len = out_embedding_len
        self.encoder = self._FFN_encoder()

    def _FFN_encoder(self) -> nn.Module:
        w_1 = nn.Linear(self.in_embedding_len, self.in_embedding_len)
        bn_1 = nn.BatchNorm2d(self.in_embedding_len)
        tanh_1 = nn.Tanh()
        w_2 = nn.Linear(self.in_embedding_len, self.in_embedding_len)
        bn_2 = nn.BatchNorm2d(self.in_embedding_len)
        tanh_2 = nn.Tanh()
        w_3 = nn.Linear(self.in_embedding_len, self.out_embedding_len)
        bn_3 = nn.BatchNorm2d(self.out_embedding_len)
        return nn.Sequential(w_1, bn_1, tanh_1, w_2, bn_2, tanh_2, w_3, bn_3)

    def _GAT(self, emb: torch.tensor) -> torch.tensor:
        w = torch.matmul(emb, torch.t(emb))
        return F.softmax(w)

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = self.encoder(x)
        x_GAT = self._GAT(x)
        return x, x_GAT

if __name__ == "__main__":
    x = torch.rand(50, 10)
    labels = torch.tensor([[1, 30, 48, 2, 5],
                           [2, 4, 5, 3]])
    model = GraphProd2Vec()
    model(x)

# edge_index=[
#     [1,1,1,1,2,2,2],
#     [30,48,2,5,4,5,3]
# ]