from typing import Tuple, Union

import torch
import torch.nn as nn

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
        bn_1 = nn.LayerNorm(self.in_embedding_len)
        tanh_1 = nn.Tanh()
        w_2 = nn.Linear(self.in_embedding_len, self.in_embedding_len)
        bn_2 = nn.LayerNorm(self.in_embedding_len)
        tanh_2 = nn.Tanh()
        w_3 = nn.Linear(self.in_embedding_len, self.out_embedding_len)
        bn_3 = nn.LayerNorm(self.out_embedding_len)
        return nn.Sequential(w_1, bn_1, tanh_1, w_2, bn_2, tanh_2, w_3, bn_3)

    def _GAT(self, emb: torch.tensor, edge: torch.tensor) -> torch.tensor:
        # should add test on matrix positive semi-definite
        emb_dim = emb.size(dim=0)
        edge_dim = edge.size(dim=1)
        e = torch.exp(torch.matmul(emb, torch.t(emb)))
        att_mask = torch.zeros(emb_dim, emb_dim, dtype=torch.int)
        edge = edge - 1
        for i in range(edge_dim):
            att_mask[edge[0, i], edge[1, i]] = 1
            att_mask[edge[1, i], edge[0, i]] = 1
        e_adj = e * att_mask
        den = e_adj.sum(dim=1)
        output = e.t() / den
        return output.t()

    def forward(self, x: torch.tensor, edge_index: Union[torch.tensor, None]) -> Tuple[torch.tensor, torch.tensor]:
        x = self.encoder(x)
        x_GAT = None
        if edge_index is not None:
            x_GAT = self._GAT(x, edge_index)
        return x, x_GAT

if __name__ == "__main__":
    n_items = 5
    emb_size = 50
    x = torch.rand(n_items, emb_size)
    edge_index = torch.tensor([[1, 1, 1, 1, 3, 4, 2, 2, 5, 5], [3, 4, 5, 2, 2, 5, 4, 5, 3, 1]])
    model = GraphProd2Vec(emb_size, emb_size)
    results = model(x, edge_index)
    print(results) # why is gradient in attention?

