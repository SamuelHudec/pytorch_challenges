import torch
import torch.nn as nn


class GraphProd2Vec(nn.Module):
    """
    p-companion - main stage
    https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf
    """
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.GAT = None

    def forward(self, x):

        x = self.encoder(x)

        x_GAT = self.GAT(x)

        return x, x_GAT

if __name__ == "__main__":
    n_items = 5
    emb_size = 50
    x = torch.rand(n_items, emb_size)
    edge_index = torch.tensor([[1, 1, 1, 1, 3, 4, 2, 2, 5, 5], [3, 4, 5, 2, 2, 5, 4, 5, 3, 1]])
    model = GraphProd2Vec(emb_size, emb_size)
    results = model(x, edge_index)
    print(results)