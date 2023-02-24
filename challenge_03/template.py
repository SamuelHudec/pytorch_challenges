import torch
import torch.nn as nn


class GraphProd2Vec(nn.Module):
    """
    p-companion - main stage section 4.1
    https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf
    """

    def __init__(self, in_embedding_len: int, out_embedding_len: int):
        super().__init__()
        self.encoder = None

    def forward(self, x, edge_index):

        x = self.encoder(x)

        x_GAT = self._GAT(x, edge_index)

        return x, x_GAT


if __name__ == "__main__":
    n_items = 6
    emb_size = 50
    x = torch.rand(n_items, emb_size)
    edge_index = torch.tensor(
        [[1, 1, 1, 1, 3, 4, 2, 2, 5, 5, 6, 6], [3, 4, 5, 2, 2, 5, 4, 5, 3, 1, 4, 2]]
    )
    labels = torch.tensor(
        [-1, 1, -1, 1, -1, 1]
    )  # TODO: is negative and positive calculated together?
    model = GraphProd2Vec(emb_size, emb_size)
    results = model(x, edge_index)
    print(results)

    # # Optional part
    # loss = HingeLoss()
    # print(loss(results, labels))
