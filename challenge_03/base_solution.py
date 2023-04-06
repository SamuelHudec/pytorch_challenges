import typing
from typing import Tuple, Union

import torch
import torch.nn as nn


class GraphProd2Vec(nn.Module):
    """
    p-companion - main stage section 4.1
    https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf
    """

    def __init__(self, in_embedding_len: int, out_embedding_len: int):
        super().__init__()
        self.in_embedding_len = in_embedding_len
        self.out_embedding_len = out_embedding_len
        self.encoder = self._FFN_encoder()

    def _FFN_encoder(self) -> nn.Module:
        """
        To instance this encoder we can use our solution from challenge_01
        eg  self.encoder = MultiLayerPerceptron([self.in_embedding_len, self.in_embedding_len, self.in_embedding_len, self.out_embedding_len], nn.Tanh, True, 0)
        """
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
        # edge labels must start with 0
        labels_unique = (edge[0]).unique(dim=0)
        e = torch.exp((emb[edge[0]] * emb[edge[1]]).sum(-1))
        e_ni = torch.zeros_like(labels_unique, dtype=torch.float).scatter_add_(0, edge[0], e)
        z_ij = e / e_ni[edge[0]]
        attention = emb[edge[1]] * z_ij.unsqueeze(-1)
        labels_att = edge[0].view(edge[0].size(0), 1).expand(-1, attention.size(1))
        unique_labels_att = labels_att.unique(dim=0)
        return torch.zeros_like(unique_labels_att, dtype=torch.float).scatter_add_(0, labels_att, attention)

    def forward(
        self, x: torch.tensor, edge_index: Union[torch.tensor, None]
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = self.encoder(x)
        x_GAT = None
        if edge_index is not None:
            x_GAT = self._GAT(x, edge_index)
        return x, x_GAT


class HingeLoss(nn.Module):
    def __init__(self, hinge_lambda: float = 1.0, hinge_epsilon: float = 1.0):
        """
        based on section last part of section. Its good to google more about hinge loss in general.
        If you are not familiar with.
        """
        super().__init__()
        self.hinge_lambda = hinge_lambda
        self.hinge_epsilon = hinge_epsilon

    def forward(self, x: Tuple[torch.tensor, torch.tensor], labels: torch.tensor):

        theta_subtraction = x[0] - x[1]
        theta_norm = torch.norm(theta_subtraction, p=2, dim=1) ** 2
        theta_count = self.hinge_epsilon - labels * (self.hinge_lambda - theta_norm)

        mask = labels == 1
        positive_loss = torch.max(theta_count[mask], torch.Tensor([0]))
        negative_loss = torch.max(theta_count[~mask], torch.Tensor([0]))
        assert (
            positive_loss.size() == negative_loss.size()
        ), "positive and negative examples must be equal"
        return torch.sum(positive_loss + negative_loss)


if __name__ == "__main__":
    n_items = 6
    emb_size = 50
    x = torch.rand(n_items, emb_size)
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 2, 3, 1, 1, 4, 4, 5, 5], [2, 3, 4, 1, 1, 4, 3, 4, 2, 0, 3, 1]]
    )
    labels = torch.tensor(
        [-1, 1, -1, 1, -1, 1]
    )  # TODO: is negative and positive calculated together?
    model = GraphProd2Vec(emb_size, emb_size)
    loss = HingeLoss()
    results = model(x, edge_index)
    print(loss(results, labels))
