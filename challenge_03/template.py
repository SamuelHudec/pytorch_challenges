import torch
import torch.nn as nn


class GraphProd2Vec(nn.Module):
    """
    p-companion - first stage
    https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf
    """
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.GAN = None

    def forward(self, x):

        x = self.encoder(x)

        x_GAN = self.GAN(x)

        return x, x_GAN

x = torch.rand(50, 10)
labels = torch.tensor([[1,30,48,2,5],
                  [2,4,5,3]])
# edge_index=[
#     [1,1,1,1,2,2,2],
#     [30,48,2,5,4,5,3]
# ]