from typing import Tuple

import torch
import torch.nn as nn
from challenge_01.base_solution import MultiLayerPerceptron


class ComplementaryTypeTransition(nn.Module):
    """
    1. Model the asymmetric relationship between query product type and complementary product types
    2. Generate diversified complementary product types for further item recommendation.
    """
    def __init__(self, type_dim, drop_out):
        super().__init__()
        self.type_enc_dec = MultiLayerPerceptron([type_dim, int(type_dim/2) ,type_dim], nn.ReLU, False, drop_out)

    def forward(self, type):
        return self.type_enc_dec(type)


class ComplementaryItemPrediction(nn.Module):
    """
     Output item recommendations given the embeddings of query product and inferred multiple complementary types.
    """
    def __init__(self, type_dim, emb_dim):
        super().__init__()
        self.weights = nn.Linear(type_dim, emb_dim)

    def forward(self, type, product):
        # dont forget, element wise is under threshold beta, equation 7 in paper
        return torch.mul(self.weights(type), product)


if __name__ == "__main__":
    n_items = 5 # edges = torch.randint(0,(n_items - 1), (2, n_items))
    type_size = 40
    product_size = 50
    type_embedding = torch.rand(n_items, type_size)
    product_to_vec_embedding = torch.rand(n_items, product_size)

    type_modul = ComplementaryTypeTransition(type_size, 0.2)
    complementary_modul = ComplementaryItemPrediction(type_size, product_size)

    type_modul(type_embedding)
    complementary_modul(type_embedding, product_to_vec_embedding)
