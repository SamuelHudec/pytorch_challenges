import os
from typing import List

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10


class MultiLayerPerceptron(nn.Module):
    """
    Model class follow rules by nn.Module
    I tried to fill class by glues I got.
    Key trick is create hidden layers by blocks follow nn.Sequential
    """

    def __init__(
        self,
        layer_dim_list: List[int],
        activation: nn.modules.activation,
        batch_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.layer_dim_list = layer_dim_list
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self._create_mlp()

    def _create_block(self, n_in: int, n_out: int, batch_norm: bool) -> nn.Module:
        if batch_norm:
            block = nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.Dropout(p=self.dropout_p),
                self.activation(),
            )
        else:
            block = nn.Sequential(
                nn.Linear(n_in, n_out), nn.Dropout(p=self.dropout_p), self.activation()
            )
        return block

    def _create_mlp(self):
        blocks = []
        blocks.append(nn.Flatten())
        for i in range(len(self.layer_dim_list) - 2):
            blocks.append(
                self._create_block(
                    self.layer_dim_list[i], self.layer_dim_list[i + 1], self.batch_norm
                )
            )
        blocks.append(nn.Linear(self.layer_dim_list[-2], self.layer_dim_list[-1]))
        self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def training_loop(model: nn.Module, dataloader: data.DataLoader, lr: float):

    # define optimizer and fill arguments
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # define loss, I choose cross entropy because it fits for classification task
    loss_function = nn.CrossEntropyLoss()

    # Iterate over the DataLoader for training data
    for batch_idx, (features, targets) in enumerate(dataloader):
        # every batch you need zero the gradients
        optimizer.zero_grad()
        # forward pass
        y_hat = model(features)
        # calculate loss
        loss = loss_function(y_hat, targets)
        # backward pass
        loss.backward()
        # update parameters
        optimizer.step()
        # prepare for statistics
        print("Loss after mini-batch %5d: %.3f" % (batch_idx, loss.item()))

    print("Training process has finished.")


if __name__ == "__main__":

    # Set fixed random number seed
    torch.manual_seed(538)

    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())

    # feed training loop
    n_featers = 32 * 32 * 3
    model = MultiLayerPerceptron([n_featers, 80, 40, 10], nn.ReLU, True, 0.1)
    dataloader = data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
    training_loop(model, dataloader, lr=0.1)
