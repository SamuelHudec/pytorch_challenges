from typing import List, Tuple, Callable

import torch
import torch.nn as nn
import torch.utils.data as data


class MultiLayerPerceptron(nn.Module):
    """
    Model class follow rules by nn.Module
    I tried to fill class by glues I got.
    Key trick is create hidden layers by blocks follow nn.Sequential
    """

    def __init__(
        self,
        layer_dim_list: List[int],
        activation: Callable,
        batch_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.layer_dim_list = layer_dim_list
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.layer = self._create_mlp()

    def _create_block(self, n_in: int, n_out: int, batch_norm: bool) -> nn.Module:
        block = []
        block.append(nn.Linear(n_in, n_out))
        if batch_norm:
            block.append(nn.BatchNorm1d(n_out))
        block.append(nn.Dropout(p=self.dropout_p))
        block.append(self.activation())
        return nn.Sequential(*block)

    def _create_mlp(self) -> nn.Module:
        blocks = []
        blocks.append(nn.Flatten())
        for i in range(len(self.layer_dim_list) - 2):
            blocks.append(
                self._create_block(
                    self.layer_dim_list[i], self.layer_dim_list[i + 1], self.batch_norm
                )
            )
        blocks.append(nn.Linear(self.layer_dim_list[-2], self.layer_dim_list[-1]))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FakeDataset(data.Dataset):
    def __init__(self, n_samples: int, n_feats: int) -> None:
        self.data = torch.rand((n_samples, n_feats))
        self.target = torch.randint(0, 2, size=(n_samples, 1), dtype=torch.float)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.target[idx]

    def __len__(self) -> int:
        return len(self.data)


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
    n_featers = 20
    fake_data = FakeDataset(500, n_featers)
    model = MultiLayerPerceptron([n_featers, 40, 40, 1], nn.ReLU, True, 0.1)
    dataloader = data.DataLoader(fake_data, batch_size=10)
    training_loop(model, dataloader, lr=0.1)
