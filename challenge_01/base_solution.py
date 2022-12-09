import torch
import torch.nn as nn
import torch.utils.data as data


class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer_dim_list, activation, batch_norm, dropout_p):
        super().__init__()
        self.layer_dim_list = layer_dim_list
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self._create_mlp()

    def _create_block(self, n_in, n_out, batch_norm):
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

    def forward(self, x):
        return self.layer(x)


class FakeDataset(data.Dataset):
    def __init__(self, n_samples, n_feats):
        self.data = torch.rand((n_samples, n_feats))
        self.target = torch.randint(0, 2, size=(n_samples, 1), dtype=torch.float)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return len(self.data)


def training_loop(model, dataloader, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    current_loss = 0.0
    for batch_idx, (features, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        y_hat = model(features)
        loss = loss_function(y_hat, targets)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        print("Loss after mini-batch %5d: %.3f" % (batch_idx, current_loss))

    print("Training process has finished.")


if __name__ == "__main__":
    n_featers = 20
    fake_data = FakeDataset(500, n_featers)
    model = MultiLayerPerceptron([n_featers, 40, 40, 1], nn.ReLU, True, 0.1)
    dataloader = data.DataLoader(fake_data, batch_size=10)
    training_loop(model, dataloader, lr=0.1)
