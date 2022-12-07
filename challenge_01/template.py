import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer_dim_list, activation, batch_norm, dropout_p):

    def create_mlp(self):
        pass

    def forward(self):
        pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, n_feats):
        data = torch.rand((n_samples, n_feats))
        target = torch.randint(0, 2, size=(n_samples, 1))

    def __getitem__(self, idx):
        return None

    def __len__(self):
        return None


def training_loop(model, dataloader):
    for batch in dataloader:
        model()


model = MultiLayerPerceptron([20, 40, 40, 1], torch.relu, True, 0.1)
dataloader = torch.utils.data.DataLoader(Dataset, batch_size=)

training_loop(model, dataloader)