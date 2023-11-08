import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from base_solution import MultiLayerPerceptron


def training_loop(model: nn.Module, dataloader: data.DataLoader, lr: float):

    # define optimizer and fill arguments
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    model = MultiLayerPerceptron([n_featers, 1500, 500, 100, 10], nn.ReLU, True, 0.1)
    dataloader = data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
    training_loop(model, dataloader, lr=0.1)
