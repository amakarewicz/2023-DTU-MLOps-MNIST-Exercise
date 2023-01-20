import os

import click
import matplotlib.pyplot as plt # mypy error?
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


@click.group()
def cli():
    pass


class MyDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        allow_pickle: torch.Tensor
    ):
        self.images = images
        self.labels = labels
        self.allow_pickle = allow_pickle

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return [image, label]

    def __len__(self):
        return len(self.images)


def mnist(directory: str = "data/processed/corruptmnist"):

    train_datasets = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        data = np.load(f)
        data = dict(
            zip(
                ("{}".format(item) for item in data),
                (torch.from_numpy(data[item]) for item in data),
            )
        )
        dataset = MyDataset(
            data["images"].to(torch.float32),
            data["labels"],
            data["allow_pickle"]
        )
        if "train" in filename:
            train_datasets.append(dataset)
        else:
            test = dataset
    # typing errors - fix!
    train = ConcatDataset(train_datasets)
    train = DataLoader(train, batch_size=64, shuffle=True)
    test = DataLoader(test, batch_size=64, shuffle=True)

    return train, test


@click.command()
@click.option("--lr",
              default=1e-3,
              help="learning rate to use for training")
@click.option("--optimizer",
              default=None,
              help="optimizer to use for training")
@click.option("--epochs",
              default=5,
              help="epochs to use for training")
def train(lr, optimizer, epochs):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    trainloader, _ = mnist()

    criterion = nn.NLLLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_vector = []
    for e in range(epochs):
        loss_sum = 0
        model.train()
        for images, labels in trainloader:
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().numpy()

        loss_vector.append(loss_sum)

    plt.plot(list(range(1, 5 + 1)), loss_vector)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/training_curve.png")

    checkpoint = {
        "input_size": 784,
        "output_size": 10,
        "hidden_layers": [each.out_features for each in model.hidden_layers],
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, "models/checkpoint.pth")


# cli.add_command(train)

if __name__ == "__main__":
    # cli()
    train()
