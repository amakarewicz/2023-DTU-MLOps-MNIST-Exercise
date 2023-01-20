import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self,
                 input_size: int = 784,
                 output_size: int = 10,
                 hidden_layers: list = [256, 128, 64]):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,
                                                      hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(
            [nn.Linear(h1, h2) for h1, h2 in layer_sizes]
            )
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = F.log_softmax(self.output(x), dim=1)

        return x
