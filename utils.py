import torch
import torch.nn as nn

# Made because of an error, but fixed
class Lambda(nn.Module):
    def __init__(self, lambd) -> None:
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def initialize_weights(
    layer: nn.Module,
    mean: float = 0.0,
    std: float = 0.02,
):
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)