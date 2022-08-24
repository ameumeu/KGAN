import torch
import torch.nn as nn

class TemporalNetwork(nn.Module):
    def __init__(
        self, 
        z_dim: int=32,
        hid_c: int=1024,
    ) -> None:
        """
        Args
            z_dim (int): Dimensionality of noise space.
            hid_c (int): Hidden channel.
        """
        super(TemporalNetwork, self).__init__()

        self.z_dim = z_dim
        self.hid_c = hid_c

        # arg 'n_bars' == 2 in the book
        self.n_bars = 2

        # Convolution 2D Transpose Layers
        self.convT_in = nn.ConvTranspose2d(
            in_channels=self.z_dim,
            out_channels=self.hid_c,
            kernel_size=(2,1),
            stride=(1,1),
            padding=0
        )
        self.convT_out = nn.ConvTranspose2d(
            in_channels=self.hid_c,
            out_channels=self.z_dim,
            kernel_size=(self.n_bars-1,1),
            stride=(1,1),
            padding=0
        )

        self.batchnorm_1 = nn.BatchNorm2d(self.hid_c, momentum=0.9)
        self.batchnorm_2 = nn.BatchNorm2d(self.z_dim, momentum=0.9)

        # activation layer
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.reshape(x, [x.size(0), self.z_dim, 1, 1])

        x = self.relu(self.batchnorm_1(self.convT_in(x)))

        x = self.leaky_relu(self.batchnorm_2(self.convT_out(x)))

        output = torch.reshape(x, (x.size(0), self.n_bars, self.z_dim))

        return output
        

