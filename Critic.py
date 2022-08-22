import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hid_c: int = 128,
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
        n_tracks: int = 4,
    ):
        super(Critic, self).__init__()

        self.z_dim = z_dim
        self.hid_c = hid_c
        self.n_bars = n_bars
        
        self.conv_in = nn.Conv3d(
            self.z_dim,
            self.hid_c,
            kernel_size=(2,1,1),
            stride=(1,1,1),
            padding='valid'
        )
        self.conv_1= nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(self.n_bars - 1,1,1),
            stride=(1,1,1),
            padding='valid'
        )

        self.conv_2 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,1,12),
            stride=(1,1,12),
            padding='same'
        )
        self.conv_3 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,1,7),
            stride=(1,1,7),
            padding='same'
        )
        self.conv_4 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,2,1),
            stride=(1,2,1),
            padding='same'
        )
        self.conv_5 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,2,1),
            stride=(1,2,1),
            padding='same'
        )
        self.conv_6 = nn.Conv3d(
            self.hid_c,
            self.hid_c*2,
            kernel_size=(1,4,1),
            stride=(1,2,1),
            padding='same'
        )
        self.conv_7 = nn.Conv3d(
            self.hid_c*2,
            self.hid_c*4,
            kernel_size=(1,3,1),
            stride=(1,2,1),
            padding='same'
        )

        self.fc_1 = nn.Linear(1, self.hid_c*8)
        self.fc_out = nn.Linear(self.hid_c*8, 1)

        # activation layer
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.leaky_relu(self.conv_in(x))
        x = self.leaky_relu(self.conv_1(x))

        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_3(x))
        x = self.leaky_relu(self.conv_4(x))
        x = self.leaky_relu(self.conv_5(x))
        x = self.leaky_relu(self.conv_6(x))
        x = self.leaky_relu(self.conv_7(x))

        x = torch.flatten(x)

        x = self.leaky_relu(self.fc_1(x))

        output = self.fc_out(x)

        return output


