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
        self.n_tracks = n_tracks
        
        self.conv_in = nn.Conv3d(
            self.n_tracks,
            self.hid_c,
            kernel_size=(2,1,1),
            stride=(1,1,1),
            padding=0
        )
        self.conv_1= nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(self.n_bars - 1,1,1),
            stride=(1,1,1),
            padding=0
        )

        self.conv_2 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,1,12),
            stride=(1,1,12),
            padding=0
        )
        self.conv_3 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,1,7),
            stride=(1,1,7),
            padding=0
        )
        self.conv_4 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,2,1),
            stride=(1,2,1),
            padding=0
        )
        self.conv_5 = nn.Conv3d(
            self.hid_c,
            self.hid_c,
            kernel_size=(1,2,1),
            stride=(1,2,1),
            padding=0
        )
        self.conv_6 = nn.Conv3d(
            self.hid_c,
            self.hid_c*2,
            kernel_size=(1,4,1),
            stride=(1,2,1),
            padding=(0,1,0)
        )
        self.conv_7 = nn.Conv3d(
            self.hid_c*2,
            self.hid_c*4,
            kernel_size=(1,3,1),
            stride=(1,2,1),
            padding=(0,1,0)
        )

        self.fc_1 = nn.Linear(self.hid_c*4, self.hid_c*8)
        self.flatten = nn.Flatten()
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

        x = self.flatten(x)

        x = self.leaky_relu(self.fc_1(x))

        output = self.fc_out(x)

        return output


