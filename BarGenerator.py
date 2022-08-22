import torch
import torch.nn as nn

class BarGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hid_c: int = 1024,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
    ) -> None:

        """
        Args
            z_dim (int): Dimensionality of noise space.
            hid_c (int): Hidden channel.
        """
        super(BarGenerator, self).__init__()

        self.z_dim = z_dim
        self.hid_c = hid_c
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.fc_1 = nn.Linear(4*self.z_dim, self.hid_c)

        self.relu = nn.ReLU()

        self.convT_in =  nn.ConvTranspose2d(
            self.hid_c,
            self.hid_c//2,
            kernel_size=(2,1),
            stride=(2,1),
            padding='same',
        )
        self.convT_1 =  nn.ConvTranspose2d(
            self.hid_c//2,
            self.hid_c//4,
            kernel_size=(2,1),
            stride=(2,1),
            padding='same',
        )
        self.convT_2 =  nn.ConvTranspose2d(
            self.hid_c//4,
            self.hid_c//4,
            kernel_size=(2,1),
            stride=(2,1),
            padding='same',
        )
        self.convT_3 =  nn.ConvTranspose2d(
            self.hid_c//4,
            self.hid_c//4,
            kernel_size=(1,7),
            stride=(1,7),
            padding='same',
        )
        self.convT_out =  nn.ConvTranspose2d(
            self.hid_c//4,
            1,
            kernel_size=(1,12),
            stride=(1,12),
            padding='same',
        )

        self.batchnorm_first = nn.BatchNorm2d(1, moomentum=0.9)
        self.batchnorm_in = nn.BatchNorm2d(self.hid_c, momentum=0.9)
        self.batchnorm_1 = nn.BatchNorm2d(self.hid_c/2, momentum=0.9)
        self.batchnorm_2 = nn.BatchNorm2d(self.hid_c/2, momentum=0.9)
        self.batchnorm_3 = nn.BatchNorm2d(self.hid_c/2, momentum=0.9)


        # activation layer
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.reshape(x, (2,1,512))

        x = self.relu(self.batchnorm_first(self.fc_1(x)))

        x = self.relu(self.batchnorm_in(self.convT_in(x)))

        x = self.relu(self.batchnorm_1(self.convT_1(x)))
        x = self.relu(self.batchnorm_2(self.convT_2(x)))
        x = self.relu(self.batchnorm_3(self.convT_3(x)))

        x = self.tanh((self.convT_out(x)))

        output = torch.reshape(x, (1, self.n_steps_per_bar, self.n_pitches, 1))

        return output
