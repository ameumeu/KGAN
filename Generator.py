import torch
import torch.nn as nn
from torchvision.transforms import Lambda

from TemporalNetwork import TemporalNetwork
from BarGenerator import BarGenerator

class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hid_c: int = 1024,
        n_tracks: int = 4,
        n_bars: int = 2,
    ) -> None:
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.hid_c = hid_c
        self.n_tracks = n_tracks
        self.n_bars = n_bars 

        # chords -> TemporalNetwork
        self.chords_tempNet = TemporalNetwork(
            z_dim=self.z_dim,
            hid_c=self.hid_c
        )

        # melody -> TemmporalNetwork
        self.melody_tempNet = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.melody_tempNet[track] = TemporalNetwork()

        # BarGenerator per track
        self.barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.barGen[track] = BarGenerator()

        # output per track and bar

    
    def forward(
        self,
        chords_input: torch.Tensor,
        style_input: torch.Tensor,
        melody_input: torch.Tensor,
        groove_input: torch.Tensor,
    ) -> torch.Tensor:

        chords_over_time = self.chords_tempNet(chords_input) # [n_bars, z_dim]

        melody_over_time = [None] * self.n_tracks # list of n_tracks [n_bars, z_dim] tensors
        for track in range(self.n_tracks):
            melody_track = Lambda(lambda x: x[:,track,:])(melody_input)
            melody_over_time[track] = self.melody_tempNet[track](melody_track)

        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:,bar,:])(chords_over_time)
            s = style_input # [z_dim]

            for track in range(self.n_tracks):

                m = Lambda(lambda x: x[:,bar,:])(melody_over_time[track]) # [z_dim]
                g = Lambda(lambda x: x[:,track,:])(groove_input) # [z_dim]

                z_input = torch.cat([c,s,m,g], axis=1)

                track_output[track] = self.barGen(z_input)

            bars_output[bar] = torch.cat(track_output, axis=-1)

        generator_output = torch.cat(bars_output, axis=1)