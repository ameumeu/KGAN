import torch
import torch.nn as nn
import numpy as np

from loaders import load_music


class ChordsLSTM(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(ChordsLSTM, self).__init__()

        self.lstm = nn.LSTM(10, 20, 2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x, (hn,cn) = self.lstm(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    import sys
    sys.path.append('./LSTM')
    BATCH_SIZE = 64
    n_bars = 2
    n_steps_per_bar = 16
    n_pitches = 84
    n_tracks = 4

    FILEPATH = './LSTM/data/Jsb16thSeparated.npz'

    data_binary, data_ints, raw_data = load_music(FILEPATH, n_bars, n_steps_per_bar)
    data_binary = np.squeeze(data_binary)
    cl = ChordsLSTM()
    print(cl)
