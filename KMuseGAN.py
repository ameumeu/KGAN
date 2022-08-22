import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Iterable

from Generator import Generator
from Critic import Critic
from utils import initialize_weights
from criterion import WassersteinLoss, GradientPenalty

class MuseGAN():
    def __init__(
        self,
        z_dim: int = 32,
        n_tracks: int = 4,
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
        g_channel: int = 1024,
        c_channel: int = 128,
        g_lr: float = 0.001,
        c_lr: float = 0.001,
        device: str = "cuda:0",
    ) -> None:

        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.device = device
        
        # build generator
        self.generator = Generator(
            z_dim=z_dim,
            hid_c=g_channel,
        ).to(device)
        self.generator.apply(initialize_weights)
        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.9)
        )

        # build critic
        self.critic = Critic(
            z_dim=z_dim,
            hid_c=c_channel,
        ).to(device)
        self.critic.apply(initialize_weights)
        self.c_optim = optim.Adam(
            self.critic.parameters(),
            lr=c_lr,
            betas=(0.5, 0.9),
        )

        # loss function and gradient penalty (wasserstein)
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)

        # dictionary
        self.data = {
            "g_loss": [],
            "c_loss": [],
            "cf_loss": [],
            "cr_loss": [],
            "cp_loss": [],
        }
        print("KMuseGan is ready!")

    def train_critic(
        self,
        x_train,
        batch_size,
        using_generator,
    ):
        valid = np.ones((batch_size,1), dtype=np.float32)
        fake = -np.ones((batch_size,1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32) # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        d_loss = self.critic_model.train_on_batch([true_imgs, chords_noise, style_noise,melody_noise,groove_noise], [valid, fake, dummy])
        return d_loss

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1), dtype=np.float32)
        
        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        return self.model.train_on_batch([chords_noise, style_noise,melody_noise,groove_noise], valid)

    def train(
        self,
        x_train, 
        batch_size: int = 64,
        epochs: int = 500,
        run_folder: str = '',
        print_every_n_batches: int = 10,
        n_critic: int = 5,
        using_generator: bool = False
    ) -> None:

        self.alpha = torch.rand((batch_size,1,1,1,1)).requires_grad_().to(self.device)

        for epoch in range(epochs):
            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic

            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)