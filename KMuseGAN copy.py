import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import os
import pickle
import matplotlib.pyplot as plt

from music21 import note, stream, duration, tempo


from Generator import Generator
from Critic import Critic
from utils import initialize_weights
from criterion import WassersteinLoss, GradientPenalty

class KMuseGAN():
    def __init__(
        self,
        z_dim: int,
        n_tracks: int,
        n_bars: int,
        n_steps_per_bar: int,
        n_pitches: int,
        g_channel: int,
        c_channel: int,
        g_lr: float,
        c_lr: float,
        device: str,
        output_path: str,
    ) -> None:

        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.g_lr = g_lr
        self.c_lr = c_lr
        self.device = device

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        
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

        d_loss = self.critic.train_on_batch([true_imgs, chords_noise, style_noise,melody_noise,groove_noise], [valid, fake, dummy])
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
        dataloader, 
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
            
            g_loss = self.train_generator(batch_size)

            
            print(f"{epoch} ({critic_loops}, {1}) [D loss: ({d_loss[0]:.1f})(R {d_loss[1]:.1f}, F {d_loss[2]:.1f}, G {d_loss[3]:.1f})] [G loss: {g_loss:.1f}]")
            

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                
                self.generator.save_weights(os.path.join(run_folder, 'weights/weights-g.h5'))
                
                self.critic.save_weights(os.path.join(run_folder, 'weights/weights-c.h5'))

                self.save_model(run_folder)

            if epoch % 500 == 0:
                self.generator.save_weights(os.path.join(run_folder, f'weights/weights-g-{epoch}.h5'))
                self.critic.save_weights(os.path.join(run_folder, f'weights/weights-c-{epoch}.h5'))
                

            self.epoch+=1

    def sample_images(self, run_folder):
        r = 5

        chords_noise = np.random.normal(0, 1, (r, self.z_dim))
        style_noise = np.random.normal(0, 1, (r, self.z_dim))
        melody_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))

        gen_scores = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

        np.save(os.path.join(run_folder, f"images/sample_{self.epoch}.npy"), gen_scores)

        self.notes_to_midi(run_folder, gen_scores, 0)

    def binarise_output(self, output):
        # output is a set of scores: [batch size , steps , pitches , tracks]

        max_pitches = np.argmax(output, axis = 3)

        return max_pitches


    def notes_to_midi(self, run_folder, output, filename = None):

        for score_num in range(len(output)):

            max_pitches = self.binarise_output(output)

            midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number= 66))

            for i in range(self.n_tracks):
                last_x = int(midi_note_score[:,i][0])
                s= stream.Part()
                dur = 0
                

                for idx, x in enumerate(midi_note_score[:, i]):
                    x = int(x)
                
                    if (x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0

                    last_x = x
                    dur = dur + 0.25
                
                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)
                
                parts.append(s)

            if filename is None:
                parts.write('midi', fp=os.path.join(run_folder, "samples/sample_{}_{}.midi".format(self.epoch, score_num)))
            else:
                parts.write('midi', fp=os.path.join(run_folder, "samples/{}.midi".format(filename)))





    def save(self, folder):

            with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
                pickle.dump([
                    self.c_lr,
                    self.g_lr,
                    self.z_dim,
                    self.n_tracks,
                    self.n_bars,
                    self.n_steps_per_bar,
                    self.n_pitches,
                    ], f)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.critic.save(os.path.join(run_folder, 'critic.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

    def load_weights(self, run_folder, epoch=None):

        if epoch is None:

            self.generator.load_weights(os.path.join(run_folder, 'weights', 'weights-g.h5'))
            self.critic.load_weights(os.path.join(run_folder, 'weights', 'weights-c.h5'))
        else:
            self.generator.load_weights(os.path.join(run_folder, 'weights', f'weights-g-{epoch}.h5'))
            self.critic.load_weights(os.path.join(run_folder, 'weights', f'weights-c-{epoch}.h5'))

    def draw_bar(self, data, score_num, bar, part):
        plt.imshow(data[score_num,bar,:,:,part].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)

    def draw_score(self, data, score_num):


        fig, axes = plt.subplots(ncols=self.n_bars, nrows=self.n_tracks,figsize=(12,8), sharey = True, sharex = True)
        fig.subplots_adjust(0,0,0.2,1.5,0,0)

        for bar in range(self.n_bars):
            for track in range(self.n_tracks):

                if self.n_bars > 1:
                    axes[track, bar].imshow(data[score_num,bar,:,:,track].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
                else:
                    axes[track].imshow(data[score_num,bar,:,:,track].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
