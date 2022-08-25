import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import os
import pickle
from typing import Iterable
from progress.bar import IncrementalBar
import time
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
        g_path: str,
        c_path: str,
        g_channel: int,
        c_channel: int,
        g_lr: float,
        c_lr: float,
        device: str,
    ) -> None:

        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.g_channel = g_channel
        self.c_channel = c_channel
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
            device=self.device,
        ).to(device)
        self.generator.apply(initialize_weights)
        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.9)
        )
        if g_path:
            self.generator.load_state_dict(torch.load(g_path))
            self.generator.eval()

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
        if c_path:
            self.critic.load_state_dict(torch.load(c_path))
            self.critic.eval()

        # loss function and gradient penalty (wasserstein)
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)

        # dictionary
        self.data = {
            "g_loss": [], # generator loss
            "c_loss": [], # critic loss
            "cf_loss": [], # critic fake loss
            "cr_loss": [], # critic real loss
            "cp_loss": [], # critic penalty loss
        }
        print("KMuseGan is ready!")

    def train(
        self,
        dataloader: Iterable,
        output_path: str,
        epochs: int = 500,
        batch_size: int = 64,
        display_epoch: int = 100,
        n_critic = 5,
    ) -> None:
        """Train GAN.
        Parameters
        ----------
        dataloader: Iterable
            Dataloader.
        epochs: int, (default=500)
            Number of epochs.
        batch_size: int, (default=64)
            Batch size.
        display_epoch: int, (default=10)
            Display step.
        """
        # alpha parameter for mixing images
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)
        for epoch in range(self.epoch, self.epoch + epochs):
            ge_loss, ce_loss = 0, 0
            cfe_loss, cre_loss, cpe_loss = 0, 0, 0
            start = time.time()
            bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(dataloader))
            for real in dataloader:
                real = real.to(self.device)
                # train Critic
                cb_loss = 0
                cfb_loss, crb_loss, cpb_loss = 0, 0, 0

                if epoch % 100 == 0:
                    critic_loops = 5
                else:
                    critic_loops = n_critic

                for _ in range(critic_loops):
                    # create random `noises`
                    chords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, 4, 32).to(self.device)
                    groove = torch.randn(batch_size, 4, 32).to(self.device)
                    # forward to generator
                    self.c_optim.zero_grad()
                    with torch.no_grad():
                        fake = self.generator(chords, style, melody, groove).detach()
                    # mix `real` and `fake` melody
                    realfake = self.alpha * real + (1. - self.alpha) * fake
                    # get critic's `fake` loss
                    fake_pred = self.critic(fake)
                    fake_target = - torch.ones_like(fake_pred)
                    fake_loss = self.c_criterion(fake_pred, fake_target)
                    # get critic's `real` loss
                    real_pred = self.critic(real)
                    real_target = torch.ones_like(real_pred)
                    real_loss = self.c_criterion(real_pred, real_target)
                    # get critic's penalty
                    realfake_pred = self.critic(realfake)
                    penalty = self.c_penalty(realfake, realfake_pred)
                    # sum up losses
                    closs = fake_loss + real_loss + 10 * penalty
                    # retain graph
                    closs.backward(retain_graph=True)
                    # update critic parameters
                    self.c_optim.step()
                    # devide by number of critic updates in the loop (5)
                    cfb_loss += fake_loss.item() / 5
                    crb_loss += real_loss.item() / 5
                    cpb_loss += 10 * penalty.item() / 5
                    cb_loss += closs.item() / 5

                cfe_loss += cfb_loss / len(dataloader)
                cre_loss += crb_loss / len(dataloader)
                cpe_loss += cpb_loss / len(dataloader)
                ce_loss += cb_loss / len(dataloader)

                # train generator
                self.g_optim.zero_grad()
                # create random `noises`
                chords = torch.randn(batch_size, 32).to(self.device)
                style = torch.randn(batch_size, 32).to(self.device)
                melody = torch.randn(batch_size, 4, 32).to(self.device)
                groove = torch.randn(batch_size, 4, 32).to(self.device)
                # forward to generator
                fake = self.generator(chords, style, melody, groove)
                # forward to critic (to make prediction)
                fake_pred = self.critic(fake)
                # get generator loss (idea is to fool critic)
                gb_loss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                gb_loss.backward()
                # update critic parameters
                self.g_optim.step()
                ge_loss += gb_loss.item() / len(dataloader)
                bar.next()
            bar.finish()
            end = time.time()
            tm = (end - start)
            # save history
            self.data['g_loss'].append(ge_loss)
            self.data['c_loss'].append(ce_loss)
            self.data['cf_loss'].append(cfe_loss)
            self.data['cr_loss'].append(cre_loss)
            self.data['cp_loss'].append(cpe_loss)
            # display losses
            print(f"[Epoch {epoch+1}/{epochs}] [G loss: {ge_loss:.3f}] [D loss: {ce_loss:.3f}] ETA: {tm:.3f}s")
            print(f"[C loss | (fake: {cfe_loss:.3f}, real: {cre_loss:.3f}, penalty: {cpe_loss:.3f})]")
            if epoch % display_epoch == 0:
                self.sample_images(output_path)
                
                torch.save(self.generator.state_dict(), os.path.join(output_path, f'weights/weights-g-{epoch}.pth'))
                
                torch.save(self.critic.state_dict(), os.path.join(output_path, f'weights/weights-c-{epoch}.pth'))

            self.epoch += 1


    def sample_images(self, output_path):
        r = 5

        chords_input = torch.normal(0, 1, (r, self.z_dim)).to(self.device)
        style_input = torch.normal(0, 1, (r, self.z_dim)).to(self.device)
        melody_input = torch.normal(0, 1, (r, self.n_tracks, self.z_dim)).to(self.device)
        groove_input = torch.normal(0, 1, (r, self.n_tracks, self.z_dim)).to(self.device)

        with torch.no_grad():
            gen_scores = self.generator(chords_input, style_input, melody_input, groove_input).detach()

        np.save(os.path.join(output_path, f"images/sample_{self.epoch}.npy"), gen_scores.to('cpu'))

        self.notes_to_midi(output_path, gen_scores, 0)

    def binarise_output(self, output):
        # output is a set of scores: [batch size , steps , pitches , tracks]

        max_pitches = np.argmax(output.to('cpu'), axis = 3)

        return max_pitches


    def notes_to_midi(self, output_path, output, filename = None):

        for score_num in range(len(output)):

            max_pitches = self.binarise_output(output)

            # midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            midi_note_score = max_pitches[score_num].T.reshape([-1, self.n_tracks])
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
                parts.write('midi', fp=os.path.join(output_path, "samples/sample_{}_{}.midi".format(self.epoch, score_num)))
            else:
                parts.write('midi', fp=os.path.join(output_path, "samples/{}.midi".format(filename)))


    def save(self, folder):

            with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
                pickle.dump([
                    self.z_dim,
                    self.n_tracks,
                    self.n_bars,
                    self.n_steps_per_bar,
                    self.n_pitches,
                    self.g_channel,
                    self.c_channel,
                    self.g_lr,
                    self.c_lr,
                    ], f)

    def load_weights(self, output_path, epoch=None):

        if epoch is None:

            self.generator.load_state_dict(torch.load(os.path.join(output_path, 'weights', 'weights-g.h5')))
            self.critic.load_state_dict(torch.load(os.path.join(output_path, 'weights', 'weights-c.h5')))
        else:
            self.generator.load_state_dict(torch.load(os.path.join(output_path, 'weights', f'weights-g-{epoch}.h5')))
            self.critic.load_state_dict(torch.load(os.path.join(output_path, 'weights', f'weights-c-{epoch}.h5')))

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