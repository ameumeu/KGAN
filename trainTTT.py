from cProfile import run
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from KMuseGAN import KMuseGAN
from utils import MidiDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="top", description="Train MusaGAN.")
    parser.add_argument("--path", type=str, default="data/Jsb16thSeparated.npz", help="Path to dataset.")
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--g_path', default='', help="path to generator (to continue training)")
    parser.add_argument('--c_path', default='', help="path to critic (to continue training)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--z_dim", type=int, default=32, help="z-dimension of noise")
    parser.add_argument("--n_tracks", type=int, default=4, help="number of tracks")
    parser.add_argument("--n_bars", type=int, default=2, help="number of bars")
    parser.add_argument("--n_steps_per_bar", type=int, default=16, help="number of steps per bar")
    parser.add_argument("--n_pitches", type=int, default=84, help="number of pitches")
    parser.add_argument("--g_lr", type=float, default=0.0001, help="learning rate of generator")
    parser.add_argument("--g_channel", type=int, default=1024, help="hidden channels of generator")
    parser.add_argument("--c_lr", type=float, default=0.0001, help="learning rate of critic")
    parser.add_argument("--c_channel", type=int, default=128, help="hidden channels of critic")
    parser.add_argument('--output_path', default='./run', help='folder to output sample and model checkpoints')

    args = parser.parse_args()
    # parameters of musegan
    gan_args = args.__dict__.copy()
    gan_args.pop("path", None)
    gan_args.pop("epochs", None)
    gan_args.pop("batch_size", None)
    gan_args.pop("output_path", None)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_path):
        print('Making output path ...')
        os.mkdir(args.output_path)
        os.mkdir(os.path.join(args.output_path, 'viz'))
        os.mkdir(os.path.join(args.output_path, 'images'))
        os.mkdir(os.path.join(args.output_path, 'weights'))
        os.mkdir(os.path.join(args.output_path, 'samples'))

    print("Loading dataset ...")
    dataset = MidiDataset(path=args.path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("Loading model ...")
    gan = KMuseGAN(
        **gan_args,
        device=device,
    )
    print("Start training ...")
    gan.train(dataloader=dataloader, output_path=args.output_path, epochs=args.epochs)
    print("Training finished.")

    # plot losses
    print("Loading plot")
    fig = plt.figure()
    plt.plot([x for x in gan.data['c_loss']], color='black', linewidth=0.25)
    plt.plot([x for x in gan.data['cr_loss']], color='green', linewidth=0.25)
    plt.plot([x for x in gan.data['cf_loss']], color='red', linewidth=0.25)
    plt.plot(gan.data['g_loss'], color='orange', linewidth=0.25)

    plt.xlabel('batch', fontsize=18)
    plt.ylabel('loss', fontsize=16)

    plt.xlim(0, len(gan.data['c_loss']))
    # plt.ylim(0, 2)

    plt.show()