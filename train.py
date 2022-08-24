import torch
import argparse
import os
import torch.backends.cudnn as cudnn

import numpy as np

from KMuseGAN import KMuseGAN
from utils import load_music

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False, default="/data", help='path to dataset')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--generator_path', default='', help="path to generator (to continue training)")
    parser.add_argument('--critic_path', default='', help="path to critic (to continue training)")
    parser.add_argument('--g_lr', default=0.0001, help="learning rate of generator")
    parser.add_argument('--c_lr', default=0.0001, help="kearning rate of critic")
    parser.add_argument('--output_path', default='.', help='folder to output sample and model checkpoints')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.output_path)
    except OSError:
        pass

    DATAROOT = "/Users/ameu/Documents/KGan/Jsb16thSeparated.npz"#opt.dataroot

    # run params
    SECTION = 'test'
    RUN_ID = '1'
    DATA_NAME = 'chorales'
    FILENAME = 'Jsb16thSeparated.npz'
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'samples'))


    # CUDA
    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if opt.cuda else "cpu")


    BATCH_SIZE = 64
    z_dim=32
    n_tracks=4
    n_bars=2
    n_steps_per_bar=16
    n_pitches=84

    gan = KMuseGAN(
        z_dim=z_dim,
        n_tracks=n_tracks,
        n_bars=n_bars,
        n_steps_per_bar=n_steps_per_bar,
        n_pitches=n_pitches,
        device=device,
    )
    #  batch_size=64
    print(gan.generator)
    print(gan.critic)



    # load score data
    data_binary, data_ints, raw_data = load_music(DATAROOT, n_bars, n_steps_per_bar)
    data_binary = np.squeeze(data_binary)



    EPOCHS = 6000
    PRINT_EVERY_N_BATCHES = 10

    gan.epoch = 0

    gan.train(     
        data_binary,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        run_folder = RUN_FOLDER,
        print_every_n_batches = PRINT_EVERY_N_BATCHES,
    )




