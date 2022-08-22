import torch
import argparse
import os
import torch.backends.cudnn as cudnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False, help='path to dataset')
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

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    gan = KMuseGAN(
        input_dim = data_binary.shape[1:],
        critic_learning_rate=0.001,
        optimizer='adam',
        grad_weight=10,
        z_dim=32,
        batch_size=64,
        n_tracks=4,
        n_bars=2,
        n_steps_per_bar=16,
        n_pitches=84,
    )



