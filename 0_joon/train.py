import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import torch

from data import create_dataset
from model import create_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("main")
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("available gpus:", available_gpus)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = "cpu"
        

    root_dir = "/nobackup/joon/1_Projects/220119_Pix2pixPytorch"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_mode', type=str, default='aligned')
    parser.add_argument('--dataroot', type=str, default='{}/datasets/edges2shoes'.format(root_dir))
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--max_dataset_size', type=float, default=float("inf"))
    parser.add_argument('--direction', type=str, default="AtoB")
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--serial_batches', type=bool, default=False)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--preprocess', type=str, default="resize_and_scale")
    parser.add_argument('--no_flip', type=bool, default=False)
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_epochs_decay', type=int, default=100)

    # model
    parser.add_argument("--model", type=str, default="pix2pix")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--isTrain", type=bool, default=True)
    parser.add_argument("--print_freq", type=int, default=100)


    opt, _ = parser.parse_known_args()
    dataset = create_dataset(opt)
    print("{} data loaded".format(len(dataset)))

    model = create_model(device, opt)      # create a model given opt.model and other options

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for i, data in enumerate(dataset):  # inner loop within one epoch
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            total_iters += opt.batch_size

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(losses)

    print("@@@@@@ DONE @@@@@@@@@@")