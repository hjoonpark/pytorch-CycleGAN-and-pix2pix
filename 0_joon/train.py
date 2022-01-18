import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np

from data import create_dataset
from models import create_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("main")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_mode', type=str, default='aligned')
    parser.add_argument('--dataroot', type=str, default='../datasets/edges2shoes')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--max_dataset_size', type=float, default=float("inf"))
    parser.add_argument('--direction', type=str, default="AtoB")
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--load_size', type=int, default=286)
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
    opt, _ = parser.parse_known_args()
    dataset = create_dataset(opt)
    print("{} data loaded".format(len(dataset)))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)

    # for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    #     print(epoch)
    #     for i, data in enumerate(dataset):  # inner loop within one epoch
    #         break

    print("@@@@@@ DONE @@@@@@@@@@")