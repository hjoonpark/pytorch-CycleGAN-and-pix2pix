import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import torch
import time

from data import create_dataset
from model import create_model
import matplotlib.pyplot as plt

# python test.py --dataroot ./datasets/edges2shoes --name edges2shoes_pix2pix --model pix2pix --netG resnet_6blocks --direction AtoB --dataset_mode aligned --norm batch
if __name__ == '__main__':
    root_dir = "/nobackup/joon/1_Projects/220119_Pix2pixPytorch"
    log_dir = os.path.join(root_dir, "0_joon", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'loss_log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)

    plot_path = os.path.join(log_dir, "loss_plot.jpg")
    if os.path.exists(plot_path):
        os.remove(plot_path)

    model_dir = os.path.join(root_dir, "0_joon", "model")
    os.makedirs(model_dir, exist_ok=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--serial_batches", type=bool, default=True)
    parser.add_argument("--no_flip", type=bool, default=True)
    parser.add_argument("--display_id", type=int, default=-1)
    opt, _ = parser.parse_known_args()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
