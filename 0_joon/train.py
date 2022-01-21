import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import torch
import time

from data import create_dataset
from model import create_model
import matplotlib.pyplot as plt

def print_current_losses(log_path, epoch, epoch_iters, dataset_size, losses, t_comp, t_data):
    message = '(epoch: %d, epoch_iters: %d/%d, time: %.3f, data: %.3f) ' % (epoch, epoch_iters, dataset_size, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    # print(message)  # print the message
    with open(log_path, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

def plot_current_losses(save_path, epoch, epoch_iters, dataset_size, total_iters, losses):
    plt.figure(figsize=(10, 5))
    legends = []
    for k, l in losses.items():
        x = np.arange(1, 1+len(l)) * total_iters / len(l)
        plt.plot(x, l)
        legends.append(k)
    plt.legend(legends, loc='upper right')
    plt.xlabel("Total iterations")
    plt.title("epoch {} | dataset {}/{} | total iters {}".format(epoch, epoch_iters, dataset_size, total_iters))
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_result(save_path, epoch, epoch_iters, dataset_size, total_iters, real_A, real_B, fake_B):
    n_imgs = real_A.shape[0]
    row_height = 2
    fig = plt.figure(figsize=(row_height*3, row_height*(n_imgs+1)))
    for r in range(n_imgs):
        rA = np.transpose(real_A[r, :, :, :], (1, 2, 0))
        rB = np.transpose(real_B[r, :, :, :], (1, 2, 0))
        fB = np.transpose(fake_B[r, :, :, :], (1, 2, 0))
        img = np.clip(np.hstack((rA, rB, fB)), a_min=0, a_max=1)
        ax = fig.add_subplot(n_imgs, 1, r+1)
        ax.imshow(img)
    plt.tight_layout()
    title = "epoch {} | dataset {}/{} | total iters {}".format(epoch, epoch_iters, dataset_size, total_iters)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    print("main")
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("available gpus:", available_gpus)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = "cpu"

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
    parser.add_argument('--dataset_mode', type=str, default='aligned')
    parser.add_argument('--dataroot', type=str, default='{}/datasets/edges2shoes'.format(root_dir))
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--max_dataset_size', type=float, default=float("inf"))
    parser.add_argument('--direction', type=str, default="AtoB")
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--serial_batches', type=bool, default=False)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--preprocess', type=str, default="resize_and_scale")
    parser.add_argument('--no_flip', type=bool, default=False)
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_epochs_decay', type=int, default=100)
    # model
    parser.add_argument("--model", type=str, default="pix2pix")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--isTrain", type=bool, default=True)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000)

    opt, _ = parser.parse_known_args()
    # opt.max_dataset_size = 123
    dataset = create_dataset(opt)
    print("{} data loaded".format(len(dataset)))

    model = create_model(device, opt)      # create a model given opt.model and other options

    total_iters = 0                # the total number of training iterations
    losses_all = {}
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        
            if epoch_iter == opt.batch_size:
                save_path = os.path.join(log_dir, "visual_{}.jpg".format(total_iters))
                visualize_result(save_path, epoch, epoch_iter, len(dataset), total_iters, model.real_A.detach().cpu().numpy(), model.real_B.detach().cpu().numpy(), model.fake_B.detach().cpu().numpy())

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                for k, v in losses.items():
                    if k not in losses_all:
                        losses_all[k] = []
                    losses_all[k].append(v)

                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(log_path, epoch, epoch_iter, len(dataset), losses, t_comp, t_data)
                plot_current_losses(plot_path, epoch, epoch_iter, len(dataset), total_iters, losses_all)

            if total_iters > 0 and total_iters & opt.save_freq == 0:
                model_path = os.path.join(model_dir, "pix2pix_edges2shoes.pt")
                model.save_networks(model_path)
                
            iter_data_time = time.time()

    print("@@@@@@ DONE @@@@@@@@@@")