import os
import json
import einops
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid, save_image


def plot_result(losses, args):
    epoch_full = np.arange(0, args.num_epochs)
    epoch_sub = np.arange(0, args.num_epochs, 5)
    epoch_sub = np.append(epoch_sub, epoch_full[-1])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plt.title('Training loss curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epoch_full, losses, label='mse')  # c='silver',
    ax1.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('{}/trainingCurve.png'.format(args.figure_dir))
    print("-- Save training figure")
