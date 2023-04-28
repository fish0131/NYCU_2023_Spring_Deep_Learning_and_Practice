import os
import re
import torch
import numpy as np
import seaborn as sn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataloader import RetinopathyLoader
from train import ResNet18, ResNet50, evaluate

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def plot_confusion_matrix(y_true, y_pred, labels, fn):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize='true')
    fig, ax = plt.subplots()
    sn.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.1f')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')
    ax.xaxis.set_ticklabels(labels, rotation=45)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.title('Normalized comfusion matrix')
    plt.savefig(fn, dpi=300)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    pt_path = "./model_weights/resnet50_lr0.001_20epoch_bs8_lr_decay_pretrained/epoch20.pt"
    # ResNet18 without pretraining: resnet18_lr0.001_10epoch_bs32_weight_decay
    # ResNet18 with pretraining: resnet18_lr0.001_10epoch_bs16_weight_decay_pretrained
    # ResNet50 without pretraining: resnet50_lr0.001_10epoch_bs16
    # ResNet50 with pretraining: resnet50_lr0.001_20epoch_bs8_lr_decay_pretrained

    if args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'resnet50':
        model = ResNet50()
    else:
        print('Unknown model')
        raise NotImplementedError

    model.load_state_dict(torch.load(pt_path))

    test_dataset = RetinopathyLoader('./dataset', 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model1 = model.to(device)
    _, acc, gt, pred = evaluate(model, test_loader)
    # plot_confusion_matrix(gt, pred, [0, 1, 2, 3, 4], os.path.join('cm', '{}.png'.format("ResNet18_bs16_lr_decay(with pretrained)")))
    print(f'Accuracy of ResNet18 without pretraining on testing set: {100*acc:.2f}%')

