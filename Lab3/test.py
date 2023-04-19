from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from dataloader import read_bci_data
from train import BCIDataset, EEGNet, DeepConvNet, evaluate


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eegmodel', type=str, default='EEGNet')
    parser.add_argument('--deepmodel', type=str, default='DeepConvNet')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    pt_path = './model_weights/EEGNet_relu_lr0.001_450epoch_bs64_lr_decay_dropout0.5/epoch450.pt'
    ## leaky relu
    # EEGNet_lrelu_lr0.001_450epoch_bs64_lr_decay_dropout0.5
    ## elu
    # EEGNet_elu_lr0.001_450epoch_bs64_lr_decay_dropout0.5

    model = vars()[args.eegmodel](args.activation)
    model.load_state_dict(torch.load(pt_path))

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model.to(device)
    _, acc = evaluate(model, test_loader)
    print(f'Accuracy of EEGNet using relu: {100*acc:.2f}%')



    pt_path = './model_weights/DeepConvNet_relu_lr0.01_400epoch_bs32_lr_decay/epoch400.pt'
    ## leaky relu
    # DeepConvNet_lrelu_lr0.01_500epoch_bs32
    ## elu
    # DeepConvNet_elu_lr0.01_300epoch_bs64_lr_decay

    model = vars()[args.deepmodel](args.activation)
    model.load_state_dict(torch.load(pt_path))

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model.to(device)
    _, acc = evaluate(model, test_loader)
    print(f'Accuracy of DeepConvNet using relu: {100*acc:.2f}%')