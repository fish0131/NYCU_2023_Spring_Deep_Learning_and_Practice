import os
import csv
import codecs
import math
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from dataloader import read_bci_data


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


class EEGNet(nn.Module):
    def __init__(self, activation='relu'):
        super(EEGNet, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            print('Unknown activation function')
            raise NotImplementedError

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.5))

        self.separableconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.5))

        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseconv(out)
        out = self.separableconv(out)
        out = self.classify(out.flatten(start_dim=1))
        return out


class DeepConvNet(nn.Module):
    def __init__(self, activation='relu'):
        super(DeepConvNet, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            print('Unknown activation function')
            raise NotImplementedError

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.classify = nn.Sequential(nn.Linear(in_features=8600, out_features=2, bias=True))

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classify(out.flatten(start_dim=1))
        return out


class ShallowConvNet(nn.Module):
    # Reference: https://github.com/CECNL/ExBrainable/blob/main/ExBrainable/models.py
    def __init__(self, activation='relu'):
        super(ShallowConvNet, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            print('Unknown activation function')
            raise NotImplementedError

        self.tp= 750
        self.ch= 2
        self.sf= 125 
        self.n_class=2
        self.octsf= int(math.ceil(self.sf*0.1))
        self.tpconv1= int(self.tp- self.octsf+1)
        self.apstride= int(math.ceil(self.octsf/2))
        
        # kernel size=(ceil(sf*0.1)
        self.conv1 = nn.Conv2d(1, 40, (1, self.octsf), bias=False)
        #(n_ch,1)
        self.conv2 = nn.Conv2d(40, 40, (self.ch, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        self.acti = self.activation
        self.AvgPool1 = nn.AvgPool2d((1, int(self.apstride*5)), stride=(1,self.apstride ))
       
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(int(40*math.ceil((self.tpconv1-self.apstride*5) / self.apstride)), 
                                    self.n_class, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x**2
        x = self.acti(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, int(40*math.ceil((self.tpconv1-self.apstride*5 )/ self.apstride))) #40*74
        x = self.classifier(x)
        return x


def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ',
                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Save accuracy Finished!")


def train(model, train_loader, test_loader, optimizer, loss_fn, epoch, lr_decay, logger, cpt_dir, phase_dir):
    if lr_decay:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
    
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)
    train_acc_values = [[]]
    test_acc_values = [[]]

    epoch_bar = tqdm(range(1, epoch+1))

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if lr_decay:
            scheduler.step()

        _, train_acc = evaluate(model, train_loader, loss_fn)
        test_avg_loss, test_acc = evaluate(model, test_loader, loss_fn)

        logger.add_scalar('train/loss', train_loss, epoch)
        logger.add_scalar('train/acc', train_acc, epoch)
        logger.add_scalar('test/loss', test_avg_loss, epoch)
        logger.add_scalar('test/acc', test_acc, epoch)

        epoch_bar.set_description(f'[epoch:{epoch:>4}/{epoch}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')

        train_acc_values.append([epoch, train_acc])
        test_acc_values.append([epoch, test_acc])
    
    torch.save(model.state_dict(), os.path.join(cpt_dir, f'epoch{epoch}.pt'))

    data_write_csv(os.path.join(phase_dir, f'epoch{epoch}_train.csv'), train_acc_values)
    data_write_csv(os.path.join(phase_dir, f'epoch{epoch}_test.csv'), test_acc_values)


def evaluate(model, loader, loss_fn=None):

    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(device=device)
            labels = labels.to(device=device)
            outputs = model(data)
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs==labels).sum().cpu().item()
        test_loss /= len(loader)
        acc = correct / len(loader.dataset)

    if loss_fn is None:
        return None, acc
    else:
        return test_loss, acc


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='ShallowConvNet')
    parser.add_argument('--activation', type=str, default='elu')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_decay', action='store_true', default=True)
    parser.add_argument('--cpt_dir', type=str, default='model_weights')
    parser.add_argument('--phase_dir', type=str, default='learning_phase')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    model = vars()[args.model](args.activation)
    activation = args.activation
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    task_name = f'{args.model}_{activation}_lr{lr}_{epoch}epoch_bs{batch_size}_lr_decay'
    logger = SummaryWriter(os.path.join(args.log_dir, task_name))

    model.to(device)
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=epoch,
        lr_decay=args.lr_decay,
        logger=logger,
        cpt_dir=os.path.join(args.cpt_dir, task_name),
        phase_dir=os.path.join(args.phase_dir, task_name))