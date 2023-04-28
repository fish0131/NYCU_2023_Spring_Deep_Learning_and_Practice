import os
import csv
import codecs
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dataloader import RetinopathyLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def downsample(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=stride, bias=False),
        nn.BatchNorm2d(out_ch))


class BasicBlock(nn.Module):
    '''
    input -> con2d(3x3) -> BN -> activation -> con2d(3x3) -> BN -> activation -> output
    Perform downsampling directly by convolutional layers that have a stride of 2
    '''
    def __init__(self, in_ch, out_ch, downsample_stride):
        super(BasicBlock, self).__init__()
        if downsample_stride is None:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = downsample(in_ch, out_ch, downsample_stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out
        
        
class Bottleneck(nn.Module):
    '''
    Use a stack of 3 layers instead of 2: the three layers are 1×1, 3×3, and 1×1 convolutions
    input -> con2d(1x1) -> BN -> activation -> con2d(3x3) -> BN -> activation -> con2d(1x1) -> BN -> activation -> output
    Perform downsampling directly by convolutional layers that have a stride of 2
    '''
    def __init__(self, in_ch, mid_ch, out_ch, downsample_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        if downsample_stride is None:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=(3, 3), stride=downsample_stride, padding=(1, 1), bias=False)
            self.downsample = downsample(in_ch, out_ch, downsample_stride)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, None),
            BasicBlock(64, 64, None)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, (2, 2)),
            BasicBlock(128, 128, None)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, (2, 2)),
            BasicBlock(256, 256, None)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, (2, 2)),
            BasicBlock(512, 512, None)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256, (1, 1)),
            Bottleneck(256, 64, 256, None),
            Bottleneck(256, 64, 256, None))
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, (2, 2)),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None))
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, (2, 2)),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None))
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, (2, 2)),
            Bottleneck(2048, 512, 2048, None),
            Bottleneck(2048, 512, 2048, None))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ',
                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Save accuracy Finished!")


def train(model, train_loader, test_loader, optimizer, loss_fn, epoch, lr_decay, cpt_dir, phase_dir):
    if lr_decay:
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)
    train_acc_values = [[]]
    test_acc_values = [[]]

    epoch_pbar = tqdm(range(1, epoch+1))

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader)
        for batch_idx, (data, labels) in enumerate(batch_pbar):
            data = data.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            batch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch}] [batch: {batch_idx+1:>5}/{len(train_loader)}] loss: {loss.item():.4f}')

        train_loss /= len(train_loader)

        if lr_decay:
            scheduler.step()

        _, train_acc, _, _ = evaluate(model, train_loader, loss_fn)
        test_avg_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn)

        epoch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')

        train_acc_values.append([epoch, train_acc])
        test_acc_values.append([epoch, test_acc])
    
    torch.save(model.state_dict(), os.path.join(cpt_dir, f'epoch{epoch}.pt'))

    data_write_csv(os.path.join(phase_dir, f'epoch{epoch}_train.csv'), train_acc_values)
    data_write_csv(os.path.join(phase_dir, f'epoch{epoch}_test.csv'), test_acc_values)


def evaluate(model, loader, loss_fn=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    gt, pred = [], []

    with torch.no_grad():
        batch_pbar = tqdm(loader)
        for batch_idx, (data, labels) in enumerate(batch_pbar):
            data = data.to(device=device)
            labels = labels.to(device=device)
            outputs = model(data)
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs==labels).sum().cpu().item()
            gt += labels.detach().cpu().numpy().tolist()
            pred += outputs.detach().cpu().numpy().tolist()

        test_loss /= len(loader)
        acc = correct / len(loader.dataset)

    if loss_fn is None:
        return None, acc, gt, pred
    else:
        return test_loss, acc, gt, pred



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=5e-04)
    parser.add_argument('--lr_decay', action='store_true', default=True)
    parser.add_argument('--pretrain', action='store_true', default=True)
    parser.add_argument('--cpt_dir', type=str, default='model_weights')
    parser.add_argument('--phase_dir', type=str, default='learning_phase')
    args = parser.parse_args()

    if args.model == 'resnet18':
        if args.pretrain:
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
        else:
            model = ResNet18()
    elif args.model == 'resnet50':
        if args.pretrain:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
        else:
            model = ResNet50()
    else:
        print('Unknown model')
        raise NotImplementedError

    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = RetinopathyLoader('./dataset', 'train')
    test_dataset = RetinopathyLoader('./dataset', 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    task_name = f'{args.model}_lr{lr}_{epoch}epoch_bs{batch_size}_lr_decay_pretrained'

    model = model.to(device)
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=epoch,
        lr_decay=args.lr_decay,
        cpt_dir=os.path.join(args.cpt_dir, task_name),
        phase_dir=os.path.join(args.phase_dir, task_name))