import csv
import matplotlib.pyplot as plt

import numpy as np
from dataloader import read_bci_data


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=' ')
    next(plots)
    x = []
    y = []
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]) * 100)
    return x ,y


plot_result = True
plot_data = False

if plot_result:
    # plt accuracy figure
    plt.figure()

    plt.plot([1, 500], [87, 87], color='black', lw=1)
    plt.gca().text(505.5, 87, '87%', color='black')
    plt.plot([1, 500], [85, 85], color='black', lw=1)
    plt.gca().text(505.5, 85, '85%', color='black')


    x1, y1 = readcsv("./learning_phase/DeepConvNet_elu_lr0.01_500epoch_bs64_lr_decay/epoch500_train.csv")
    plt.plot(x1, y1, color = 'purple', linewidth = '1', label = 'elu_train')

    x2, y2 = readcsv("./learning_phase/DeepConvNet_elu_lr0.01_500epoch_bs64_lr_decay/epoch500_test.csv")
    plt.plot(x2, y2, color = 'brown', linewidth = '1', label = 'elu_test')

    x3, y3 = readcsv("./learning_phase/DeepConvNet_relu_lr0.01_500epoch_bs32_lr_decay/epoch500_train.csv")
    plt.plot(x3, y3, color = 'orange', linewidth = '1', label = 'relu_train')

    x4, y4 = readcsv("./learning_phase/DeepConvNet_relu_lr0.01_500epoch_bs32_lr_decay/epoch500_test.csv")
    plt.plot(x4, y4, color = 'blue', linewidth = '1', label = 'relu_test')

    x5, y5 = readcsv("./learning_phase/DeepConvNet_lrelu_lr0.01_500epoch_bs32/epoch500_train.csv")
    plt.plot(x5, y5, color = 'green', linewidth = '1', label = 'leaky_relu_train')

    x6, y6 = readcsv("./learning_phase/DeepConvNet_lrelu_lr0.01_500epoch_bs32/epoch500_test.csv")
    plt.plot(x6, y6, color = 'red', linewidth = '1', label = 'leaky_relu_test')

    plt.ylim(60, 105)
    plt.xlim(0, 550)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.title('Activation function comparison(DeepConvNet)')
    plt.legend()

    plt.savefig('./plots/activation_function_comparison(DeepConvNet).png')
    plt.show()



if plot_data:

    train_data, train_label, test_data, test_label = read_bci_data()
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    axs[0].plot(np.arange(train_data.shape[-1]), train_data[0,0,0,...], c='dodgerblue', lw=1.5)
    axs[0].set_title('Channel 1')
    axs[1].plot(np.arange(train_data.shape[-1]), train_data[0,0,1,...], c='dodgerblue', lw=1.5)
    axs[1].set_title('Channel 2')
    plt.savefig('./plots/data.png', dpi=300)
    plt.show()