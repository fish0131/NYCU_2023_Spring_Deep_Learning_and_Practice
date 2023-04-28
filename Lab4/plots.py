import csv
import numpy as np
import matplotlib.pyplot as plt


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



# plt accuracy figure
plt.figure()

plt.plot([1, 10.5], [85, 85], color='black', linestyle='dotted', lw=1)
plt.gca().text(10.75, 85, '85%', color='black')
plt.plot([1, 10.5], [82, 82], color='black', linestyle='dotted', lw=1)
plt.gca().text(10.75, 82, '82%', color='black')
plt.plot([1, 10.5], [75, 75], color='black', linestyle='dotted', lw=1)
plt.gca().text(10.75, 75, '75%', color='black')


x1, y1 = readcsv("./learning_phase/resnet18_lr0.001_10epoch_bs16_weight_decay/epoch10_train.csv")
plt.plot(x1, y1, color = 'green', linewidth = '1', label = 'Train(w/o pretraining)')

x2, y2 = readcsv("./learning_phase/resnet18_lr0.001_10epoch_bs16_weight_decay/epoch10_test.csv")
plt.plot(x2, y2, color = 'blue', linestyle='dashed', linewidth = '1', label = 'Test(w/o pretraining)')

x3, y3 = readcsv("./learning_phase/resnet18_lr0.001_10epoch_bs16_weight_decay_pretrained/epoch10_train.csv")
plt.plot(x3, y3, color = 'red', linewidth = '1', label = 'Train(with pretraining)')

x4, y4 = readcsv("./learning_phase/resnet18_lr0.001_10epoch_bs16_weight_decay_pretrained/epoch10_test.csv")
plt.plot(x4, y4, color = 'orange', linestyle='dashed', linewidth = '1', label = 'Test(with pretraining)')

plt.xlim(0.5, 11.5)
plt.ylim(70, 92)
plt.xlabel('Epochs')
plt.ylabel('Accuracy(%)')

plt.title('Result Comparison(ResNe18)')
plt.legend(loc='upper left')

plt.savefig('./plots/result_comparison(ResNet18)_bs16.png')
plt.show()