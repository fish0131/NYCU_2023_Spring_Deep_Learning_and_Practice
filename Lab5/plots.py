import matplotlib.pyplot as plt
import csv

epoch = []
kl_weight = []
tfr = []
loss = []
mse_loss = []
kld = []
psnr = []

expt = 'kl_cyclical=True-rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000-test02/'
log_dir = './logs/lp/{}'.format(expt)

with open(log_dir + 'epoch_curve_plotting_data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines, None)
    for row in lines:
        epoch.append(int(row[0]))
        kl_weight.append(float(row[1]))
        tfr.append(float(row[2]))
        loss.append(float(row[3]))
        mse_loss.append(float(row[4]))
        kld.append(float(row[5]))
        psnr.append(float(row[6]))

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(epoch, kl_weight, color = 'g', linestyle = 'dashed', label = "KL Weight")
plt.xlabel("Epoch")
plt.ylabel("KL Weight")
plt.title("KL Weight")
plt.legend(loc='lower right')

plt.subplot(2, 2, 2)
plt.plot(epoch, tfr, color = 'y', linestyle = 'dashed', label = "TFR")
plt.xlabel("Epoch")
plt.ylabel("TFR")
plt.title("Teacher Forcing Ratio")
plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
plt.plot(epoch, loss, color='royalblue', label = "Loss")
plt.plot(epoch, mse_loss, color='tomato', label = "MSE Loss")
plt.plot(epoch, kld, color='lightseagreen', label = "KLD Loss")
plt.ylim(0.00, 0.10)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.scatter(epoch, psnr, color = 'r', label = "PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("PSNR")
plt.legend(loc='lower right')

#plt.savefig('./plots/monotonic_test01.png', dpi=300)
plt.show()