import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):

    pts = np.random.uniform(0, 1, (n, 2)) # low bound, high bound, size
    #print(pts)
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else:
            labels.append(1)
    #print(labels)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():

    inputs = []
    labels = []    

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i==0.5 :
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)


def show_result(x, y, pred_y, acc, fn=None):

    # plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].set_title('Ground truth', fontsize=18)
    axes[1].set_title('Predict result', fontsize=18)
    color = ['bo', 'ro']
    for i in range(x.shape[0]):
        axes[0].plot(x[i,0], x[i,1], color[int(y[i]==0)])
        axes[1].plot(x[i,0], x[i,1], color[int(pred_y[i]==0)])
        if pred_y[i] != y[i]:
            circle = plt.Circle((x[i,0], x[i,1]), 0.06, color='black', fill=False)
            axes[1].add_patch(circle)
    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')
    plt.suptitle(f'Acc: {acc:.2f}%', fontsize=16)
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('plots', fn), dpi=200)
    

def show_data(x, y, title, fn=None):

    plt.clf()
    color = ['bo', 'ro']
    for i in range(x.shape[0]):
        plt.plot(x[i,0], x[i,1], color[int(y[i]==0)])
    plt.title(title)
    plt.axis('square')
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('plots', fn), dpi=200)

def show_learning_curve(train_loss, fn=None):
    plt.clf()
    plt.plot(np.arange(len(train_loss))+1, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('train_loss_single', fn), dpi=200)
    np.save(os.path.join('train_loss', fn.replace('.png', '.npy')), np.array(train_loss))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1.0 - x ** 2

def relu(x):
    return np.maximum(0.0, x)

def derivative_relu(x):
    return np.heaviside(x, 0.0)


def mse_loss(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)

def derivative_mse_loss(y_pred, y_true):
    return 2 * (y_pred-y_true) / y_true.shape[0]


class Layer():

    def __init__(self, input_size, output_size, activation = 'sigmoid', optimizer = 'gd'):
        # combine the bias in w
        self.w = np.random.normal(0, 1, (input_size + 1, output_size))  # mean, standard deviation, size
        self.momentum = np.zeros((input_size + 1, output_size))
        self.sum_of_squares_of_gradients = np.zeros((input_size + 1, output_size))
        self.activation = activation
        self.optimizer = optimizer

    # dL / dw = (dz / dw) * (dL / dz)
    # dz / dw: forward
    # dL / dz: backward
    def forward(self, x):
        """
        x: input data for this layer
        return y: output computed by this layer
        """
        self.forward_gradient = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        # y = sigma(input * weights), sigma is the activation function
        if self.activation == 'sigmoid':
            self.y = sigmoid(np.matmul(self.forward_gradient, self.w))  
        elif self.activation == 'tanh':
            self.y = tanh(np.matmul(self.forward_gradient, self.w))
        elif self.activation == 'relu':
            self.y = relu(np.matmul(self.forward_gradient, self.w))
        # without activation function
        else:
            self.y = np.matmul(self.forward_gradient, self.w)

        
        return self.y

    def backward(self, derivative_loss):
        """
        derivative_loss: loss from the next layer
        return loss of this layer
        """
        # chain rule: dL / dz  = dy / dz * dL / dy
        # 1. dy / dz (by y = sigma(z) -> dy / dz = derivative_sigma(z))
        # 2. dL / dy
        if self.activation == 'sigmoid':
            self.backward_gradient = np.multiply(derivative_sigmoid(self.y), derivative_loss)
        elif self.activation == 'tanh':
            self.backward_gradient = np.multiply(derivative_tanh(self.y), derivative_loss)
        elif self.activation == 'relu':
            self.backward_gradient = np.multiply(derivative_relu(self.y), derivative_loss)
        else:
            # Without activation function
            self.backward_gradient = derivative_loss

        # loss of this layer
        return np.matmul(self.backward_gradient, self.w[:-1].T)

    def update(self, lr):
        # dL / dw: the gradient of the weight
        self.gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)

        if self.optimizer == 'gd':
            self.w -= lr * self.gradient
        elif self.optimizer == 'momentum':
            # momemtum = beta * the previous momentum - lr * (dL / dw)
            # beta = 0.9
            self.momentum = 0.9 * self.momentum - lr * self.gradient
            self.w = self.momentum
        elif self.optimizer == 'adagrad':
            # w = w - lr * (1 / sqrt(n + epsilon)) * (dL / dw) = lr * (dL / dw) / sqrt(n + epsilon)
            # n = sum of ((dL / dw) ** 2)
            # epsilon = 1e-08
            self.sum_of_squares_of_gradients += np.square(self.gradient)  # n
            self.w = -self.lr * self.gradient / np.sqrt(self.sum_of_squares_of_gradients + 1e-8) 
        return self.gradient


class Network():

    def __init__(self, num_features, lr, activation, optimizer):
        """
        num_features: number of features (input, hidden, hidden, output) -> array
        """
        self.lr = lr
        self.activation = activation
        self.layers = []

        for i in range(1, len(num_features)): 
            self.layers.append(Layer(num_features[i-1], num_features[i], self.activation))

    def forward(self, x):
        """
        x: input data
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, derivative_loss):
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)


def train(model, x, y, epochs, plot_fn):

    print('------------------')
    print('| Start training |')
    print('------------------')

    learning_epoch = []
    learning_loss = []

    for epoch in range(epochs):
        pred_y = model.forward(x)
        loss = mse_loss(pred_y, y)
        model.backward(derivative_mse_loss(pred_y, y))
        model.update()

        learning_epoch.append(epoch)
        learning_loss.append(loss)

        if epoch % 500 == 0:
            print(f'Epoch {epoch} loss : {loss}')
    
    # Plot learning curve
    # plt.figure()
    # plt.title('Learning curve', fontsize=18)
    # plt.plot(learning_epoch, learning_loss)
    # plt.savefig(os.path.join('plots/curve', plot_fn), dpi=200)

    # plt.show()

    return model, learning_loss


def test(model, x, y, plot_fn):
    print('-----------------')
    print('| Start testing |')
    print('-----------------')
    pred_y = model.forward(x)
    print(pred_y)
    loss = mse_loss(pred_y, y)
    pred_y_rounded = np.round(pred_y)
    pred_y_rounded[pred_y_rounded<0] = 0
    correct = np.sum(pred_y_rounded==y)
    acc = 100 * correct / len(y)
    show_result(x, y, pred_y_rounded, acc, plot_fn)
    print(f'Testing loss: {loss:.8f}')
    print(f'Acc: {correct}/{len(y)} ({acc:.2f}%)')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1.0)     # sigmoid, tanh->1.0, None->0.001
    parser.add_argument('--epochs', type=int, default=10000)  # sigmoid, tanh->10000, None->50000
    parser.add_argument('--num_features', type=str, default='2-4-4-1')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--optimizer', type=str, default='gd')
    args = parser.parse_args()
    # print(args)

    lr = args.lr
    epochs = args.epochs
    num_features = args.num_features.split('-')
    num_features = [int(n) for n in num_features]
    activation = args.activation
    optimizer = args.optimizer

    train_x, train_y = generate_linear(n=100)
    show_data(train_x, train_y, title='Linear', fn=None)
    test_x, test_y = generate_linear(n=100)
    model_linear = Network(num_features, lr, activation, optimizer)
    model_linear, train_loss = train(model_linear, train_x, train_y, epochs, plot_fn=None)
    test(model_linear, test_x, test_y, plot_fn=None)
    # show_learning_curve(train_loss, fn='linear_lc.png')

    print("===================================================")

    train_x, train_y = generate_XOR_easy()
    show_data(train_x, train_y, title='XOR', fn=None)
    test_x, test_y = generate_XOR_easy()
    model_xor = Network(num_features, lr, activation, optimizer)
    model_xor, train_loss = train(model_xor, train_x, train_y, epochs, plot_fn=None)
    test(model_xor, test_x, test_y, plot_fn=None)
    # show_learning_curve(train_loss, fn='xor_lc.png')
