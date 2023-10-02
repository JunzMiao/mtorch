import numpy as np
import torch
from torch import nn

def read_Fashion_MNIST(data_path_list):
    with open(data_path_list[0], 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape((60000, 28, 28))
    with open(data_path_list[1], 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape((10000, 28, 28))
    with open(data_path_list[2], 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(data_path_list[3], 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return train_images, test_images, train_labels, test_labels

fashion_minst_path = ['./dataset/train-images-idx3-ubyte', './dataset/t10k-images-idx3-ubyte', './dataset/train-labels-idx1-ubyte', './dataset/t10k-labels-idx1-ubyte']

train_images, test_images, train_labels, test_labels = read_Fashion_MNIST(fashion_minst_path)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]

        yield features[batch_indices] / 255, labels[batch_indices]


batch_size = 256
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)

    return X_exp / partition


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

def evaluate_accuracy(net, dataset, labels):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter(batch_size, dataset, labels):
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

lr = 0.1
num_epochs = 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

metric = Accumulator(3)

for epoch in range(num_epochs):
    metric.reset()
    for X, y in data_iter(batch_size, train_images, train_labels):
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
        y = torch.tensor(y, requires_grad=False, dtype=torch.float32)
        y_hat = net(X)
        l = loss(y_hat ,y.long())
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    print(metric[0] / metric[2], metric[1] / metric[2])
