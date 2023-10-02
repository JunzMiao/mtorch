from DLTK import mnn
from DLTK import mtorch
import numpy as np
from DLTK import moptim
from torch import nn
import torch

np.random.seed(1)
torch.manual_seed(1)

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
# ========================================================================= #
def accuracy(y_hat, train_labels):
    count_num = 0
    y_hat_data = y_hat.data

    for i in range(y_hat.shape[0]):
        if np.argmax(y_hat_data[i]) == train_labels[i]:
            count_num += 1

    return count_num


class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

metric = Accumulator(3)
# ========================================================================= #



batch_size = 256
lr = 0.9
num_epochs = 10

net = mnn.Sequential(
    mnn.Cov(1, 6, 5, 5, pad=2),
    mnn.Sigmoid(),
    mnn.Pool(2, 2, 'average'),
    mnn.Cov(6, 16, 5, 5),
    mnn.Sigmoid(),
    mnn.Pool(2, 2, 'average'),
    mnn.Flatten(),
    mnn.Linear(16 * 5 * 5, 120),
    mnn.Sigmoid(),
    mnn.Linear(120, 84),
    mnn.Sigmoid(),
    mnn.Linear(84, 10))


# ===========================调用pytorch的初始化方法====================================== #
pynet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10))

net.seqential[0].cov_kernel.data = pynet[0].state_dict()['weight'].numpy()
net.seqential[0].bias.data = pynet[0].state_dict()['bias'].numpy()


net.seqential[3].cov_kernel.data = pynet[3].state_dict()['weight'].numpy()
net.seqential[3].bias.data = pynet[3].state_dict()['bias'].numpy()

net.seqential[7].weight.data = pynet[7].state_dict()['weight'].numpy().T
net.seqential[7].bias.data = pynet[7].state_dict()['bias'].numpy()

net.seqential[9].weight.data = pynet[9].state_dict()['weight'].numpy().T
net.seqential[9].bias.data = pynet[9].state_dict()['bias'].numpy()

net.seqential[11].weight.data = pynet[11].state_dict()['weight'].numpy().T
net.seqential[11].bias.data = pynet[11].state_dict()['bias'].numpy()
# ===================================================================================== #


loss = mnn.CrossEntropyLoss()

trainer = moptim.SGD(net.get_parameters(), lr=lr)


for epoch in range(num_epochs):
    metric.reset()
    for X, y in data_iter(batch_size, train_images, train_labels):
        y_hat = net(X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])))
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(float(l.data) * len(y), accuracy(y_hat, y), len(y))
    print(metric[0] / metric[2], metric[1] / metric[2])








