import numpy as np
from DLTK import mnn
from DLTK import moptim


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

net = mnn.Sequential(
    mnn.Flatten(), 
    mnn.Linear(784, 256), 
    mnn.ReLU(), 
    mnn.Linear(256, 128),
    mnn.ReLU(), 
    mnn.Linear(128, 10))

loss = mnn.CrossEntropyLoss()


trainer = moptim.SGD(net.get_parameters(), lr=0.05)

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


num_epochs = 20

def accuracy(y_hat, train_labels):
    count_num = 0
    y_hat_data = y_hat.data

    for i in range(y_hat.shape[0]):
        if np.argmax(y_hat_data[i]) == train_labels[i]:
            count_num += 1

    return count_num

for epoch in range(num_epochs):
    metric.reset()
    for X, y in data_iter(batch_size, train_images, train_labels):
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(float(l.data) * len(y), accuracy(y_hat, y), len(y))
    print(metric[0] / metric[2], metric[1] / metric[2])
    
