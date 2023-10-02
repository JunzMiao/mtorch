import numpy as np
import torch


np.random.seed(1)


def synthetic_data(w, b, num_examples):
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)

    return X, y.reshape((-1, 1))


true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]

        yield features[batch_indices], labels[batch_indices]


batch_size = 10
w = torch.tensor(np.random.normal(0, 0.01, 2).reshape((2, 1)), requires_grad=True, dtype=torch.float32)
b = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)


def linreg(X, w, b):

    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
        y = torch.tensor(y, requires_grad=False, dtype=torch.float32)
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(torch.tensor(features, requires_grad=False, dtype=torch.float32), w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
