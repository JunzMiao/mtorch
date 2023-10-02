import numpy as np
from DLTK import mnn
from DLTK import moptim


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


w = np.random.normal(0, 0.01, 2)
batch_size = 10
net = mnn.Sequential(mnn.Linear(2, 1))
net.seqential[0].weight.data[0][0] = w[0]
net.seqential[0].weight.data[1][0] = w[1]

loss = mnn.MSELoss()

trainer = moptim.SGD(net.get_parameters(), lr=0.03)


num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels, requires_grad=False)
    print(f'epoch {epoch + 1}, loss {l:f}')


