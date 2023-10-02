from turtle import forward
from DLTK import mtorch
import numpy as np


class Module:
    def __call__(self, *args):

        return self.forward(*args)
    
    def get_parameters(self):

        return []
    

class Sequential(Module):
    def __init__(self, *args):
        self.seqential = args
    
    def forward(self, x):
        for s in self.seqential:
            x = s(x)
        
        return x
    
    def get_parameters(self):
        parameters_array = []

        for s in self.seqential:
            parameters_array += s.get_parameters()
        
        return parameters_array


class Linear(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = mtorch.tensor(np.random.randn(in_channels, out_channels) * 0.01, requires_grad=True)
        self.bias = None

        if bias:
            self.bias = mtorch.tensor(np.zeros(out_channels), requires_grad=True)
    
    def forward(self, x):
        if self.bias:

            return mtorch.matmul(x, self.weight) + self.bias
        
        else:

            return mtorch.matmul(x, self.weight)

    def get_parameters(self):
        if self.bias:
        
            return [self.weight, self.bias]

        else:

            return [self.weight]


class Flatten(Module):
    def forward(self, x):
        
        return mtorch.flatten(x)


class ReLU(Module):
    def forward(self, x):

        return mtorch.clip(x, 0.0)


class Sigmoid(Module):
    def forward(self, x):

        return 1.0 / (1.0 + mtorch.exp(-x))


class Cov(Module):
    def __init__(self, in_channels, out_channels, cov_kernel_H, cov_kernel_W, stride=1, pad=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pad = pad
        self.cov_kernel = mtorch.tensor(np.random.randn(out_channels, in_channels, cov_kernel_H, cov_kernel_W) * 0.01, requires_grad=True)
        self.bias = mtorch.tensor(np.zeros(out_channels), requires_grad=bias)
    
    def forward(self, x):

        return mtorch.cov_2d(x, self.cov_kernel, self.bias, self.stride, self.pad)

    def get_parameters(self):

        return [self.cov_kernel, self.bias]


class Pool(Module):
    def __init__(self, pool_size, stride, mode):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
    
    def forward(self, x):

        return mtorch.pool(x, self.pool_size, self.stride, self.mode)


class MSELoss:
    def build_graph(self, y_hat, y):

        return mtorch.mean((((y_hat - y) ** 2) / 2))
    
    def compute_loss(self, y_hat, y):
        
        return np.mean(((y_hat.data - y) ** 2) / 2)


    def __call__(self, y_hat, y, requires_grad=True):
        if requires_grad:

            return self.build_graph(y_hat, y)

        else:

            return self.compute_loss(y_hat, y)


class CrossEntropyLoss:
    def build_graph(self, y_hat, y):
        y_hat_softmax = mtorch.exp(y_hat)
        y_hat_sum = mtorch.matmul(y_hat_softmax, np.ones((10, 1))) * np.ones(y_hat.shape)
        y_hat_softmax = y_hat_softmax / y_hat_sum

        y_onehot = np.zeros((len(y), 10))
        
        for i in range(len(y)):
            y_onehot[i][int(y[i])] = 1

        return mtorch.mean(-mtorch.log(mtorch.matmul(y_hat_softmax * y_onehot, np.ones((10, 1)))))
    
    def compute_loss(self, y_hat, y):
        y_hat_softmax = np.exp(y_hat.data)
        y_hat_sum = np.dot(y_hat_softmax, np.ones((10, 1))) * np.ones(y_hat.shape)
        y_hat_softmax = y_hat_softmax / y_hat_sum

        y_onehot = np.zeros((len(y), 10))

        for i in range(len(y)):
            y_onehot[i][int(y[i])] = 1

        return -(np.log(np.dot(y_hat_softmax * y_onehot, np.ones((10, 1))))).mean()


    def __call__(self, y_hat, y, requires_grad=True):
        if requires_grad:

            return self.build_graph(y_hat, y)
        else:

            return self.compute_loss(y_hat, y)
