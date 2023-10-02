from DLTK.mtensor import MTensor
from DLTK.moperate import *


def tensor(data, requires_grad=False):

    return MTensor(data, requires_grad=requires_grad)


def exp(tensor):
    
    return Exp()(tensor)


def log(tensor):
    
    return Log()(tensor)


def sin(tensor):
    
    return Sin()(tensor)


def cos(tensor):
    
    return Cos()(tensor)


def matmul(tensor_1, tensor_2):

    return MatMul()(tensor_1, tensor_2)


def sum(tensor):

    return SUM()(tensor)


def mean(tensor):

    return MEAN()(tensor)


def clip(tensor, min=None, max=None):

    return Clip()(tensor, min, max)


def flatten(tensor):

    return Flatten_OP()(tensor)


def cov_2d(tensor, cov_kernel, bias, stride, pad):

    return Cov_2D()(tensor, cov_kernel, bias, stride, pad)


def pool(tensor, pool_size, stride, mode):
    if mode == 'max':
        mode = 1
    elif mode == 'average':
        mode = 2
    
    return Pool_OP()(tensor, pool_size, stride, mode)
