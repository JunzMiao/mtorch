from DLTK.mtensor import Operate, compute_grad, to_Tensor
import numpy as np


class Exp(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.exp(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        compute_grad(sub_node_1, tensor_node.grad * tensor_node.data)

        return


class Log(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.log(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        compute_grad(sub_node_1, tensor_node.grad / sub_node_1.data)

        return


class Sin(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.sin(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        compute_grad(sub_node_1, tensor_node.grad * np.cos(sub_node_1.data))

        return 


class Cos(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.cos(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        compute_grad(sub_node_1, tensor_node.grad * np.sin(sub_node_1.data))

        return 

class MatMul(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        sub_node_2 = to_Tensor(args[1])
        args[0] = sub_node_1
        args[1] = sub_node_2
        compute_result = np.dot(sub_node_1.data, sub_node_2.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        sub_node_2 = tensor_node.sub_nodes[1]
        
        grad_1 = np.dot(tensor_node.grad, sub_node_2.data.T)
        grad_2 = np.dot(sub_node_1.data.T, tensor_node.grad)
        
        compute_grad(sub_node_1, grad_1)
        compute_grad(sub_node_2, grad_2)
        
        return 


class SUM(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.sum(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        compute_grad(sub_node_1, tensor_node.grad * np.ones(sub_node_1.shape))
        
        return 


class MEAN(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        compute_result = np.mean(sub_node_1.data)

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        ele_num = 1
        for _, dim in enumerate(sub_node_1.shape):
            ele_num *= dim
        
        compute_grad(sub_node_1, tensor_node.grad * np.ones(sub_node_1.shape) / ele_num)
        
        return 


class Clip(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        min = to_Tensor(args[1])
        max = to_Tensor(args[2])
        args[0] = sub_node_1
        args[1] = min
        args[2] = max
        if min.data == None:
            compute_result = sub_node_1.data.clip(None, max.data)
        if max.data == None:
            compute_result = sub_node_1.data.clip(min.data, None)
        
        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        min = tensor_node.sub_nodes[1]
        max = tensor_node.sub_nodes[2]

        mask = np.ones(tensor_node.shape, dtype=bool)
        if min.data != None:
            mask &= sub_node_1.data >= min.data
        if max.data != None:
            mask &= sub_node_1.data <= max.data

        grad = tensor_node.grad * mask
        
        compute_grad(sub_node_1, grad)
        
        return 


class Flatten_OP(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        args[0] = sub_node_1
        all_ndim = 1
        for dim in sub_node_1.data[0].shape:
            all_ndim *= dim
        num = sub_node_1.shape[0]
        
        return sub_node_1.data.reshape((num, all_ndim))
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        
        grad = tensor_node.grad.reshape(sub_node_1.shape)

        compute_grad(sub_node_1, grad)

        return


class Cov_2D(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        cov_kernel = to_Tensor(args[1])
        bias = to_Tensor(args[2])
        stride = to_Tensor(args[3])
        pad = to_Tensor(args[4])
        args[0] = sub_node_1
        args[1] = cov_kernel
        args[2] = bias
        args[3] = stride
        args[4] = pad

        out_channel, input_channel, filter_H, filter_W = cov_kernel.shape
        N, C, H, W = sub_node_1.shape
        out_h = int(1 + (H + 2 * pad.data - filter_H) / stride.data)
        out_w = int(1 + (W + 2 * pad.data - filter_W) / stride.data)

        col = im2col(sub_node_1.data, filter_H, filter_W, stride=stride.data, pad=pad.data)
        col_W = cov_kernel.data.reshape(out_channel, -1).T
        compute_result = np.dot(col, col_W) + bias.data

        compute_result = compute_result.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return compute_result
    
    def backward(self, tensor_node):
        FN, C, FH, FW = tensor_node.sub_nodes[1].shape
        grad_out = tensor_node.grad.transpose(0,2,3,1).reshape(-1, FN)

        col = im2col(tensor_node.sub_nodes[0].data, FH, FW, stride=tensor_node.sub_nodes[3].data, pad=tensor_node.sub_nodes[4].data)

        grad_bias = np.sum(grad_out, axis=0)
        grad_cov_kernel = np.dot(col.T, grad_out)
        grad_cov_kernel = grad_cov_kernel.transpose(1, 0).reshape(FN, C, FH, FW)

        grad_sub_node_1 = np.dot(grad_out, tensor_node.sub_nodes[1].data.reshape(FN, -1))
        grad_sub_node_1 = col2im(grad_sub_node_1, tensor_node.sub_nodes[0].shape, FH, FW, tensor_node.sub_nodes[3].data, tensor_node.sub_nodes[4].data)

        compute_grad(tensor_node.sub_nodes[0], grad_sub_node_1)
        compute_grad(tensor_node.sub_nodes[1], grad_cov_kernel)
        compute_grad(tensor_node.sub_nodes[2], grad_bias)
        
        return 
        

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y: y_max: stride, x: x_max: stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y: y_max:stride, x: x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad: H + pad, pad: W + pad]

class Pool_OP(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        pool_size = to_Tensor(args[1])
        stride = to_Tensor(args[2])
        mode = to_Tensor(args[3])
        args[0] = sub_node_1
        args[1] = pool_size
        args[2] = stride
        args[3] = mode

        batch_size, C_prev, H_prev, W_prev = sub_node_1.shape

        H = int(1 + (H_prev - pool_size.data) / stride.data)
        W = int(1 + (W_prev - pool_size.data) / stride.data)


        col = im2col(sub_node_1.data, pool_size.data, pool_size.data, stride.data, 0)
        col = col.reshape(-1, pool_size.data ** 2)
        if mode.data == 1:
            out = np.max(col, axis=1)
            args.append(to_Tensor(np.argmax(col, axis=1)))
        elif mode.data == 2:
            out = np.average(col, axis=1)
        compute_result = out.reshape(batch_size, H, W, C_prev).transpose(0, 3, 1, 2)
        
        return compute_result
    
    def backward(self, tensor_node):
        tensor_node_grad = tensor_node.grad.transpose(0, 2, 3, 1)
        
        pool_size = tensor_node.sub_nodes[1].data ** 2

        if tensor_node.sub_nodes[3].data == 1:
            dmax = np.zeros((tensor_node_grad.size, pool_size))

            dmax[np.arange(tensor_node.sub_nodes[4].data.size), tensor_node.sub_nodes[4].data.flatten()] = tensor_node_grad.flatten()
        elif tensor_node.sub_nodes[3].data == 2:
            dmax = (np.ones((tensor_node_grad.size, pool_size)) / pool_size) * tensor_node_grad.flatten().reshape((-1, 1))

        dmax = dmax.reshape(tensor_node_grad.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        grad = col2im(dcol, tensor_node.sub_nodes[0].data.shape, tensor_node.sub_nodes[1].data, tensor_node.sub_nodes[1].data, tensor_node.sub_nodes[2].data, 0)

        compute_grad(tensor_node.sub_nodes[0], grad)

        return 
                    

