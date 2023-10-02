import numpy as np

class MTensor:
    def __init__(self, data, sub_nodes=None, operator=None, requires_grad=False):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.sub_nodes = sub_nodes
        self.operator = operator
        self.requires_grad = requires_grad
        self.parent_node_num = 0
        self.backward_num = 0

        if self.sub_nodes:
            self.leaf = False
            for sub in sub_nodes:
                if isinstance(sub, MTensor):
                    sub.parent_node_num += 1
        else:
            self.leaf = True
        
        if self.requires_grad:
            self.grad = np.zeros(self.shape)
    
    def __neg__(self):

        return Neg()(self)
    
    def __add__(self, another):

        return Add()(self, another)
    
    def __radd__(self, another):

        return Add()(self, another)
    
    def __sub__(self, another):

        return Add()(self, -another)
    
    def __rsub__(self, another):

        return Add()(another, -self)
    
    def __mul__(self, another):

        return Mul()(self, another)
    
    def __rmul__(self, another):

        return Mul()(self, another)
    
    def __truediv__(self, another):

        return Div()(self, another)
    
    def __rtruediv__(self, another):

        return Div()(another, self)
    
    def __pow__(self, another):

        return Pow()(self, another)
    
    def __rpow__(self, another):

        return Pow()(another, self)
    
    def backward(self, top=False):
        if not top:
            self.grad = np.ones(self.shape)
        self.backward_num += 1
        if not self.requires_grad:

            return 
        if self.leaf:

            return 
        if self.backward_num < self.parent_node_num:

            return
        if isinstance(self.operator, Operate):
            self.operator.backward(self)
        
        for sub_node in self.sub_nodes:
            sub_node.backward(top=True)
    
    def grad_zero(self):
        self.grad = 0
        self.backward_num = 0
        self.parent_node_num = 0

        return


def to_Tensor(other):
    if not isinstance(other, MTensor):

        return MTensor(other, requires_grad=False)
    
    return other


def compute_grad(tensor_node, grad):
    if not tensor_node.requires_grad:

        return

    for _ in range(grad.ndim - tensor_node.data.ndim):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(tensor_node.data.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    tensor_node.grad += grad

    return


def judge_grad(args):
    for node in args:
        if isinstance(node, MTensor):
            if node.requires_grad:
                return True
    
    return False


class Operate:
    def forward(self, args):
        pass

    def __call__(self, *args):
        sub_nodes_list = [ele for ele in args]
        compute_result = self.forward(sub_nodes_list)
        requires_grad = judge_grad(sub_nodes_list)
        
        return MTensor(compute_result, sub_nodes=sub_nodes_list, operator=self, requires_grad=requires_grad)

    def backward(self, tensor_node):
        pass


class Neg(Operate):
    def forward(self, args):
        sub_node_1 = args[0]
        compute_result = -sub_node_1.data

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]

        grad_1 = -tensor_node.grad

        compute_grad(sub_node_1, grad_1)

        return


class Add(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        sub_node_2 = to_Tensor(args[1])
        args[0] = sub_node_1
        args[1] = sub_node_2
        compute_result = sub_node_1.data + sub_node_2.data

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        sub_node_2 = tensor_node.sub_nodes[1]

        grad_1 = tensor_node.grad
        grad_2 = tensor_node.grad
        
        compute_grad(sub_node_1, grad_1)
        compute_grad(sub_node_2, grad_2)
        
        return


class Mul(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        sub_node_2 = to_Tensor(args[1])
        args[0] = sub_node_1
        args[1] = sub_node_2
        compute_result = sub_node_1.data * sub_node_2.data

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        sub_node_2 = tensor_node.sub_nodes[1]

        grad_1 = sub_node_2.data * tensor_node.grad
        grad_2 = sub_node_1.data * tensor_node.grad
        
        compute_grad(sub_node_1, grad_1)
        compute_grad(sub_node_2, grad_2)
        
        return


class Div(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        sub_node_2 = to_Tensor(args[1])
        args[0] = sub_node_1
        args[1] = sub_node_2
        compute_result = sub_node_1.data / sub_node_2.data

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        sub_node_2 = tensor_node.sub_nodes[1]

        grad_1 = tensor_node.grad / sub_node_2.data
        grad_2 = -tensor_node.grad * sub_node_1.data / sub_node_2.data ** 2
        
        compute_grad(sub_node_1, grad_1)
        compute_grad(sub_node_2, grad_2)
        
        return 


class Pow(Operate):
    def forward(self, args):
        sub_node_1 = to_Tensor(args[0])
        sub_node_2 = to_Tensor(args[1])
        args[0] = sub_node_1
        args[1] = sub_node_2
        compute_result = sub_node_1.data ** sub_node_2.data

        return compute_result
    
    def backward(self, tensor_node):
        sub_node_1 = tensor_node.sub_nodes[0]
        sub_node_2 = tensor_node.sub_nodes[1]

        grad_1 = tensor_node.grad * sub_node_2.data * sub_node_1.data ** (sub_node_2.data - 1)
        grad_2 = -tensor_node.grad * np.log(sub_node_1.data) * sub_node_1.data ** sub_node_2.data
        
        compute_grad(sub_node_1, grad_1)
        compute_grad(sub_node_2, grad_2)
        
        return 
