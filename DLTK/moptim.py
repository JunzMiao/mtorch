class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad_zero()
    
    def step(self):
        for p in self.parameters:
            p.data = p.data - self.lr * p.grad
