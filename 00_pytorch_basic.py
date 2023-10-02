import torch

# 生成一个向量，该向量包含从0开始的前12个整数
x = torch.arange(12)
# 在torch中无论向量、矩阵还是更高维的数组统称为为张量，张量中的每个值称为元素
print(x)
# 通过张量的shape属性可以查看其形状，即每一维的长度
print(x.shape)
# 通过张量的numel()方法可以查看其供含有多少个元素，即其形状的所有元素乘积
print(x.numel())
# 通过张量的reshape()方法可以改变其形状，其首先按照最后一个维度开始排，排满就向前一个维度进1，然后再从新开始排，依次类推，reshape之后会创建新的内存空间，即reshape为非原地操作
X = x.reshape(3, 4)
print(X)
print(id(X) == id(x))
# torch创建全0的张量，参数指定张量的形状
print(torch.zeros((2, 3, 4)))
# torch创建全1的张量，参数指定张量的形状
print(torch.ones((2, 3, 4)))
# torch创建一个张量，元素值是从某个特定的概率分布中采样而来，randn()方法生成的张量中每个元素是从均值为0，标准差为1的标准高斯分布中随机采样
print(torch.randn((3, 4)))
# 通过列表或嵌套列表来创建张量
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
# 对于常见的算术运算+、-、*、/、**，张量间的这些运算为按元素操作，这些方法也均为非原地操作
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
# 更多按元素计算的方法
print(torch.exp(x))
# 张量的连结，dim指定按照哪个维度进行连结，连结原理是，每个元素非指定的那个维度坐标不变，而指定的维度的坐标值为自身的维度加上前面的该维度的总数
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 3, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
# ==逻辑比较会生成与原张量形状相同的bool值张量，通过比较张量各位置上的元素值是否相同
print(X == Y)
# 对张量中的所有元素求和，产生一个单元素张量，单元素的形状为[]
print(X.sum())
# 广播机制可以通过复制元素来扩展一个或两个张量，转换后将有相同的形状，然后对其进行基本算术运算，广播机制适用的条件是，1、两个张量的维度相等，但形状不同，不同的那个维度值为1
a = torch.arange(4).reshape((4, 1))
b = torch.arange(4).reshape((1, 4))
print(a)
print(b)
# 按照大的维度算，进行复制，如果是最里层的为1，则将元素值直接复制一份，即原来只有a[]...[][0]，复制之后变成出现多个a[]...[][0]、a[]...[][1]、...，如果是其他维度则以该维度量级为单位做复制
print(a + b)

# 索引和切片
print(X[-1])
print(X[1: 3])
# 指定索引将元素写入矩阵
X[1, 2] = 9
print(X)
# 为多个元素赋相同的值
X[0: 2, :] = 12
print(X)

# 算术运算非原地操作
before = id(Y)
Y = Y + X
print(id(Y) == before)
# 利用切片赋值的方式将算术运算转为原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
# 或利用+=的方式转换
before = id(Y)
Y += X
print(before == id(Y))
# 将Tensor转换为numpy对象，将numpy对象转换为Tensor对象
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))
# 将大小为1的张量转换为python表量。使用item()方法或python的内置函数
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))

