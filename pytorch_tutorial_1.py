import torch

y = torch.empty(2, 2)
x = torch.ones(2, 2, dtype=torch.float64)
z = torch.tensor([2.5, 0.1])
# print(x.size())
# print(x.dtype())
# print(z)

# create two tensors of random values
x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print(x)
# print(y)

# these both are the same
# z = x + y
# z = torch.add(x, y)
# z = torch.sub(x, y)
# z = x * y
# print(z)

x = torch.rand(5, 3)
# print(x[:, 1])
# print(x[1, 1].item())

x = torch.rand(4, 4)
print(x)
# reshape into different dimension, -1 will infer
y = x.view(8, -1)
# print(y)
# print(y.size())

# convert numpy to torch tensor, the memory are shared
import numpy as np
a = torch.ones(5)
# print(a)
b = a.numpy()
# print(b)
# print(type(b))

a.add_(1)
# print(a)
# print(b)

# opposite operation, memory is shared. If tensor is on GPU
a = np.ones(5)
b = torch.from_numpy(a)

a += 1
# print(a)
# print(b)

if torch.cuda.is_available():
    # print("cuda is available")
    device = torch.device("cuda")
else:
    # print("mps is available")
    device = torch.device("mps")

x = torch.ones(5, device=device)
y = torch.ones(5)
# operation performed on GPU which is must faster
z = x + y

# Numpy will have errors since it only deals with CPU tensors!!!
# you cannot convert a GPU tensor back to numpy

z = z.to("cpu") # back to cpu

# alot of times when a tensor is created, then you see the argument
# requires_grad=True, this means that it will later calculate the gradience
# which means you want to optimize