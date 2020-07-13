""" automatic differenatiation """


import torch

x = torch.randn(3, requires_grad=True)

y = x*2

while y.data.norm() < 1e3:
    y = y*2

print(y)


v = torch.tensor([0.1, 1.0, 0.00001], dtype=torch.float)

y.backward(v)
print(x.grad)
